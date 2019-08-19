"""
PG training for the Iterated Prisoner's Dilemma and Matching Pennies.
"""
import numpy as np
import tensorflow as tf
from . import logger
from .corrections import *
from .networks import *
from .utils import *

SUMMARYLENGTH = 10  # Number of episodes to periodically save for analysis


def update_policy(mainQN, lr, final_delta_1_v, final_delta_2_v):
    mainQN[0].setparams(mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    mainQN[1].setparams(mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def log_performance(gamma_discount, reward_list, i_episode):
    log_items = {}
    log_items['i_episode'] = i_episode
    log_items['reward_agent0'] = np.mean(reward_list[-SUMMARYLENGTH:], 0)[0] / gamma_discount
    log_items['reward_agent1'] = np.mean(reward_list[-SUMMARYLENGTH:], 0)[1] / gamma_discount

    for key in sorted(log_items.keys()):
        logger.record_tabular(key, log_items[key])
    logger.dump_tabular()
    logger.info('')


def collect_one_trajectory(env, mainQN, gamma, trace_length, sess):
    episodeBuffer = [[] for _ in range(env.NUM_AGENTS)]
    rewards_episode = np.zeros((env.NUM_AGENTS))
    state = env.reset()
    timestep = 0

    while timestep < trace_length + 1:
        timestep += 1

        # Get action
        actions = []
        for i_agent in range(env.NUM_AGENTS):
            a = sess.run(
                [mainQN[i_agent].predict],
                feed_dict={mainQN[i_agent].scalarInput: [state[i_agent]]})
            actions.append(a[0])

        # Take step in the environment
        next_state, rewards, done = env.step(actions)

        # Add to experience
        for i_agent in range(env.NUM_AGENTS):
            episodeBuffer[i_agent].append([
                state[0], actions[i_agent], rewards[i_agent], 
                next_state[0], done, i_agent])

        # For next timestep
        state = next_state

        rewards_episode += [rewards[i_agent] * gamma**(timestep - 1) for i_agent in range(2)]

        if done:
            break

    return episodeBuffer, rewards_episode


def train(env, *, num_episodes, trace_length, batch_size, gamma,
          lr, lr_correction, corrections, simple_net, hidden):
    tf.reset_default_graph()

    # Q-networks
    mainQN = []
    for i_agent in range(env.NUM_AGENTS):
        mainQN.append(Qnetwork(
            'main' + str(i_agent), 
            i_agent, env, lr_correction=lr_correction, 
            gamma=gamma, batch_size=batch_size, trace_length=trace_length, 
            hidden=hidden, simple_net=simple_net))

    # Corrections
    corrections_func(
        mainQN,
        batch_size=batch_size, trace_length=trace_length,
        corrections=corrections, cube=None)

    # Initialize buffer
    buffers = [ExperienceBuffer(batch_size) for _ in range(env.NUM_AGENTS)]

    # Misc
    pow_series = np.arange(trace_length)
    discount = np.expand_dims(np.array([pow(gamma, item) for item in pow_series]), 0)
    discount_array = np.reshape(gamma**trace_length / discount[0, :], [1, -1])
    gamma_discount = 1. / (1. - gamma)
    reward_list = []

    # Start training
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i_episode in range(num_episodes):
            # Collect one trajectory experience
            episodeBuffer, rewards_episode = collect_one_trajectory(
                env, mainQN, gamma, trace_length, sess)

            # Add the episode to the experience buffer
            for i_agent in range(env.NUM_AGENTS):
                buffers[i_agent].add(np.array(episodeBuffer[i_agent]))
            reward_list.append(rewards_episode)

            if (i_episode + 1) % batch_size == 0:
                # Update policy
                trainBatch0 = buffers[0].sample(batch_size, trace_length)
                trainBatch1 = buffers[1].sample(batch_size, trace_length)

                sample_return0 = np.reshape(
                    get_monte_carlo(trainBatch0[:, 2], gamma, trace_length, batch_size), [batch_size, -1])
                sample_return1 = np.reshape(
                    get_monte_carlo(trainBatch1[:, 2], gamma, trace_length, batch_size), [batch_size, -1])

                sample_reward0 = np.reshape(trainBatch0[:, 2] - np.mean(trainBatch0[:, 2]), [-1, trace_length]) * discount
                sample_reward1 = np.reshape(trainBatch1[:, 2] - np.mean(trainBatch1[:, 2]), [-1, trace_length]) * discount

                last_state = np.reshape(np.vstack(trainBatch0[:, 3]), [-1, trace_length, env.NUM_STATES])[:, -1, :]

                value_0_next, value_1_next = sess.run(
                    [mainQN[0].value, mainQN[1].value],
                    feed_dict={
                        mainQN[0].scalarInput: last_state,
                        mainQN[1].scalarInput: last_state})
                fetches = [
                    mainQN[0].values,
                    mainQN[0].updateModel,
                    mainQN[1].updateModel,
                    mainQN[0].delta, mainQN[1].delta,
                    mainQN[0].grad,
                    mainQN[1].grad,
                    mainQN[0].v_0_grad_01,
                    mainQN[1].v_1_grad_10]
                feed_dict = {
                    mainQN[0].scalarInput: np.vstack(trainBatch0[:, 0]),
                    mainQN[0].sample_return: sample_return0,
                    mainQN[0].actions: trainBatch0[:, 1],
                    mainQN[1].scalarInput: np.vstack(trainBatch1[:, 0]),
                    mainQN[1].sample_return: sample_return1,
                    mainQN[1].actions: trainBatch1[:, 1],
                    mainQN[0].sample_reward: sample_reward0,
                    mainQN[1].sample_reward: sample_reward1,
                    mainQN[0].next_value: value_0_next,
                    mainQN[1].next_value: value_1_next,
                    mainQN[0].gamma_array: discount,
                    mainQN[1].gamma_array: discount,
                    mainQN[0].gamma_array_inverse: discount_array,
                    mainQN[1].gamma_array_inverse: discount_array}

                values, _, _, update1, update2, grad_1, grad_2, v0_grad_01, v1_grad_10 = \
                    sess.run(fetches, feed_dict=feed_dict)
                update_policy(mainQN, lr, update1, update2)

                # Log performance
                log_performance(gamma_discount, reward_list, i_episode + 1)
