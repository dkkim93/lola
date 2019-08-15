"""
Trains LOLA on IPD or MatchingPennies with exact value functions.
Note: Interfaces are a little different form the code that estimates values,
hence moved into a separate module.
"""
import numpy as np
import tensorflow as tf
from . import logger
from .utils import *


class Qnetwork:
    """
    Q-network that is either a look-up table or an MLP with 1 hidden layer.
    """
    def __init__(self, myScope, num_hidden, simple_net=True):
        with tf.variable_scope(myScope):
            if simple_net:
                self.p_act = tf.Variable(tf.random_normal([5, 1]))
            else:
                self.input_place = tf.placeholder(shape=[5], dtype=tf.int32)
                act = tf.nn.tanh(
                    layers.fully_connected(
                        tf.one_hot(self.input_place, 5, dtype=tf.float32),
                        num_outputs=num_hidden, activation_fn=None))
                self.p_act = layers.fully_connected(
                    act, num_outputs=1, activation_fn=None)

        self.parameters = []
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=myScope):
            self.parameters.append(i)
        self.setparams = SetFromFlat(self.parameters)
        self.getparams = GetFlat(self.parameters)


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    mainQN[0].setparams(mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v))
    mainQN[1].setparams(mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def corrections_func(mainQN, corrections, gamma, pseudo, reg):
    # Set placeholder
    mainQN[0].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)
    mainQN[1].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)

    mainQN[0].reward_func = tf.placeholder(shape=[4, 1], dtype=tf.float32)
    mainQN[1].reward_func = tf.placeholder(shape=[4, 1], dtype=tf.float32)

    # Extract all five parameters
    theta0_all = mainQN[0].p_act
    theta1_all = mainQN[1].p_act

    # Extract the initial parameter and its action
    theta0_initial = tf.slice(theta0_all, [4, 0], [1, 1])
    theta1_initial = tf.slice(theta1_all, [4, 0], [1, 1])

    p0_initial = tf.nn.sigmoid(theta0_initial)
    p1_initial = tf.nn.sigmoid(theta1_initial)

    # Construct initial state (p0 in Appendix)
    p0_initial_vector = tf.concat([p0_initial, (1 - p0_initial)], 0)
    p1_initial_vector = tf.concat([p1_initial, (1 - p1_initial)], 0)

    initial_state = tf.reshape(tf.matmul(p0_initial_vector, tf.transpose(p1_initial_vector)), [-1, 1])

    # Extract the four parameters and their actions
    theta0 = tf.slice(theta0_all, [0, 0], [4, 1])
    theta1 = tf.slice(theta1_all, [0, 0], [4, 1])

    p0 = tf.nn.sigmoid(theta0)
    p1 = tf.nn.sigmoid(theta1)

    # Construct transition matrix P in Appendix
    P = tf.concat(
        values=[
            tf.multiply(p0, p1),
            tf.multiply(p0, 1 - p1),
            tf.multiply(1 - p0, p1),
            tf.multiply(1 - p0, 1 - p1)], 
        axis=1)

    # Compute value (V1 and V2 in Appendix)
    I_m_P = tf.diag([1.0, 1.0, 1.0, 1.0]) - P * gamma

    naive_value0 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), mainQN[0].reward_func), 
        initial_state,
        transpose_a=True)
    naive_value1 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), mainQN[1].reward_func), 
        initial_state,
        transpose_a=True)

    # Apply regularization if needed
    if reg > 0:
        for indx, _ in enumerate(mainQN[0].parameters):
            naive_value0 -= reg * tf.reduce_sum(tf.nn.l2_loss(tf.square(mainQN[0].parameters[indx])))
            naive_value1 -= reg * tf.reduce_sum(tf.nn.l2_loss(tf.square(mainQN[1].parameters[indx])))

    # Get second order correction
    grad_naive_value0_wrt_agent0 = flatgrad(naive_value0, mainQN[0].parameters)
    grad_naive_value0_wrt_agent1 = flatgrad(naive_value0, mainQN[1].parameters)

    grad_naive_value1_wrt_agent0 = flatgrad(naive_value1, mainQN[0].parameters)
    grad_naive_value1_wrt_agent1 = flatgrad(naive_value1, mainQN[1].parameters)

    param_len = grad_naive_value0_wrt_agent0.get_shape()[0].value

    # pseudo is default to be False. The only difference is to apply stop_grad or not
    if pseudo:
        multiply0 = tf.matmul(
            tf.reshape(grad_naive_value0_wrt_agent1, [1, param_len]),
            tf.reshape(grad_naive_value1_wrt_agent1, [param_len, 1]))
        multiply1 = tf.matmul(
            tf.reshape(grad_naive_value1_wrt_agent0, [1, param_len]),
            tf.reshape(grad_naive_value0_wrt_agent0, [param_len, 1]))
    else:
        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(grad_naive_value0_wrt_agent1), [1, param_len]),
            tf.reshape(grad_naive_value1_wrt_agent1, [param_len, 1]))
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(grad_naive_value1_wrt_agent0), [1, param_len]),
            tf.reshape(grad_naive_value0_wrt_agent0, [param_len, 1]))

    second_order0 = flatgrad(multiply0, mainQN[0].parameters)
    second_order1 = flatgrad(multiply1, mainQN[1].parameters)

    mainQN[0].v = naive_value0
    mainQN[1].v = naive_value1
    mainQN[0].delta = grad_naive_value0_wrt_agent0
    mainQN[1].delta = grad_naive_value1_wrt_agent1
    mainQN[0].delta += tf.multiply(second_order0, mainQN[0].lr_correction)
    mainQN[1].delta += tf.multiply(second_order1, mainQN[1].lr_correction)


def train(env, *, num_episodes=50, trace_length=200,
          simple_net=True, corrections=True, pseudo=False,
          num_hidden=10, reg=0.0, lr=1., lr_correction=0.5, gamma=0.96):
    logger.reset()

    # Get info about the env
    payout_mat_1 = np.reshape(env.payout_mat.T, [-1, 1])
    payout_mat_2 = np.reshape(env.payout_mat, [-1, 1])

    # Sanity
    tf.reset_default_graph()

    # Q-networks
    mainQN = [
        Qnetwork('main' + str(agent_id), num_hidden, simple_net) 
        for agent_id in range(2)]

    # Corrections
    corrections_func(mainQN, corrections, gamma, pseudo, reg)

    results = []
    norm = 1. / (1. - gamma)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        lr_coor = np.ones(1) * lr_correction

        for episode in range(num_episodes):
            sess.run(init)

            log_items = {}
            log_items['episode'] = episode + 1

            res, params_time, delta_time = [], [], []
            for i in range(trace_length):
                params0 = mainQN[0].getparams()
                params1 = mainQN[1].getparams()
                outputs = [mainQN[0].delta, mainQN[1].delta, mainQN[0].v, mainQN[1].v]

                update1, update2, v1, v2 = sess.run(
                    outputs,
                    feed_dict={
                        mainQN[0].reward_func: payout_mat_1,
                        mainQN[1].reward_func: payout_mat_2,
                        mainQN[0].lr_correction: lr_coor,
                        mainQN[1].lr_correction: lr_coor})
                update(mainQN, lr, update1, update2)
                params_time.append([params0, params1])
                delta_time.append([update1, update2])

                log_items['ret1'] = v1[0][0] / norm
                log_items['ret2'] = v2[0][0] / norm
                res.append([v1[0][0] / norm, v2[0][0] / norm])
            results.append(res)

            for k, v in sorted(log_items.items()):
                logger.record_tabular(k, v)
            logger.dump_tabular()
            logger.info('')

    result0, result1 = [], []
    for i_episode in range(num_episodes):
        result0.append(results[i_episode][-1][0])
        result1.append(results[i_episode][-1][1])

    print("mean: {:.5f}, std: {:.5f}".format(np.mean(result0), np.std(result0)))
    print("mean: {:.5f}, std: {:.5f}".format(np.mean(result1), np.std(result1)))

    return results
