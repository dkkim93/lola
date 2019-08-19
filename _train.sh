#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard
# rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Export current path
export PYTHONPATH=$PYTHONPATH:$DIR

# Train tf 
print_header "Training network"
cd $DIR

# Begin experiment
python3.6 scripts/run_lola.py \
--no-exact \
--trials 1 \
--lr 1 \
--lr_correctio 1 \
