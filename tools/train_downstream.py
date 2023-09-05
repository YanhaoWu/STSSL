# 2022.7.20 WuYanhao XJTU
# STSSL 

import os
import numpy as np
import argparse
import subprocess
import sys
import time
from generate import *

parser = argparse.ArgumentParser(description='STSSL_Params')
parser.add_argument('--max_epoch', type=int, default=300, help='max epochs to run')
parser.add_argument('--save_index', type=int, default=0, help='where to save the pre-trained models')
parser.add_argument('--dataset_path', type=str, default='/home/ssd/3d', help='KITTI save path')
parser.add_argument('--save_path_base', type=str, default='/home/WuYanhao/WorkSpace/STSSL/fine_tune_save/', help='where to save to models and logs')
parser.add_argument('--load_path', type=str, default='/home/WuYanhao/WorkSpace/STSSL_Save/model_save/0/stage2/epoch39_model.pt', help='where to save to tracking results')
parser.add_argument('--percentages', type=float, default=0.001, help='the percentage of labels to train')

args = parser.parse_args()
current_path = os.path.abspath(os.path.dirname(__file__)) 
root_path = os.path.abspath(os.path.join(current_path, '..')) 
args.train_command = root_path + '/downstream_train.py'
last_epoch_name = 'epoch' + str(args.max_epoch - 1)
# last_epoch_name = 'epoch0'

result_all = []
# fine-tune
result = subprocess.Popen(['python', str(args.train_command), '--epochs', str(args.max_epoch), '--load_path', str(args.load_path), '--data_dir', str(args.dataset_path), '--percentage_labels', str(args.percentages), '--log_dir', str(args.save_path_base)])
result.wait()

# eval
args.train_command = root_path + '/inference.py'
result = subprocess.run(['python', str(args.train_command), '--load_path', str(args.save_path_base), '--data_dir', str(args.dataset_path), '--best', str(last_epoch_name)], check=True)
