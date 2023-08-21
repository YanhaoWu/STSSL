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
parser.add_argument('--max_epoch', type=int, default=200, help='max epochs to run')
parser.add_argument('--save_index', type=int, default=0, help='where to save the pre-trained models')
parser.add_argument('--dataset_path', type=str, default='/home/ssd/3d', help='KITTI save path')
parser.add_argument('--save_path_base', type=str, default='/home/WuYanhao/WorkSpace/STSSL/model_save/', help='where to save to models and logs')
parser.add_argument('--segment_save_path', type=str, default='/home/ssd/3d/STSSL_Segments/', help='where to save segmented scenes')
parser.add_argument('--track_save_path', type=str, default='/home/ssd/3d/STSSL_Tracks/', help='where to save to tracking results')
args = parser.parse_args()

current_path = os.path.abspath(os.path.dirname(__file__)) 
root_path = os.path.abspath(os.path.join(current_path, '..')) 
args.track_command = root_path + '/data_preprocessing/tracking.py'
args.segment_command = root_path + '/data_preprocessing/clustering.py'
args.train_command = root_path + '/train.py'
print("clustering part is saved at ", args.segment_command)
print("tracking part is saved at ", args.track_command)


save_index = args.save_index
max_epoch = args.max_epoch
stop_epoch = 160
segment_save_path = args.segment_save_path
track_save_path = args.track_save_path
use_previous_stage = None # None / int(args)
save_index_cmd = '--save_index ' + str(save_index) + ' '
oldstdout = sys.stdout
model_save_base = args.save_path_base + str(save_index) 
model_save_dir = model_save_base + '/stage1' 
if not(os.path.isdir(model_save_dir)):
    print("making dir", model_save_dir)
    os.makedirs(model_save_dir)
txt_path = model_save_base + '/' + 'log.txt'
segment_path = model_save_base + '/' + 'seg_log.txt'
train_path = model_save_base + '/' + 'train_path_log.txt'
track_path = model_save_base + '/' + 'track.txt'

print("---------------Start training----------------")
print("all logs can be found in ", txt_path, 'not the terminal')
file = open(txt_path ,'a')
segment_file = open(segment_path ,'a')
train_file = open(train_path ,'a')
box_file = open(track_path ,'a')
track_file = open(track_path, 'a')
sys.stdout = file
print("use_previous_stage ", use_previous_stage )
pre_stop_epoch = 0
  
# For Generating Clusters 
print("Start generating segments for scenes, it may take a few minutes to dozens of minutes")
file.flush()
generate_obs(args.segment_command, args.dataset_path, segment_save_path, track_save_path, 'None', oldstdout)
print("Finish generating segments")
file.flush()

cmd = 'python ' + args.train_command + ' '
stop_epoch_cmd = ' --stop_epoch ' + str(stop_epoch)
pre_stop_epoch_cmd = ' --pre_stop_epoch ' + str(pre_stop_epoch)
stage = '--stage ' + str(0)
cmd = cmd + save_index_cmd + stage + stop_epoch_cmd + pre_stop_epoch_cmd
print(time.strftime('%Y-%m-%d %H:%M:%S')) #结构化输出当前的时间
print(cmd)
print("starting pretraining")
file.flush()

# 执行Pretraining
result = subprocess.run(['python', str(args.train_command), '--stage', str(0), '--stop_epoch', str(int(stop_epoch)), '--model_save_dir', str(model_save_dir), '--segment_pathbase', segment_save_path],  stdout = train_file, check=True)
pre_stop_epoch = stop_epoch # 后面再更新
 

# For Generating Tracks

models_backbone_path = model_save_dir + '/epoch' + str(stop_epoch - 1) + '_model.pt'
models_head_path = model_save_dir + '/epoch' + str(stop_epoch - 1) + '_k_model_head.pt' 
model_path = [models_backbone_path, models_head_path]
# model_path应该包含Backbone 和 Project heads
print("Start generating tracks for scenes")
generate_obs(args.track_command, args.dataset_path, segment_save_path, track_save_path, model_path, track_file)
print("Finish generating tracks")
stop_epoch = max_epoch - stop_epoch

print("inter-frame training")
model_save_dir = model_save_base + '/stage2' 
result = subprocess.run(['python', str(args.train_command), '--loading_dir', models_backbone_path, '--stage', str(1), '--stop_epoch', str(int(stop_epoch)), '--model_save_dir', str(model_save_dir), '--segment_pathbase', segment_save_path, '--tracking_pathbase', track_save_path],  stdout = train_file, check=True)
pre_stop_epoch = stop_epoch # 后面再更新



file.close()
box_file.close()
segment_file.close()
train_file.close()
track_file.close()
sys.stdout = oldstdout
