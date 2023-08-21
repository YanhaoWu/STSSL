import sys
# print(sys.path)
from cProfile import label
import numpy as np
import warnings
import os
from torch.utils.data import Dataset

import torch
import json
from torch.utils.data import Dataset, DataLoader
current_path = os.path.abspath(os.path.dirname(__file__)) 
root_path = os.path.abspath(os.path.join(current_path, '..')) 
sys.path.append(root_path) 

from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *

warnings.filterwarnings('ignore')

class SemanticKITTIDataLoader(Dataset):
    def __init__(self, root, seq, split='train', intensity_channel=True, clusterd_flag=False):
        self.n_clusters = 50 # seg 50
        self.clusterd_flag = clusterd_flag  # 是否使用已经聚类过的数据
        self.root = root
        self.seq_ids = {}
        self.seq_ids['train'] = [seq]
        self.split = split
        self.intensity_channel = intensity_channel
        self.path_all = []
        if clusterd_flag:
            self.segments_datapath = []
            self.segments_datapath_list(split)
            # print('The size of %s data is %d'%(split,len(self.segments_datapath)))
            self.path_all = self.segments_datapath
        else:
            self.datapath_list(split)
            # print('The size of %s data is %d'%(split,len(self.points_datapath)))
            self.path_all = self.points_datapath

        self.up_bound = [20, 20, 20]
        self.down_bound = [-20, -20, -20]




    def segments_datapath_list(self, split):
        
        for seq in self.seq_ids[split]:
            segments_seq_path = os.path.join(self.root, seq)
            segments_point_seq_bin = os.listdir(segments_seq_path)
            segments_point_seq_bin.sort(key=lambda x:int(x[:-4]))
            self.segments_datapath += [ os.path.join(segments_seq_path, point_file) for point_file in segments_point_seq_bin ]

        # print("finish listing segments datapath")

    def datapath_list(self, split):
        # 将所有的split中的bin文件都排列起来，从00到21
        # 所以只要有index，拿到points_datapath[index]就可以取出来对应的东西了
        self.points_datapath = []
        self.labels_datapath = []

        for seq in self.seq_ids[split]:
            if self.clusterd_flag:
                 point_seq_path = os.path.join(self.root, seq)
            else:
                point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort(key=lambda x:int(x[:-4]))
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            try:
                label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
                point_seq_label = os.listdir(label_seq_path)
                point_seq_label.sort()
                self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
            except:
                pass
        # print("finish datapath_list")


    # def get_labels(self):


    def __len__(self):
        return len(self.path_all)



    def tracking_item(self, index):
        
        try:
            points_set = np.load(self.segments_datapath[index])
            points_set = points_set.reshape((-1, 6))
            points_set, close = self.crop_pc(points=points_set.T, up_bound=self.up_bound, down_bound=self.down_bound)
            return points_set[:,0:5], points_set[:,-1]
        except:
            return None, None


    def crop_pc(self, points, up_bound, down_bound, scale=1.0, return_mask=False):
        """
        crop the pc using the box in the axis-aligned manner
        """
        maxi = up_bound
        mini = down_bound

        x_filt_max = points[0, :] < up_bound[0]
        x_filt_min = points[0, :] > down_bound[0]
        y_filt_max = points[1, :] < up_bound[1]
        y_filt_min = points[1, :] > down_bound[1]
        z_filt_max = points[2, :] < up_bound[2]
        z_filt_min = points[2, :] > down_bound[2]

        close = np.logical_and(x_filt_min, x_filt_max)
        close = np.logical_and(close, y_filt_min)
        close = np.logical_and(close, y_filt_max)
        close = np.logical_and(close, z_filt_min)
        close = np.logical_and(close, z_filt_max)

        return points[:, close].T, close


    def cluster_points(self, index, save_path):
        save_dir = save_path + str(self.seq_ids['train'][0])
        save_index_dir = save_dir + '/' + str(index) + '.npy'

        if not os.path.isfile(save_index_dir):
          points_set = np.fromfile(self.points_datapath[index], dtype=np.float32)
          points_set = points_set.reshape((-1, 4))
          labels = np.fromfile(self.labels_datapath[index], dtype=np.uint32)
          labels = labels.reshape((-1))
          labels = labels & 0xFFFF
          labels = np.vectorize(learning_map.get)(labels)
          labels = np.expand_dims(labels, axis=-1)
          points_set = clusterize_pcd(points_set, self.n_clusters)
          if not(os.path.isdir(save_dir)):
              os.makedirs(save_dir)
              print("making," , save_dir)
          save_index_dir = save_dir + '/' + str(index) + '.npy'
          np.save(save_index_dir,  np.concatenate((points_set, labels.astype(np.int32)),axis=1))
          return points_set, labels.astype(np.int32)
        










# if __name__ == '__main__':
#     a = SemanticKITTIDataLoader(root=r'C:\Users\Wuyanhao\PycharmProjects\BAT\segcontrast-main\SegContrast_Mutil_view')
#     data_load = DataLoader(dataset=a, batch_size=5, shuffle=False)
#     for step, (batch_x, batch_y) in enumerate(data_load):
#          a = 1