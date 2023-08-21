import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
import copy
warnings.filterwarnings('ignore')


class SemanticKITTIDataLoader(Dataset):
    def __init__(self, split='train',  resolution=0.05, intensity_channel=False, args=None):
        self.args = args
        self.n_clusters = 50
        self.resolution = resolution
        self.intensity_channel = intensity_channel
        self.track_pro = args.track_pro
        self.seq_ids = {}
        print(split)
        # self.seq_ids['train'] = [ '01'] # for debuging 
        # print("for debugging, only using seq 01")
        self.seq_ids['train'] = [ '00', '01', '02', '03', '04', '05', '06', '07', '09', '10' ]
        self.seq_ids['validation'] = ['08']
        self.pre_index = 0 #
        self.split = split
        self.segments_datapath = []
        self.track_datapath = []
        self.number_seq = []    # 
        self.info_datapath = []
        self.up_bound = [20, 20, 20]
        self.down_bound = [-20, -20, -20]
        assert (split == 'train' or split == 'validation')
        self.pro_rate = 1
        self.index_plus_change_get = 0
        self.segments_datapath_list(split)
        if self.args.stage == 1:
            self.tracking_datapath_list(split)

        print('The size of %s tracking_data is %d'%(split,len(self.track_datapath)))
        print('The size of %s cluster_data is %d'%(split,len(self.segments_datapath)))
        


    def segments_datapath_list(self, split):
        
        for seq in self.seq_ids[split]:
            segments_seq_path = os.path.join(self.args.segment_pathbase, seq)
            segments_point_seq_bin = os.listdir(segments_seq_path)
            segments_point_seq_bin.sort(key=lambda x:int(x[:-4]))
            self.segments_datapath += [ os.path.join(segments_seq_path, point_file) for point_file in segments_point_seq_bin ]

        print("finish listing segments datapath")


    def tracking_datapath_list(self, split):
        self.number_seq = []      
        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.args.tracking_pathbase, 'points', seq)
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort(key=lambda x:int(x[:-4]))
            self.track_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]
            if self.number_seq == []:
                last_id = 0
            else:
                last_id = self.number_seq[-1]
            self.number_seq.append(last_id + len(point_seq_bin))

        print("finish listing track datapath")


    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])  #
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3]) #
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])   # 
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3]) # 
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])  # 
        points = random_drop_n_cuboids(points)   # 

        return np.squeeze(points, axis=0)


    def __len__(self):
        return len(self.segments_datapath) 

    def __getitem__(self, index):

        temp_index = index
        
        if self.args.stage == 1:
            single_frame_flag = False   
            # since we list pointclouds from different scenes, we need to separate them
            for i in range(len(self.number_seq)):
                if temp_index < self.number_seq[i]:
                    seq_idx = i
                    break
            index_plus = self.index_plus_change_get

            if (temp_index + index_plus) < self.number_seq[seq_idx]:                                 
                track_index = temp_index + index_plus
            else:
                track_index = temp_index
                
            data_path1 = self.track_datapath[temp_index]
            data_path2 = self.track_datapath[track_index]
        else:
            
            single_frame_flag = True   
            data_path = self.segments_datapath[temp_index]


        if single_frame_flag:  # intra-frame match       
                    
            points_set = np.load(data_path).reshape(-1, 6)
            points_set = np.concatenate((points_set[:,:4], points_set[:,-2].reshape(-1,1)),axis=1) 
            
            points_set_save = copy.deepcopy(points_set)
            points_i = random_cuboid_point_cloud(points_set.copy())
            points_i = self.transforms(points_i)

            points_j = random_cuboid_point_cloud(points_set_save.copy())
            points_j = self.transforms(points_j)

            if not self.intensity_channel:
                points_i = points_i[:, :3]
                points_j = points_j[:, :3]
            return points_i, points_j    
                
        else: # inter-frame match

            points_set = np.load(data_path1).reshape(-1, 7)
            points_set = np.concatenate((points_set[:,:4], points_set[:,-2].reshape(-1,1)),axis=1)      # 

            track_points_set = np.load(data_path2).reshape(-1, 7)
            track_points_set = np.concatenate((track_points_set[:,:4], track_points_set[:,-2].reshape(-1,1)),axis=1)


            points_i = random_cuboid_point_cloud(points_set.copy())
            points_i = self.transforms(points_i)

            points_k = random_cuboid_point_cloud(points_set.copy())
            points_k = self.transforms(points_k)   

            points_j = random_cuboid_point_cloud(track_points_set.copy())
            points_j = self.transforms(points_j)

            if not self.intensity_channel:
                points_i = points_i[:, :3]
                points_j = points_j[:, :3]
                points_k = points_k[:, :3]
            return points_i, points_j, points_k


        
            
