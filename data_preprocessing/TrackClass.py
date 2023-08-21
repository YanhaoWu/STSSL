# 2022.5.17
# by wyh
# if any question, be free to contact me with wuyanhao@stu.xjtu.edu.cn  
# : )

from math import fabs
import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
import torch
from scipy.optimize import linear_sum_assignment
import copy
from track_model import model_simi_class
import sys




class TrackDataClass(object): # baseclass for saving and loading a cluster
    def __init__(self):

        self.id = None        # unique id for a segment / cluster
        
        self.start_frame = 0  # which frame the cluser frist appears

        self.end_frame = 0    # which frame the clsuter disapper

        self.index = None     # N * M * 3

        self.center = None    # N * 1 * 3

        self.number_list = []  # 


    def first_deal(self, frame, index, center):     # create a cluster instance

        self.start_frame = frame  # 

        self.index = index

        self.center = center.reshape(-1, 3)

        self.number_list.append(index.shape[0])


    def add_new_data(self, frame, index, center):

        center = center.reshape(-1, 3)

        self.end_frame = frame                     

        self.index = np.concatenate((self.index, index), axis=0)

        self.center = np.concatenate((self.center, center.reshape(-1, 3)), axis=0)

        self.number_list.append(index.shape[0])    


    def set_data(self, idx, start_frame, end_frame,  index, center):

        self.id = idx     

        self.start_frame = start_frame  

        self.end_frame = end_frame    

        self.index = index    

        self.center = center  



class Track(): #  baseclass for tracking clusters
    def __init__(self, seq, model_path, save_path, gpu_use='0'):

        self.seq = seq                      

        self.model_feature_pre = None 
       
        self.cluster_points_center_list_pre = None

        self.cluster_points_number_list_pre = None

        self.model_feature = None

        self.cluster_points_center_list = None

        self.cluster_points_number_list = None

        self.show_part_cluster = False

        self.cluster_method = 'DBSCAN'  # 'DBSCAN' or other methods

        self.match_threshold = 5

        self.id_count = 0

        self.id_count_max = 0            # id count

        self.id_mapping = {}             # 
        
        self.id_det_mapping = {}         # 

        self.id_tck_mapping = {}

        self.list_tck_index = []

        self.list_det_index = []

        self.done_list = []

        self.dealing_list = []

        self.save_dict = {}    

        self.batch_id = 0

        self.frame_id = 0     

        self.outlier_cloud = None

        self.label_matched = None   # 

        self.label_global = None    # 

        self.multi_view_map = np.zeros((700000, 1))   #

        self.frame_index = []       # 

        self.start_index = 0
      
        self.save_frame_info_path = save_path + 'info/'

        self.save_frame_points_path = save_path + 'points/'

        self.model = model_simi_class(model_path, gpu_use)       # the model to computer similarty between clusters


       
        if not os.path.isdir(os.path.join(self.save_frame_points_path, self.seq)):
            os.makedirs(os.path.join(self.save_frame_points_path, self.seq))

        if not os.path.isdir(os.path.join(self.save_frame_info_path)):
            os.makedirs(os.path.join(self.save_frame_info_path))
            
        model_save_dir = self.save_frame_info_path
        txt_path = model_save_dir  + 'param.txt'
        file = open(txt_path ,'a')
        oldstdout = sys.stdout
        sys.stdout = file
        print("the model_path we use is ", model_path)
        file.close()
        sys.stdout = oldstdout


    def show_cluster_points(self, points):
        # show clusters using open3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='part_cluster', width=800, height=600)
        vis.add_geometry(points)
        vis.run()





    def statistics_different_cluster(self, points, labels):
        # get points and computer centers of a clusters
        # in:
        # points: PointCloud
        # labels: cluster label
        # out:
        # center of clusters
        # number of points in each cluster
        cluster_points_center_list = []
        cluster_points_number_list = []
        for i in range(labels.max()):
            temp = np.where(labels == i)
            cluster_points = points.select_by_index(list(temp[0]), invert=False)
            cluster_points_center = cluster_points.get_center()
            cluster_points_number = np.array(cluster_points.points).shape[0]
            cluster_points_center_list.append([cluster_points_center])
            cluster_points_number_list.append([cluster_points_number])
        return cluster_points_center_list, cluster_points_number_list




    def id_manage(self, fisrt=False, pre_none=False, labels_global=None):    # 
    # manage clusters in scenes
    # assign cluster id

        if fisrt:     # first time to assign id for each cluster

            size = np.array(self.cluster_points_center_list).reshape(-1, 3).shape[0]   


            for i in range(size):

                self.id_mapping[str(i)] = i

                self.id_count_max = self.id_count_max + 1                                # 

        elif pre_none:      # the previous one frame is none

            size = np.array(self.cluster_points_center_list).reshape(-1, 3).shape[0]    #

            labels_global_copy = copy.deepcopy(labels_global)

            for i in range(size):       


                labels_global[labels_global_copy==i] = self.id_count_max

                self.id_mapping[str(i)] = self.id_count_max

                self.id_count_max = self.id_count_max + 1


            return labels_global

        else:
            
            model_feature_now = self.model_feature

            model_feature_pre = self.model_feature_pre

            center_now = np.array(self.cluster_points_center_list).reshape(-1, 3)

            center_pre = np.array(self.cluster_points_center_list_pre).reshape(-1, 3)

            space_dist_mat = cdist(center_now, center_pre, metric='euclidean')

            space_dist_reverse_mat = cdist(center_pre, center_now, metric='euclidean')

            sort_dist = np.sort(space_dist_mat, axis=1)          

            sort_dist_reverse = np.sort(space_dist_reverse_mat, axis=1)          

            center_now_filter = copy.deepcopy(center_now)

            center_pre_filter = copy.deepcopy(center_pre)

            born_index = [i for i in np.arange(0, sort_dist.shape[0])]

            death_index = [i for i in np.arange(0, sort_dist_reverse.shape[0])]

            for i in range(sort_dist.shape[0]):
                self.list_det_index.append(i)                                                    # 
                if sort_dist[i][0] > self.match_threshold:                                       # if two cluster are too distant in two frames, we do not  match them 
                    born_index.remove(i)
                    self.id_det_mapping[str(i)] = self.id_count_max                              # 
                    self.list_det_index.remove(i)                                                # 
            center_now_filter = center_now_filter[born_index]

            model_feature_now = model_feature_now[born_index]                                    # keep the remain features

            for i in range(sort_dist_reverse.shape[0]):
                self.list_tck_index.append(i)
                if sort_dist_reverse[i][0] > self.match_threshold:
                    death_index.remove(i)                                                         # 
                    self.list_tck_index.remove(i)                                                 # 

            center_pre_filter = center_pre_filter[death_index]

            model_feature_pre = model_feature_pre[death_index]

            return center_now_filter, center_pre_filter, model_feature_now, model_feature_pre, born_index, death_index                   




    def life_manage(self, center_now_filter, center_pre_filter, feature_now_filter, feature_pre_filter):  
        # match cluster using hungarian algorithm
        # out:
        #   matched_index, match_index : index of cluster in two frames 
        #   
   
        space_dist_filter_mat = cdist(center_now_filter, center_pre_filter, metric='euclidean')     
        feature_dist_mat = cdist(feature_now_filter, feature_pre_filter, metric='euclidean')
        
        if space_dist_filter_mat.shape == (1,1):    
            matched_index = np.array([0])
            match_index = np.array([0])
        elif space_dist_filter_mat.shape[0]==0:
            matched_index = np.array([]).astype(np.int32)
            match_index = np.array([]).astype(np.int32)
        else:
            normal_space_dist_mat = (space_dist_filter_mat - np.min(space_dist_filter_mat))/(np.max(space_dist_filter_mat) - np.min(space_dist_filter_mat))
            normal_feature_dist_mat = (feature_dist_mat - np.min(feature_dist_mat))/(np.max(feature_dist_mat) - np.min(feature_dist_mat))
            dist_mat = normal_space_dist_mat + 0.5 * (normal_feature_dist_mat)      #  compute Similarity using position distances and features distances
            matched_index, match_index = linear_sum_assignment(dist_mat)    # 
        

    
        return matched_index, match_index



    def cluster_iteration(self, matched_index, match_index, labels, labels_all, born_index, death_index): 
        # Modify global subscripts based on matching idx
        # map the global index from previous frames to current frames


        labels_matched = copy.deepcopy(labels)
        labels_all_copy = copy.deepcopy(labels_all)
        temp_id_mapping = {}
        for i in range(labels.max()+1):                                                     # 
            if i in born_index:                                                             # 
                born_arg_i = born_index.index(i)                                            # 
                if born_arg_i in matched_index:                                             #
                    match_arg_i = np.where(matched_index == born_arg_i)[0][0]               # 
                    corres_match_index = match_index[match_arg_i]                           # 
                    match_death_index = death_index[corres_match_index]                     # 
                    absolute_index = self.id_mapping[str(match_death_index)]                # 
                    temp_id_mapping[str(i)] = absolute_index
                    labels_matched[labels == i] = absolute_index
                    labels_all[labels_all_copy == i] = absolute_index

                else:                                                                       # unmatched objects -> new cluster~
                    absolute_index = self.id_count_max
                    labels_matched[labels == i] = absolute_index                            # 
                    labels_all[labels_all_copy == i] = absolute_index
                    temp_id_mapping[str(i)] = absolute_index
                    self.id_count_max = self.id_count_max + 1                               # 

            else:                                                                           # unmatched objects which distant from other cluster -> new cluster~

                absolute_index = self.id_count_max
                labels_matched[labels == i] = absolute_index                                # 
                labels_all[labels_all_copy == i] = absolute_index
                temp_id_mapping[str(i)] = absolute_index
                self.id_count_max = self.id_count_max + 1                                   # 


        self.id_mapping = temp_id_mapping
        return labels_matched, labels_all



    def statistics_and_fliter_cluster(self, points, labels): 
        # Remove some clusters whose number of points does not meet the requirements such as clusters with two litter points

        cluster_points_center_list = []
        cluster_points_number_list = []
        fliter_points = None
        eff_label = None                                                            # 
        eff_cluster_count = 0
        label_add = np.zeros_like(labels) - 1                                       # 
                                                                                    # 
                                                                                    #
        for i in range(labels.max()):                                               # 
            temp = np.where(labels == i)
            cluster_points = points.select_by_index(list(temp[0]), invert=False)
            cluster_points_center = cluster_points.get_center()
            cluster_points_number = np.array(cluster_points.points).shape[0]
            if not(cluster_points_number > 20000 or cluster_points_number < 200):  # 
                if fliter_points is None:
                    fliter_points = np.array(cluster_points.points)
                    eff_label = np.zeros((fliter_points.shape[0], 1))               # 
                    label_add[temp] = eff_cluster_count
                    eff_cluster_count = eff_cluster_count + 1
                else:
                    fliter_points = np.concatenate((fliter_points, np.array(cluster_points.points)))
                    temp_label = np.zeros((np.array(cluster_points.points).shape[0], 1)) + eff_cluster_count
                    eff_label = np.concatenate((eff_label, temp_label))
                    label_add[temp] = eff_cluster_count     # 
                    eff_cluster_count = eff_cluster_count + 1

                cluster_points_center_list.append([cluster_points_center])
                cluster_points_number_list.append([cluster_points_number])
            
        if eff_label is None:       # 
                                    #
            return cluster_points_center_list, cluster_points_number_list, fliter_points, eff_label, label_add     

        else:
            return cluster_points_center_list, cluster_points_number_list, fliter_points, eff_label.flatten().astype(np.int32), label_add
           



    def seg_api(self, index, outlier_cloud, lables, label_anno):

        # Input:
        #  outlier_cloud, (x, y, z, i, n) n is the cluster id
        #  label_anno: annotation, we do not use it in tracking, only for convenient loading of downstream tasks
        # Output:
        #  Outlier_ Cloud output, (x, y, z, n, n', label) n': global index 
        self.frame_id = index - self.start_index    #

        labels = lables.astype(np.int)
        all_points = copy.deepcopy(outlier_cloud)  
        labels_local = copy.deepcopy(labels)
        outlier_cloud = outlier_cloud[:, 0:3]

        outlier_cloud = DataFrame(outlier_cloud)                              #
        outlier_cloud.columns = ['x', 'y', 'z']                               
        outlier_cloud = PyntCloud(outlier_cloud)                              # 
        outlier_cloud = outlier_cloud.to_instance("open3d", mesh=False)       # 

        cluster_points_center_list, cluster_points_number_list, fliter_points, labels_fliter, labels_global = self.statistics_and_fliter_cluster(
            outlier_cloud, labels)  

        self.cluster_points_center_list = cluster_points_center_list

        self.cluster_points_number_list = cluster_points_number_list

        labels_matched = copy.deepcopy(labels_fliter)                                 # 

        if labels_fliter is None:

            self.id_mapping = {}

        else:
            model_point_set = np.concatenate((all_points, labels_global.reshape(-1, 1)), axis=1)

            model_feature = self.model.forward(model_point_set)

            model_feature = model_feature.cpu().numpy()

            self.model_feature = model_feature

            outlier_cloud.points = o3d.utility.Vector3dVector(fliter_points)


            if not ((self.cluster_points_center_list_pre == None) or (self.cluster_points_center_list_pre == [])):
                center_now_filter, center_pre_filter, feature_now_filter, feature_pre_filter, born_index, death_index  = self.id_manage(fisrt=False)
                matched_index, match_index = self.life_manage(center_now_filter, center_pre_filter, feature_now_filter, feature_pre_filter)
                labels_matched, labels_global = self.cluster_iteration(matched_index, match_index, labels_fliter, labels_global, born_index, death_index)
            elif self.cluster_points_center_list_pre == None:       # 
                self.id_manage(fisrt=True)

            else:                                                   # 
                labels_global = self.id_manage(pre_none=True, labels_global=labels_global)
            

            labels_match_temp = labels_matched % 50
            
            max_label = labels_match_temp.max()

            colors = plt.get_cmap("tab20")
            
            colors = colors(labels_match_temp / (max_label if max_label > 0 else 1))
            
            colors[labels_match_temp < 0] = 0
            
            outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

            self.model_feature_pre = model_feature        # 


        self.cluster_points_center_list_pre = cluster_points_center_list

        self.cluster_points_number_list_pre = cluster_points_number_list

        labels_global = labels_global.reshape(-1, 1)

        labels_local = labels_local.reshape(-1, 1)

        all_points = np.concatenate((all_points, labels_local), axis=1)

        all_points = np.concatenate((all_points, labels_global), axis=1)

        all_points = np.concatenate((all_points, label_anno.reshape(-1,1)), axis=1)


        self.label_global = labels_global
        self.save_frame(index=self.frame_id, all_points=all_points)
        torch.cuda.empty_cache()    
        return outlier_cloud, cluster_points_center_list, labels_matched, all_points  


    def end_save(self,index):
        self.save_frame(end=True)
        self.done_list.sort()



    def save_frame(self, index=None, all_points=None, end=False):

        if not end:
            assert len(self.id_mapping) == (len(np.unique(all_points[:,5]))-1)
            for current_idx in np.unique(all_points[:,5]):
                if current_idx in self.done_list:
                    print("wrong !, current_idx already in done")
                    print("wrong !, current_idx already in done")
                    print("wrong !, current_idx already in done")
                    a = np.zeros(shape=(1,1)) + index
                    np.save('wrong.npy', a)

            this_time_deal = []  
            for idx in range(0, len(self.cluster_points_center_list)):
                absolute_idx = self.id_mapping[str(idx)]                # 
                if absolute_idx in self.done_list:
                    print("wrong !, abs idx already in done_list")
                    print("wrong !, abs idx already in done_list")
                    print("wrong !, abs idx already in done_list")
                    a = np.zeros(shape=(1,1)) + index
                    np.save('wrong.npy', a)

                if absolute_idx in self.dealing_list:
                    points_index = np.where(self.label_global == absolute_idx)[0]
                    self.save_dict[str(absolute_idx)].add_new_data(self.frame_id, points_index, self.cluster_points_center_list[idx][0])
                    this_time_deal.append(absolute_idx)                 # 
                else:                                                   # 
                    self.dealing_list.append(absolute_idx)              # 
                    temp_data = TrackDataClass()
                    points_index = np.where(self.label_global == absolute_idx)[0]
                    temp_data.first_deal(self.frame_id, points_index, self.cluster_points_center_list[idx][0])
                    self.save_dict[str(absolute_idx)] = temp_data       # 
                    this_time_deal.append(absolute_idx)                 # 




            for j in self.dealing_list:                                 # 
                if j not in this_time_deal:                             # 
                    view_frame = len(self.save_dict[str(j)].number_list)
                    self.multi_view_map[j, 0] = view_frame
                    self.dealing_list.remove(j)
                    self.done_list.append(j)
            self.frame_index.append(this_time_deal)
            np.save((self.save_frame_points_path + self.seq + '/' + str(index) + '.npy'), all_points)
        
        else:
            len_dealing_list = len(self.dealing_list)
            copy_dealing_list = copy.deepcopy(self.dealing_list)
            for j in range(len_dealing_list):                                 # 
                temp_j = self.dealing_list[j]
                # print("saved", temp_j)
                view_frame = len(self.save_dict[str(temp_j)].number_list)
                self.multi_view_map[temp_j, 0] = view_frame
                copy_dealing_list.remove(temp_j)
                self.done_list.append(temp_j)

            self.dealing_list = copy_dealing_list


 

    # def read_index_and_show(self, index, frame, points):
    #     # 读取已经保存的信息，进行测试
    #     map = np.load(self.save_frame_info_path + 'map.npy')
    #     data_info = self.data_read(index)


    #     point_cloud_array = DataFrame(points[:, 0:3])  #
    #     point_cloud_array.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    #     point_cloud_pynt = PyntCloud(point_cloud_array)  # 将points的数据 存到结构体中
    #     point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    #     point_cloud_o3d.paint_uniform_color((1, 0, 0))

    #     if frame >= data_info.start_frame and frame < (data_info.start_frame + len(data_info.number_list)):

    #         index_frame = data_info.number_list[frame - data_info.start_frame]
    #         index_frame_pre = sum(data_info.number_list[0: (max(0, frame - data_info.start_frame - 1))])
    #         data_index = data_info.index[index_frame_pre: index_frame_pre + index_frame]

    #         colors = np.zeros((points.shape[0], 3))
    #         colors[data_index] = (1, 0, 0)
    #         print(len(data_index))
    #         point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

    #         print('1')

    #         return point_cloud_o3d
    #     # index_show = data_info.number_list[]

