import numpy as np
import sys 
sys.path.append("..") 
from utils import *
from data_utils.collations import numpy_to_sparse_tensor, SparseCollationWithClusterinfo
import open3d as o3d
from data_utils.data_map import color_map
from models.minkunet import *
from models.blocks import ProjectionHead

class model_simi_class(object):
    def __init__(self, model_path, gpu_use):
        
        
        self.feature_size = 128
        self.sparse_resolution = 0.05
        self.sparse_model = 'MinkUnet'
        self.model_path = model_path
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use 
        import torch
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(0)
        print('GPU')
        set_deterministic()
        self.model = None
        print("check the model_path, now it is:", self.model_path)
        # define backbone architecture
       
        model = MinkUNet(in_channels=4, out_channels=96).type(dtype)
        model.eval()
        
        projection_head = ProjectionHead(in_channels=96, out_channels=128, batch_nor=True).type(dtype)
        projection_head.eval()

        model_filename = model_path[0]
        projection_head_filename = model_path[1]
        
        print(model_filename, projection_head_filename)
        print("the model_path is ", self.model_path)

        checkpoint = torch.load(model_filename, map_location={'cuda:6':'cuda:0'})
        model.load_state_dict(checkpoint['model'])
        
        checkpoint = torch.load(projection_head_filename, map_location={'cuda:6':'cuda:0'})
        projection_head.load_state_dict(checkpoint['model'])

        self.model = {'model': model, 'projection_head': projection_head}
        self.collect_fn = SparseCollationWithClusterinfo(self.sparse_resolution)


    def forward(self, points_set):
        with torch.no_grad():
            x_coord, x_feats, x_label = self.collect_fn.__call__(points_set)        
            # print("iter_n", iter_n)
            x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)
            h = self.model['model'](x)
            h_qs = list_segments_points(h.C, h.F, x_label.reshape(1,-1))
            z = self.model['projection_head'](h_qs)
            return z
    


    def sparse_tensor_to_pcd(self, coords, feats, sparse_resolution, shift=False):
        pcd = o3d.geometry.PointCloud()

        points = self.args.sparse_resolution * coords.numpy()

        colors = [ color_map[int(label)] for label in feats.numpy() ]
        colors = np.asarray(colors) / 255.
        colors = colors[:, ::-1]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if shift:
            shift_size = (pcd.get_max_bound()[1] - pcd.get_min_bound()[1]) * 1.5
            points[:, 1] = points[:, 1] + shift_size
            pcd.points = o3d.utility.Vector3dVector(points)

        return pcd



    # def get_args(self):
    #     parser_1 = argparse.ArgumentParser(description='model_simi')
    #     parser_1.add_argument('--feature-size', type=int, default=128,
    #                         help='Feature output size (default: 128')
    #     parser_1.add_argument('--sparse-resolution', type=float, default=0.05,
    #                         help='Sparse tensor resolution (default: 0.05')
    #     parser_1.add_argument('--sparse-model', type=str, default='MinkUNet',
    #                         help='Sparse model to be used (default: MinkUNet')
    #     parser_1.add_argument('--model_path', type=str, default='None',
    #                         help='which model to use')  
    #     args = parser_1.parse_args()
    #     print("model_simi")

        # return args



