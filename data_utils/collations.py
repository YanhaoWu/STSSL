import numpy as np
import MinkowskiEngine as ME
import torch
from pcd_utils.pcd_preprocess import overlap_clusters
import copy
def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

def list_segments_points(p_coord, p_feats, labels, collect_numbers=False):
    c_coord = []
    c_feats = []
    num_points = []
    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        batch_ind = p_coord[:,0] == batch_num       # 
                                                    # 
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue
            # batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1        # 每个聚类体单独看做一个Batch

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)
            num_points.append(segment_coord.shape[0])

    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if collect_numbers:
        return ME.SparseTensor(
                    features=seg_feats,
                    coordinates=seg_coord,
                    device=device,
                ), num_points
    else:
        return ME.SparseTensor(
                    features=seg_feats,
                    coordinates=seg_coord,
                    device=device,
                )       

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda()

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]

def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j

    
class SparseSegmentWithTrackCollation:
    # 
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        points_i, points_j, points_k = list(zip(*list_data))

        points_i = np.asarray(points_i)
        points_j = np.asarray(points_j)
        points_k = np.asarray(points_k)

        pi_feats = []
        pi_coord = []
        pi_cluster = []

        pj_feats = []
        pj_coord = []
        pj_cluster = []

        pk_feats = []
        pk_coord = []
        pk_cluster = []

        p_ik_cluster = [] 

        for pi, pj, pk in zip(points_i, points_j, points_k):
            # pi[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pi, feats_pi, cluster_pi = point_set_to_coord_feats(pi[:,:-1], pi[:,-1], self.resolution, self.num_points)
            pi_coord.append(coord_pi)
            pi_feats.append(feats_pi)

            # pj[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pj, feats_pj, cluster_pj = point_set_to_coord_feats(pj[:,:-1], pj[:,-1], self.resolution, self.num_points)
            pj_coord.append(coord_pj)
            pj_feats.append(feats_pj)

            coord_pk, feats_pk, cluster_pk = point_set_to_coord_feats(pk[:,:-1], pk[:,-1], self.resolution, self.num_points)
            pk_coord.append(coord_pk)
            pk_feats.append(feats_pk)

            cluster_pi_save = copy.deepcopy(cluster_pi)

            cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj)
            cluster_pi_k, cluster_pk = overlap_clusters(cluster_pi_save, cluster_pk)


            # we store the segment labels per point
            pi_cluster.append(cluster_pi)
            pj_cluster.append(cluster_pj)
            pk_cluster.append(cluster_pk)
            p_ik_cluster.append(cluster_pi_k)

        pi_feats = np.asarray(pi_feats)
        pi_coord = np.asarray(pi_coord)

        pj_feats = np.asarray(pj_feats)
        pj_coord = np.asarray(pj_coord)

        pk_feats = np.asarray(pk_feats)
        pk_coord = np.asarray(pk_coord)

        segment_i = np.asarray(pi_cluster)
        segment_j = np.asarray(pj_cluster)
        segment_k = np.asarray(pk_cluster)
        segment_ik = np.asarray(p_ik_cluster)

        # if not segment_contrast segment_i and segment_j will be an empty list
        return (pi_coord, pi_feats, segment_i), (pj_coord, pj_feats, segment_j), (pk_coord, pk_feats, segment_k), segment_ik



class SparseSegmentsCollation:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        points_i, points_j = list(zip(*list_data))

        points_i = np.asarray(points_i)
        points_j = np.asarray(points_j)

        pi_feats = []
        pi_coord = []
        pi_cluster = []

        pj_feats = []
        pj_coord = []
        pj_cluster = []

        for pi, pj in zip(points_i, points_j):
            # pi[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pi, feats_pi, cluster_pi = point_set_to_coord_feats(pi[:,:-1], pi[:,-1], self.resolution, self.num_points)
            pi_coord.append(coord_pi)
            pi_feats.append(feats_pi)

            # pj[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pj, feats_pj, cluster_pj = point_set_to_coord_feats(pj[:,:-1], pj[:,-1], self.resolution, self.num_points)
            pj_coord.append(coord_pj)
            pj_feats.append(feats_pj)

            cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj)

                # we store the segment labels per point
            pi_cluster.append(cluster_pi)
            pj_cluster.append(cluster_pj)

        pi_feats = np.asarray(pi_feats)
        pi_coord = np.asarray(pi_coord)

        pj_feats = np.asarray(pj_feats)
        pj_coord = np.asarray(pj_coord)

        segment_i = np.asarray(pi_cluster)
        segment_j = np.asarray(pj_cluster)

        # if not segment_contrast segment_i and segment_j will be an empty list
        return (pi_coord, pi_feats, segment_i), (pj_coord, pj_feats, segment_j)
      
      




class SparseCollationWithClusterinfo:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, points_set):
        points_set, labels = points_set[:,:4],  points_set[:,-1]

        points_set = np.asarray(points_set)
        labels = np.asarray(labels)

        points_set = points_set[np.newaxis,:]
        labels = labels.reshape(-1, 1)
        labels = labels[np.newaxis, :]
        p_feats = []
        p_coord = []
        p_label = []
        for points, label in zip(points_set, labels):
            coord, feats, label_ = point_set_to_coord_feats(points, label, self.resolution, self.num_points, True)
            p_feats.append(feats)
            p_coord.append(coord)
            p_label.append(label_)

        p_feats = np.asarray(p_feats)
        p_coord = np.asarray(p_coord)
        p_label = np.asarray(p_label)

        return p_coord, p_feats, p_label    
      





      
class SparseCollation:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        points_set, labels = list(zip(*list_data))

        points_set = np.asarray(points_set)
        labels = np.asarray(labels)

        p_feats = []
        p_coord = []
        p_label = []
        for points, label in zip(points_set, labels):
            coord, feats, label_ = point_set_to_coord_feats(points, label, self.resolution, self.num_points, True)
            p_feats.append(feats)
            p_coord.append(coord)
            p_label.append(label_)

        p_feats = np.asarray(p_feats)
        p_coord = np.asarray(p_coord)
        p_label = np.asarray(p_label)

        # if we directly map coords and feats to SparseTensor it will loose the map over the coordinates
        # if the mapping between point and voxels are necessary, please use TensorField
        # as in https://nvidia.github.io/MinkowskiEngine/demo/segmentation.html?highlight=segmentation
        # we first create TensorFields and from it we create the sparse tensors, so we can map the coordinate
        # features across different SparseTensors, i.e. output prediction and target labels

        return p_coord, p_feats, p_label
