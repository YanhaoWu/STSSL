

from SemanticKITTI import SemanticKITTIDataLoader
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Clustering Segments')
    parser.add_argument('--dataset_path', type=str, default='/home/ssd/3d',help='witch seq to deal')
    parser.add_argument('--seq', type=str, default='01',help='witch seq to deal')
    parser.add_argument('--cluster_name', type=str, default='cluster_ver100',help='cluster version')              
    parser.add_argument('--epoch', type=str, default='epoch99',help='which epoch to use')     
    parser.add_argument('--save_path', type=str, default='/home/ssd/3d/augmented_views',help='where to save')    
    parser.add_argument('--clusterd_flag', default=False, action='store_true', help='do we need to cluster')                  
    args = parser.parse_args()
    seq = args.seq
    print("now we deal with:", seq)
    a = SemanticKITTIDataLoader(root=args.dataset_path, seq=seq, clusterd_flag=args.clusterd_flag)
    start_index = 0
    # end_index = a.__len__()
    end_index = int((a.__len__())) # for testing
    for i in range(start_index, end_index):
        a.cluster_points(i, args.save_path)
        
        if i == int(end_index / 2 ):
          print(seq, "half finished")
        
    print("finish ", seq)
if __name__ == '__main__':
    main()