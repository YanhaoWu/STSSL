
import time
from SemanticKITTI import SemanticKITTIDataLoader
import argparse
from TrackClass import Track

def main():
    parser = argparse.ArgumentParser(description='Tracking_Segments')
    parser.add_argument('--seq', type=str, default='01',help='witch seq to deal')
    parser.add_argument('--gpu', type=str, default='1',help='witch gpu to use')   
    parser.add_argument('--epoch', type=str, default='epoch99',help='which epoch to use')  
    parser.add_argument('--dataset_path', type=str, default='None',help='where to save')    
    parser.add_argument('--save_path', type=str, default='None',help='where to save')    
    parser.add_argument('--clusterd_flag', default=True, action='store_true', help='have done cluster?')  
    parser.add_argument('--model_path_backbone', default='None',help='which model to use to help tracking')    
    parser.add_argument('--model_path_head', default='None',help='which model to use to help tracking')    
    args = parser.parse_args()

    model_path = [args.model_path_backbone, args.model_path_head]
    seq = args.seq
    print("now we deal with:", seq)
    a = SemanticKITTIDataLoader(root=args.dataset_path, seq=seq, clusterd_flag=args.clusterd_flag)
    TC = Track(seq=seq, model_path=model_path, save_path=args.save_path, gpu_use=args.gpu)
    start_index = 0
    TC.start_index = start_index
    start_time = time.time()
    for i in range(start_index, a.__len__()):
        outlier_cloud_info, label = a.tracking_item(i)
        if outlier_cloud_info is not None:
            outlier_cloud, cluster_points_center_list, labels_matched, all_points = TC.seg_api(i, outlier_cloud_info[:, 0:4], outlier_cloud_info[:, 4], label)
            if i % 100 == 0:
                print("index finished", i)
        else:
            TC.end_save(index=(i-TC.start_index))
            end_time =time.time()
            print("finish")
            print("it takes", end_time-start_time)
            break;
     
if __name__ == '__main__':
    main()