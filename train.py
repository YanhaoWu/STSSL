from trainer.stssl_trainer import SemanticKITTITrainer
from pytorch_lightning import Trainer
from utils import *
import argparse
from numpy import inf
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # for debuging

import torch


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='STSSL_params')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input training batch-size usually 16') 
    parser.add_argument('--epochs', type=int, default=200, metavar='N',  # 3
                        help='number of total training epochs, it decides the params')
    parser.add_argument('--stop_epoch', type=int, default=160, metavar='N',  # 3
                        help='when to stop')
    parser.add_argument('--pre_stop_epoch', type=int, default=160, metavar='N',  # 3
                        help='last time stopped epochs')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--lr', type=float, default=1.8e-1,
                        help='learning rate default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay ')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau temperature smoothing (default 0.1)')
    parser.add_argument('--loading_dir', type=str, default='None', # None
                        help='loading directory ')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='using cuda (default: False')
    parser.add_argument('--feature_size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--num-points', type=int, default=20000,
                        help='Number of points sampled from point clouds (default: 20000')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--use-intensity', action='store_true', default=True,
                        help='use points intensity (default: True')
    parser.add_argument('--num-workers', type=int, default=16,   # 16
                        help='how many workers we use to load data usually 16')                      
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help='you have to change this soon') 
    parser.add_argument('--summary_save_dir', type=str, default='lightning_summay',
                        help='you have to change this soon') 
    parser.add_argument('--pin_memory', action='store_true', default=False,
                        help='pin_memory') 
    parser.add_argument('--segment_pathbase', type=str, default=None,
                        help='where segments are saved')     
    parser.add_argument('--tracking_pathbase', type=str, default=None,
                        help='where tracking results are saved')     
    parser.add_argument('--track_pro', type=float, default = 1, 
                        help='set pro track info is used')   
    parser.add_argument('--frame_work', type=str, default = 'byol_pix', 
                        help='byol/moco/byol_pix')      
    parser.add_argument('--save_index', type=int, default=100,
                        help='where to save')   
    parser.add_argument('--stage', type=int, default=0,
                        help='0 or 1')   
    parser.add_argument('--pix_deal', default=True, help='P2C set')   
    
    
    
    
    
    
    args = parser.parse_args()
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    print('GPU')

    oldstdout = sys.stdout
    model_save_dir = args.model_save_dir
    summary_save_dir = args.model_save_dir + '/' + 'summary'
    args.summary_save_dir = summary_save_dir
    if not(os.path.isdir(model_save_dir)):
        print("making dir", model_save_dir)
        os.makedirs(model_save_dir)
        
    if os.path.isdir(model_save_dir):
        print("have dir", model_save_dir)
        print("have dir", summary_save_dir)
    else:
        os.makedirs(model_save_dir)
        os.makedirs(summary_save_dir)
        print("making dir", model_save_dir)
        print("making dir", summary_save_dir)


    argsDict = args.__dict__
    config_save_path = model_save_dir + '/' + 'config.txt'
    print("saving config at", config_save_path)
    with open(config_save_path, 'w') as f:
      for eachArg, value in argsDict.items():
          f.writelines(eachArg + ' : ' + str(value) + '\n')


    data_train = get_dataset(args)
    train_loader = get_data_loader(data_train, args ,pin_memory=args.pin_memory)
    criterion = nn.CrossEntropyLoss().cuda()
    model = get_byol_pix_model(args, dtype)
    load_path = None
    
    stop_ep = args.stop_epoch

    model_sem_kitti = SemanticKITTITrainer(model, criterion, train_loader, args)
    if load_path is not None:
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=stop_ep, accumulate_grad_batches=args.accum_steps, resume_from_checkpoint = load_path)
    else:
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=stop_ep, accumulate_grad_batches=args.accum_steps)

    trainer.fit(model_sem_kitti)




