import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional.classification import iou
# from torchmetrics.functional.classification import iou
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from data_utils.data_map import labels, content
from data_utils.ioueval import iouEval
from data_utils.collations import *
from numpy import inf, pi, cos, mean
from functools import partial
import sys
class SemanticKITTITrainer(pl.LightningModule):
    def __init__(self, model, criterion, train_loader, params, pre_training=True):
        super().__init__()
        self.train_model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.params = params
        self.frame_work = params.frame_work
        self.writer = SummaryWriter(f'{params.summary_save_dir}')
        self.iter_log = 100  # 100
        self.loss_eval = []
        self.train_step = 0
        self.pix_deal = self.params.pix_deal
        self.index_plus_count = 0


        if self.params.loading_dir is not 'None':
            self.load_checkpoint()

    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################

    def train_step_2(self, batch, batch_nb):
        
        (xi_coord, xi_feats, si), (xj_coord, xj_feats, sj),  (xk_coord, xk_feats, sk) , sik = batch

        xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)
        
        xk = numpy_to_sparse_tensor(xk_coord, xk_feats)

        loss = self.train_model.forward_inter_frames(xi, xj, xk, [si, sj, sik, sk])
        
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            
            self.iter_callback(loss.item())

            
        return {'loss': loss}

    def train_step_1(self, batch, batch_nb):
        
        (xi_coord, xi_feats, si), (xj_coord, xj_feats, sj) = batch

        xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)
        
        loss = self.train_model.forward_intra_frames(xi, xj, [si, sj], self.pix_deal)
        
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            
            self.iter_callback(loss.item())
 
        return {'loss': loss}
    
    
    def training_step(self, batch, batch_nb):

        self.train_step += 1
        torch.cuda.empty_cache()
        if self.params.stage == 0:
            loss =  self.train_step_1(batch, batch_nb)
        else:
            loss =  self.train_step_2(batch, batch_nb)
        
        return loss

    def training_epoch_end(self, outputs):
        
        self.index_plus_count += 1
        if self.index_plus_count == 7:
            self.index_plus_count = 0
            self.train_loader.dataset.index_plus_change_get += 1
        print("index_plus_count=", self.index_plus_count)
        print("self.train_loader.dataset.index_plus_change_get =", self.train_loader.dataset.index_plus_change_get)

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            self.checkpoint_callback()

    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################



    def checkpoint_callback(self):

        if (self.current_epoch + 1 ) % 10 == 0:          # +2 是前面跑错了， 所以临时+2，其他情况下应该是+1
            self.save_checkpoint(f'epoch{self.current_epoch}')

        if (self.current_epoch + 1 ) == self.params.stop_epoch:
            self.save_checkpoint(f'epoch{self.current_epoch}')

    def iter_callback(self, batch_loss, batch_pcd_loss=None, batch_segment_loss=None):
        # after each iteration we log the losses on tensorboard
        self.loss_eval.append(batch_loss)
        if self.train_step % self.iter_log == 0:
            self.write_summary(
                'training/learning_rate',
                self.scheduler.get_lr()[0],
                self.train_step,
            )

            # loss
            self.write_summary(
                'training/loss',
                mean(self.loss_eval),
                self.train_step,
            )

            self.loss_eval = []

    ############################################################################################################################################

    ############################################################################################################################################
    # SUMMARY WRITERS                                                                                                                          #
    ############################################################################################################################################

    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)


    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        
        self.configure_optimizers()
        file_name_1 = self.params.loading_dir
        checkpoint = torch.load(file_name_1, map_location='cpu')

        self.train_model.model_q.load_state_dict(checkpoint['model'])
        self.train_model.model_k.load_state_dict(checkpoint['model'])
        txt_path = self.params.model_save_dir + '/' + 'param.txt'
        
        file = open(txt_path ,'a')

        oldstdout = sys.stdout
        sys.stdout = file

        print("loaded the modle")
        print("loaded", file_name_1)

        file.close()
        sys.stdout = oldstdout

        print("loaded the model")


    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.train_model.model_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.model_save_dir}/{checkpoint_id}_model.pt'

        torch.save(state, file_name)

        state = {
            'model': self.train_model.head_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.model_save_dir}/{checkpoint_id}_model_head.pt'

        torch.save(state, file_name)


        state = {
            'model': self.train_model.model_k.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.model_save_dir}/{checkpoint_id}_k_model.pt'

        torch.save(state, file_name)

        state = {
            'model': self.train_model.head_k.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.model_save_dir}/{checkpoint_id}_k_model_head.pt'

        torch.save(state, file_name)

        state = {
            'model': self.train_model.predict_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.model_save_dir}/{checkpoint_id}_model_predict_head.pt'

        torch.save(state, file_name)
        
        torch.save(self.state_dict(), f'{self.params.model_save_dir}/{checkpoint_id}_full_model.pt')

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        print("get opt")
        # define optimizers
        optimizer = torch.optim.SGD(self.train_model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params.epochs, eta_min=self.params.lr / 1000)

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]

    ############################################################################################################################################

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        pass

    #@pl.data_loader
    def test_dataloader(self):
        pass
