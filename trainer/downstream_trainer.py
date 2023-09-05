import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import iou
from torch.utils.tensorboard import SummaryWriter
import torch
from data_utils.data_map import labels, content, color_map
from data_utils.ioueval import iouEval
from data_utils.collations import *
from numpy import inf, pi, cos, array, expand_dims
from functools import partial
import torch.distributed as dist
class SemanticKITTITrainer(pl.LightningModule):
    def __init__(self, model, model_head, criterion, train_loader, val_loader, params):
        super().__init__()
        self.model = model
        self.model_head = model_head
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.writer = SummaryWriter(f'runs/{params.checkpoint}')
        self.iter_log = 100
        self.best_acc = -1.
        self.best_loss = inf
        self.evaluator = iouEval(n_classes=len(content.keys()), ignore=0)
        self.val_step = 0
        self.train_step = 0

        if self.params.load_path is not 'None':
            self.load_checkpoint()

    ############################################################################################################################################
    # FORWARD                                                                                                                                 #
    ############################################################################################################################################

    def forward(self, x):
        h = self.model(x)
        z = self.model_head(h)

        return z

    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################

    def training_step(self, batch, batch_nb):
        # for downstream task the batch is the sample x and labels y
        self.train_step += 1
        x_coord, x_feats, x_label = batch
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        y = y[:,0]

        z = self.forward(x)
        loss = self.criterion(z, y.long())
        pred = z.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100.)

        self.downstream_iter_callback(loss.item(), batch_acc, pred, y, x.C, True)
        self.scheduler.step()

        return {'loss': loss, 'acc': batch_acc}

    def training_epoch_end(self, outputs):
        if (self.current_epoch == (self.params.epochs - 1)) or ((self.current_epoch + 1) % 300 == 0):
          if dist.get_rank() == 0:
              self.save_checkpoint(f'epoch{self.current_epoch}')
    
      
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        avg_acc = torch.FloatTensor([ x['acc'] for x in outputs ]).mean()
        epoch_dict = {'avg_loss': avg_loss, 'avg_acc': avg_acc}

    ############################################################################################################################################

    ############################################################################################################################################
    # VALIDATION                                                                                                                               #
    ############################################################################################################################################

    def validation_step(self, batch, batch_nb):
        # validation step for downstream task
        self.val_step += 1
        x_coord, x_feats, x_label = batch
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)
        y = y[:,0]

        z = self.forward(x)

        loss = self.criterion(z, y.long())
        pred = z.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100.)

        self.downstream_iter_callback(loss.item(), batch_acc, pred, y, x.C, False)

        return {'loss': loss.item(), 'acc': batch_acc}

    def validation_epoch_end(self, outputs):
        # at the end of each validation epoch will call our implemented callback to save the checkpoint
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        avg_acc = torch.FloatTensor([ x['acc'] for x in outputs ]).mean()
        epoch_dict = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        self.validation_epoch_callback(avg_loss, avg_acc)

        torch.cuda.empty_cache()
        return epoch_dict

    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################
    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)
        
    def downstream_iter_callback(self, batch_loss, batch_acc, pred, target, coord, is_train):
        # after each iteration we log the losses on tensorboard
        self.evaluator.addBatch(pred.long().cpu().numpy(), target.long().cpu().numpy())
        self.evaluator.addLoss(batch_loss)

        if self.train_step % self.iter_log == 0 or not is_train:
            if is_train:
                self.write_summary(
                    'training/learning_rate',
                    self.scheduler.get_lr()[0],
                    self.train_step,
                )

            # loss
            self.write_summary(
                'training/loss' if is_train else 'validation/loss',
                self.evaluator.getloss(),
                self.train_step if is_train else self.val_step,
            )

            # accuracy
            self.write_summary(
                'training/acc' if is_train else 'validation/acc',
                self.evaluator.getacc(),
                self.train_step if is_train else self.val_step,
            )

            # mean iou
            mean_iou, class_iou = self.evaluator.getIoU()
            self.write_summary(
                'training/miou' if is_train else 'validation/miou',
                mean_iou.item(),
                self.train_step if is_train else self.val_step,
            )

            # per class iou
            for class_num in range(class_iou.shape[0]):
                self.write_summary(
                    f'training/per_class_iou/{labels[class_num]}' if is_train else f'validation/per_class_iou/{labels[class_num]}',
                    class_iou[class_num].item(),
                    self.train_step if is_train else self.val_step,
                )

            self.evaluator.reset()

    def validation_epoch_callback(self, curr_loss, curr_acc):
      
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
      
          self.save_checkpoint(f'epoch{self.current_epoch}')

          if self.current_epoch == self.params.epochs - 1:
              self.save_checkpoint(f'lastepoch{self.current_epoch}')

        #self.validation_mesh_writer()


    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        self.configure_optimizers()

        # load model, best loss and optimizer
        file_name = self.params.load_path
        checkpoint = torch.load(file_name, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model'])
        print('loaded from', file_name)


    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.model.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'val_loss': self.best_loss,
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
            'val_step': self.val_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_model_{self.params.checkpoint}.pt'

        torch.save(state, file_name)

        state = {
            'model': self.model_head.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'val_loss': self.best_loss,
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
            'val_step': self.val_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_model_head_{self.params.checkpoint}.pt'

        torch.save(state, file_name)

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        # define optimizers
        if not self.params.linear_eval:
            print('Fine-tuning!')
            optim_params = list(self.model.parameters()) + list(self.model_head.parameters())
        else:
            print('Linear eval!')
            optim_params = list(self.model_head.parameters())
            self.model.eval()

        optimizer = torch.optim.SGD(
            optim_params, lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True
        )

        def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
            iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
            return 0.5 * (1 + cos(pi * k /
                                    (num_epochs * iter_per_epoch)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=self.params.epochs,
                batch_size=self.params.batch_size,
                dataset_size=len(self.train_loader) * self.params.batch_size,
            )
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    #@pl.data_loader
    def test_dataloader(self):
        pass