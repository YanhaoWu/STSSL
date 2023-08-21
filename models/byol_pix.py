# 实现点级别的对比学习

from typing_extensions import assert_type
import torch
import torch.nn as nn
from data_utils.collations import *

latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNet256': 256,         # 使用VoteNet的检测任务，他的Backbone出来的特征是256维的
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}

class Byol_Pix(nn.Module):
    def __init__(self, model, model_preject, model_preject_pix, mode_predict, dtype, args, K=65536, m=0.999, T=0.1):
        super(Byol_Pix, self).__init__()

        self.K = K
        self.m = m
        self.T = T


        # online 部分
        self.model_q = model(in_channels=4 if args.use_intensity else 3, out_channels=latent_features[args.sparse_model])
        self.head_q = model_preject_pix(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size, batch_nor=True, pix_level=True)
        self.predict_q  = mode_predict()

        # target 部分
        self.model_k = model(in_channels=4 if args.use_intensity else 3, out_channels=latent_features[args.sparse_model])
        self.head_k = model_preject(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size, batch_nor=True)



        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize headection head k and q
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


        if torch.cuda.device_count() > 1:
            self.model_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_q)
            self.head_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_q)
            self.predict_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.predict_q)


            self.model_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_k)
            self.head_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_k)


    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)    


    def forward_intra_frames(self, pcd_q, pcd_k, segments=None, pix_deal=True):
        """
        Input:
            pcd_q: a batch of pcds_aum1
            pcd_k: a batch of pcds_aum2
        Output:
            logits, targets
        """

        # compute query features
        representation_q = self.model_q(pcd_q)  # queries: NxC
        representation_k = self.model_q(pcd_k)  # 交换v v'进行特征提取

        if segments is None:
            project_q = self.head_q(representation_q, pix_deal)         # 根据pix_deal判断是否要进行pix-level的一个匹配
            predict_q = self.predict_q(project_q)
            project_k = self.head_q(representation_k, pix_deal)
            predict_k = self.predict_q(project_k)

            q_pcd_1 = nn.functional.normalize(predict_q, dim=1, p=2)
            k_pcd_1 = nn.functional.normalize(predict_k, dim=1, p=2)       # 除以范数，进行归一化

        else:
            # coord and feat in the shape N*SxPx3 and N*SxPxF
            # where N is the batch size and S is the number of segments in each scan
            h_qs = list_segments_points(representation_q.C, representation_q.F, segments[0])
            h_ks = list_segments_points(representation_k.C, representation_k.F, segments[1])
            
            project_q = self.head_q(h_qs, pix_deal)
            predict_q = self.predict_q(project_q)
            project_k = self.head_q(h_ks, pix_deal)
            predict_k = self.predict_q(project_k)

            q_seg_1 = nn.functional.normalize(predict_q, dim=1, p=2)
            k_seg_1 = nn.functional.normalize(predict_k, dim=1, p=2)

    
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_target_encoder()  # update the target encoder

            representation_target_q = self.model_k(pcd_q)  # queries: NxC
            representation_target_k = self.model_k(pcd_k)  # 交换v v'进行特征提取


            if segments is None:
                project_target_q = self.head_k(representation_target_q)
                predict_target_q = self.predict_k(project_target_q)
                project_target_k = self.head_k(representation_target_k)
                predict_target_k = self.predict_k(project_target_k)

                q_pcd_2 = nn.functional.normalize(predict_target_q, dim=1, p=2)
                k_pcd_2 = nn.functional.normalize(predict_target_k, dim=1, p=2)       # 除以范数，进行归一化
            else:
                # coord and feat in the shape N*SxPx3 and N*SxPxF
                # where N is the batch size and S is the number of segments in each scan
                h_qs, number_q = list_segments_points(representation_target_q.C, representation_target_q.F, segments[0], collect_numbers=True)
                h_ks, number_k = list_segments_points(representation_target_k.C, representation_target_k.F, segments[1], collect_numbers=True)
                
                project_target_q = self.head_k(h_qs)
                project_target_k = self.head_k(h_ks)        # 就只有投影部分，没有预测部分了


                # pix_level 独属，为了将size给匹配上，所以需要统计数量，然后扩展
                if pix_deal:
                    assert len(number_q) == project_target_q.shape[0]
                    assert len(number_k) == project_target_k.shape[0]
                    assert len(number_q) == len(number_k)
                    feature_size =  project_target_q.shape[1]

                    q_seg_2_points = project_target_q[0].expand(number_k[0], feature_size)
                    k_seg_2_points = project_target_k[0].expand(number_q[0], feature_size)

                    for idx in range(len(number_q)-1):  # 注意，q_seg_2应该用的是number_k  k_seg_2应该用的是number_q。因为最终是向另外一个变换对齐
                        number_ori_points = number_k[idx+1]
                        temp_tensor = project_target_q[idx+1].expand(number_ori_points, feature_size)
                        q_seg_2_points = torch.cat((q_seg_2_points, temp_tensor))

                        number_ori_points = number_q[idx+1]
                        temp_tensor = project_target_k[idx+1].expand(number_ori_points, feature_size)
                        k_seg_2_points = torch.cat((k_seg_2_points, temp_tensor))

                    q_seg_2 = nn.functional.normalize(q_seg_2_points, dim=1, p=2)
                    k_seg_2 = nn.functional.normalize(k_seg_2_points, dim=1, p=2)
                
                else:
                    q_seg_2 = nn.functional.normalize(project_target_q, dim=1, p=2)
                    k_seg_2 = nn.functional.normalize(project_target_k, dim=1, p=2)

        if segments is None:
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            
            l_1 = (q_pcd_1 * k_pcd_2.detach()).sum(dim=-1).mean()

            l_2 = (k_pcd_1 * q_pcd_2.detach()).sum(dim=-1).mean()   # 1是代表是由online encoder得到，q k分别表示两个增强

            loss = -2 * (l_1 + l_2) + 4

        
            return loss
        else:
            
            l_1 = (q_seg_1 * k_seg_2.detach()).sum(dim=-1).mean()

            l_2 = (k_seg_1 * q_seg_2.detach()).sum(dim=-1).mean()   # 1是代表是由online encoder得到，q k分别表示两个增强

            loss = -2 * (l_1 + l_2) + 4


            return loss







    def forward_inter_frames(self, pcd_q, pcd_k, pcd_c, segments=None):
        """
        Input:
            pcd_q: inter_frame 1 -> xi
            pcd_k: inter_frame 2 -> xj
            pcd_c: intra_frame 1 -> xk
        Output:
            logits, targets
        """

        # compute query features
        pix_deal = False
        representation_q = self.model_q(pcd_q)  # queries: NxC
        representation_k = self.model_q(pcd_k)  # 交换v v'进行特征提取



        h_qs = list_segments_points(representation_q.C, representation_q.F, segments[0])
        h_ks = list_segments_points(representation_k.C, representation_k.F, segments[1])
        
        try:
            project_q = self.head_q(h_qs, pix_deal)
            predict_q = self.predict_q(project_q)
            project_k = self.head_q(h_ks, pix_deal)
            predict_k = self.predict_q(project_k)
            
            
            
            q_seg_1 = nn.functional.normalize(predict_q, dim=1, p=2)
            k_seg_1 = nn.functional.normalize(predict_k, dim=1, p=2)

        
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_target_encoder()  # update the target encoder

                representation_target_q = self.model_k(pcd_q)  # queries: NxC
                representation_target_k = self.model_k(pcd_k)  # 交换v v'进行特征提取


                h_qs, number_q = list_segments_points(representation_target_q.C, representation_target_q.F, segments[0], collect_numbers=True)
                h_ks, number_k = list_segments_points(representation_target_k.C, representation_target_k.F, segments[1], collect_numbers=True)
                
                project_target_q = self.head_k(h_qs)
                project_target_k = self.head_k(h_ks)        # 就只有投影部分，没有预测部分了


                # pix_level 独属，为了将size给匹配上，所以需要统计数量，然后扩展
                if pix_deal:
                    assert len(number_q) == project_target_q.shape[0]
                    assert len(number_k) == project_target_k.shape[0]
                    assert len(number_q) == len(number_k)
                    feature_size =  project_target_q.shape[1]

                    q_seg_2_points = project_target_q[0].expand(number_k[0], feature_size)
                    k_seg_2_points = project_target_k[0].expand(number_q[0], feature_size)

                    for idx in range(len(number_q)-1):  # 注意，q_seg_2应该用的是number_k  k_seg_2应该用的是number_q。因为最终是向另外一个变换对齐
                        number_ori_points = number_k[idx+1]
                        temp_tensor = project_target_q[idx+1].expand(number_ori_points, feature_size)
                        q_seg_2_points = torch.cat((q_seg_2_points, temp_tensor))

                        number_ori_points = number_q[idx+1]
                        temp_tensor = project_target_k[idx+1].expand(number_ori_points, feature_size)
                        k_seg_2_points = torch.cat((k_seg_2_points, temp_tensor))

                    q_seg_2 = nn.functional.normalize(q_seg_2_points, dim=1, p=2)
                    k_seg_2 = nn.functional.normalize(k_seg_2_points, dim=1, p=2)
                
                else:
                    q_seg_2 = nn.functional.normalize(project_target_q, dim=1, p=2)
                    k_seg_2 = nn.functional.normalize(project_target_k, dim=1, p=2)

                
            l_1 = (q_seg_1 * k_seg_2.detach()).sum(dim=-1).mean()

            l_2 = (k_seg_1 * q_seg_2.detach()).sum(dim=-1).mean()   # 1是代表是由online encoder得到，q k分别表示两个增强

            loss_1 = -2 * (l_1 + l_2) + 4

                
            
            
            
            
        except:
            print("two few matched objects 1")
            loss_1 = 0



        # ------------------------------------------------------------------added p2c----------------------------------#
        pix_deal = True                         # 需要改过来
        representation_qc = self.model_q(pcd_q)  # queries: NxC
        representation_c = self.model_q(pcd_c)  # 交换v v'进行特征提取
        



        h_qcs = list_segments_points(representation_qc.C, representation_qc.F, segments[2])
        h_cs  = list_segments_points(representation_c.C, representation_c.F, segments[3])
        

        project_qc = self.head_q(h_qcs, pix_deal)
        predict_qc = self.predict_q(project_qc)
        project_c = self.head_q(h_cs, pix_deal)
        predict_c = self.predict_q(project_c)

        qc_seg_1 = nn.functional.normalize(predict_qc, dim=1, p=2)
        c_seg_1 = nn.functional.normalize(predict_c, dim=1, p=2)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_target_encoder()  # update the target encoder

            representation_target_qc = self.model_k(pcd_q)  # queries: NxC
            representation_target_c = self.model_k(pcd_c)  # 交换v v'进行特征提取
            
            try:
                h_qcs, number_qc = list_segments_points(representation_target_qc.C, representation_target_qc.F, segments[2], collect_numbers=True)
                h_cs, number_c = list_segments_points(representation_target_c.C, representation_target_c.F, segments[3], collect_numbers=True)
                
                project_target_qc = self.head_k(h_qcs)
                project_target_c = self.head_k(h_cs)        # 就只有投影部分，没有预测部分了


                # pix_level 独属，为了将size给匹配上，所以需要统计数量，然后扩展
                if pix_deal:
                    assert len(number_qc) == project_target_qc.shape[0]
                    assert len(number_c) == project_target_c.shape[0]
                    assert len(number_qc) == len(number_c)
                    feature_size =  project_target_qc.shape[1]

                    qc_seg_2_points = project_target_qc[0].expand(number_c[0], feature_size)
                    c_seg_2_points = project_target_c[0].expand(number_qc[0], feature_size)

                    for idx in range(len(number_qc)-1):  # 注意，q_seg_2应该用的是number_k  k_seg_2应该用的是number_q。因为最终是向另外一个变换对齐
                        number_ori_points = number_c[idx+1]
                        temp_tensor = project_target_qc[idx+1].expand(number_ori_points, feature_size)
                        qc_seg_2_points = torch.cat((qc_seg_2_points, temp_tensor))

                        number_ori_points = number_qc[idx+1]
                        temp_tensor = project_target_c[idx+1].expand(number_ori_points, feature_size)
                        c_seg_2_points = torch.cat((c_seg_2_points, temp_tensor))

                    qc_seg_2 = nn.functional.normalize(qc_seg_2_points, dim=1, p=2)
                    c_seg_2 = nn.functional.normalize(c_seg_2_points, dim=1, p=2)
                
                else:
                    qc_seg_2 = nn.functional.normalize(project_target_qc, dim=1, p=2)
                    c_seg_2 = nn.functional.normalize(project_target_c, dim=1, p=2)


            
                l2_1 = (qc_seg_1 * c_seg_2.detach()).sum(dim=-1).mean()

                l2_2 = (c_seg_1 * qc_seg_2.detach()).sum(dim=-1).mean()   # 1是代表是由online encoder得到，q k分别表示两个增强

                loss_2 = -2 * (l2_1 + l2_2) + 4
                
            except:
                print("two few matched objects 2")

                loss_2 = 0

        # return loss

        loss = loss_1 * 0.8 + loss_2 * 0.2  # loss 1_tempororal loss_2 spatial
        
        return loss

