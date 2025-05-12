import sys
from segment_anything.segment_anything import sam_model_registry
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.Fusion import Compensation_Fusion
from model.Channel_transfom import DualPathChannelAttention
import math

class SAM_pred(nn.Module):
    def __init__(self,mode='normal'):
        super().__init__()
        self.sam_model = sam_model_registry['vit_h'](r'/mnt/windows_D/datasets/wyj_dataset/sam_vit_h_4b8939.pth')
        # self.fusion = Compensation_Fusion(embed_dim=256, num_heads=8, dropout=0.1, depth=3)
        self.fusion = Compensation_Fusion(embed_dim=256, num_heads=8, dropout=0.1, depth=3)
        self.beta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.mode = mode
        # self.channel_attention = DualPathChannelAttention()

    def model_params(self):
        model_params = {
            'fusion': self.fusion.state_dict(),
            'beta': self.beta,
        }
        return model_params
    
    def forward_img_encoder(self, XPL_img, PPL_img):
        XPL_img = F.interpolate(XPL_img, (1024, 1024), mode='bilinear', align_corners=True)
        PPL_img = F.interpolate(PPL_img, (1024, 1024), mode='bilinear', align_corners=True)

        with torch.no_grad():
            # 获取最终特征和第二层特征
            query_feats_one, layer_feats_one = self.sam_model.image_encoder(XPL_img)
            query_feats_two, layer_feats_two = self.sam_model.image_encoder(PPL_img)

        return (query_feats_one, layer_feats_one), (query_feats_two, layer_feats_two)

    # 融合特征
    def fuse_features(self, query_feats_one, query_feats_two,mode='normal'):
        if mode=='normal':
            fused_feats = self.fusion(query_feats_one, query_feats_one - query_feats_two, query_feats_two)
            fused_feats = (self.beta * query_feats_one) + ((1 - self.beta) * fused_feats)
        elif mode=='ablation':
            fused_feats = (self.beta * query_feats_one) + ((1 - self.beta) * query_feats_two)
        return fused_feats

    # 单独提取 XPL_img 的特征
    def forward_img_encoder_one(self, XPL_img):
        XPL_img = F.interpolate(XPL_img, (1024, 1024), mode='bilinear', align_corners=True)
        with torch.no_grad():
            query_feats,layer_feat = self.sam_model.image_encoder(XPL_img)
        return query_feats,layer_feat

# 从缓存中加载特征或提取特征
    def get_feat_from_np(self, XPL_img, query_name, PPL_img):
        final_feat_one_list = []
        final_feat_two_list = []
        for idx, name in enumerate(query_name):
            name = os.path.splitext(name.split('/')[-1])[0]

            # if not (os.path.exists(feat_one_path) and os.path.exists(feat_two_path)):
            # 提取特征
            (q_feat_one, (l0, l1, l2, l3)), (q_feat_two, (l0_, l1_, l2_, l3_)) = self.forward_img_encoder(
                XPL_img[idx, :, :, :].unsqueeze(0),
                PPL_img[idx, :, :, :].unsqueeze(0))
            
            layers_one = [l0, l1, l2, l3]
            layers_two = [l0_, l1_, l2_, l3_]
                
            # 收集特征
            final_feat_one_list.append(q_feat_one)
            final_feat_two_list.append(q_feat_two)


        # 拼接最终特征
        final_feats_one = torch.cat(final_feat_one_list, dim=0)
        final_feats_two = torch.cat(final_feat_two_list, dim=0)

        fused_feats = self.fuse_features(final_feats_one, final_feats_two,mode=self.mode)
        return fused_feats, (layers_one, layers_two)

    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt


    
    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        """
        visual reference prompt is added to the q_sparse_em
        the orginal prompt encoder is modifiedS
        """
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512)):
        """
        query_feats: image feature from SAM image encoder
        q_sparse_em: sparse prompt embeddings
        q_dense_em: dense prompt embeddings
        """
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
        # from torch.nn.functional import threshold, normalize

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask
    
    def forward(self, XPL_img, query_name,PPL_img, protos, points_mask=None):
        B,C, h, w = XPL_img.shape
        
        # XPL_img = F.interpolate(XPL_img, (1024,1024), mode='bilinear', align_corners=True)
        protos, point_prompt = self.get_pormpt(protos, points_mask)
        # with torch.no_grad():
            #-------------save_sam_img_feat-------------------------
            # query_feats = self.forward_img_encoder(XPL_img)
        # PPL_img = self.channel_attention(XPL_img,PPL_img)
        
        # 消融设置
        # if points_mask is None:
        #     # 创建均匀网格点
        #     x_coords = torch.linspace(0, w-1, 8, device=XPL_img.device)
        #     y_coords = torch.linspace(0, h-1, 8, device=XPL_img.device)
            
        #     # 生成所有点组合
        #     grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        #     points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
            
        #     # 转换为SAM需要的格式 (B, N, 2)
        #     points = points.unsqueeze(0).repeat(B, 1, 1)
        #     point_labels = torch.ones(B, points.shape[1], device=XPL_img.device)
        #     point_prompt = (points, point_labels)
        # else:
        #     # 如果提供了points_mask，使用原始逻辑
        #     point_mask = points_mask
        #     postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 - 0.5
        #     postivate_pos = postivate_pos[:,:,[1,0]]
        #     point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
        #     point_prompt = (postivate_pos, point_label)
        query_feats = self.get_feat_from_np(XPL_img, query_name, PPL_img)

        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=point_prompt,
                boxes=None,
                protos=protos,
                masks=None)
        
        # 消融设置
        
        # q_sparse_em, q_dense_em = self.forward_prompt_encoder(
        # points=point_prompt,
        # boxes=None,
        # protos=None,
        # masks=None)
            
        low_masks, binary_mask = self.forward_mask_decoder(query_feats[0], q_sparse_em, q_dense_em, ori_size=(h, w))

        return low_masks, binary_mask.squeeze(1)