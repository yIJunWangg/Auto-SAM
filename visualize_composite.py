r""" Visual Prompt Encoder training (validation) code """

import os
os.environ['MASTER_ADDR'] = 'localhost'  # 主节点的地址，这里我们使用 localhost
os.environ['MASTER_PORT'] = '12355'     # 主节点的端口
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

import tqdm
from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred



import os
import numpy as np
from PIL import Image

def visualize(args, epoch, model, sam_model, dataloader, training,dataroot):
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    # model.train_mode() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in tqdm.tqdm(enumerate(dataloader)):
        
        batch = utils.to_cuda(batch)
        original_size = (384,576)  # (H, W)

        protos,_ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training,dataroot,batch['query_name'])
        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'],batch['support_imgs'].squeeze(1),protos)
        logit_mask = low_masks
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.compute_objective(logit_mask, batch['query_mask'])
            
        # 调整掩码尺寸到原始比例
        pred_mask = F.interpolate(pred_mask, size=original_size, mode='bilinear', align_corners=False)
        pred_mask = (pred_mask > 0.5).float()

# 保存带掩码叠加的原图
        for i in range(pred_mask.shape[0]):
            # 获取原始图像并转换为PIL格式
            query_img = batch['query_img'][i].cpu().numpy()
            query_img = (query_img * 255).astype(np.uint8).transpose(1, 2, 0)  # C,H,W -> H,W,C
            pil_img = Image.fromarray(query_img).convert("RGBA")

            # 创建蓝绿色半透明掩码
            mask = pred_mask[i, 0].cpu().numpy()
            colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)  # RGBA格式
            
            # 设置颜色 (RGB: [30, 144, 255], alpha: 0.6)
            colored_mask[..., 0] = 30    # R
            colored_mask[..., 1] = 144   # G
            colored_mask[..., 2] = 255   # B
            colored_mask[..., 3] = (mask * 153).astype(np.uint8)  # 0.6*255≈153

            # 确保两张图片尺寸相同
            if pil_img.size != (original_size[1], original_size[0]):
                pil_img = pil_img.resize((original_size[1], original_size[0]), Image.BILINEAR)
            
            # 叠加图像
            mask_pil = Image.fromarray(colored_mask, mode='RGBA')
            composite = Image.alpha_composite(pil_img, mask_pil).convert("RGB")

            # 保存结果
            img_name = f"{os.path.splitext(batch['query_name'][i])[0]}.png"
            name = f"{os.path.splitext(img_name.split('/')[-1])[0]}.png"
            composite.save(os.path.join(args.save_dir, name))
        # area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        # average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone(),preds=pred_mask, targets=batch['query_mask'])
        avg_loss = utils.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        mPA = average_meter.compute_mPA()
    return avg_loss, miou, fb_iou, mPA



if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='dataset/')
    parser.add_argument('--benchmark', type=str, default='rockre', choices=['pascal', 'coco', 'fss','rockre'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--mode', type=str, default='equal', choices=['equal', 'random'], help='mode of sampling support images')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--save_dir', type=str, default='results/composite', help='Directory to save prediction masks')
    args = parser.parse_args()
    # Distributed setting
    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    check_point=r'logs/_0401_112338.log/best_model.pt'
    model.load_state_dict(torch.load(check_point,weights_only=True))
    check_point_sam = torch.load(r'logs/_0401_112338.log/best_model_sam.pt',weights_only=True)
    sam_model.fusion.load_state_dict(check_point_sam['fusion_state_dict'])
    sam_model.beta.data = check_point_sam['beta_state_dict'].data
    Evaluator.initialize(args)
    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, split='trn', mode = args.mode)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, split='val', mode = args.mode)

    # Training 
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):#测试
        with torch.no_grad():
            # trn_loss, trn_miou, trn_fb_iou = visualize(args, epoch, model, sam_model, dataloader_trn, training=True)
            val_loss, val_miou, val_fb_iou, mPA = visualize(args, epoch, model, sam_model, dataloader_val, training=False,dataroot='dataset/ROCK/rock_orgin')

        # Save the best model          
        # if utils.is_main_process():
        #     Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        #     Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        #     Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        #     Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')