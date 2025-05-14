r""" Visual Prompt Encoder training (validation) code """
import os
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
from segment_anything.segment_anything import sam_model_registry


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training,dataroot=None):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    # model.train_mode() if training else model.eval() 
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in tqdm.tqdm(enumerate(dataloader)):
        
        batch = utils.to_cuda(batch)

        protos,_ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training,dataroot,batch['query_name'])

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'],batch['support_imgs'].squeeze(1),protos)
        logit_mask = low_masks
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone(),preds=pred_mask, targets=batch['query_mask'])
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    mPA = average_meter.compute_mPA()

    return avg_loss, miou, fb_iou, mPA


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='dataset/')
    parser.add_argument('--benchmark', type=str, default='rockre', choices=['pascal', 'coco', 'fss','rock','rockre'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=2)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--mode', type=str, default='equal', choices=['equal', 'random'], help='mode of sampling support images')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    device = torch.device('cuda')
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone,True)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)

    # optimizer = optim.AdamW([
    #     # {'params': model.transformer_decoder.parameters()},
    #     {'params':model.parameters()},
    #     # {'params': model.downsample_query.parameters(), "lr": args.lr},
    #     # {'params': model.merge_1.parameters(), "lr": args.lr},
    #     # {'params': model.rock_feat, "lr": args.lr},
    #     # {'params':model.assign.parameters(),"lr":args.lr},
    #     {'params': sam_model.fusion.parameters(), "lr": args.lr},
    #     {'params': sam_model.beta, "lr": args.lr},
    #     # {'params':sam_model.channel_attention.parameters(),"lr":args.lr},
    #     ],lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # Evaluator.initialize(args)
    optimizer_adamw = None
    optimizer_sgd = None


    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, split='trn', mode = args.mode)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, split='val', mode = args.mode)

    
    # Training 
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    best_val_mPA = float('-inf')
    for epoch in range(args.epochs):
        if epoch < 12:  # 前8轮使用AdamW
            if optimizer_adamw is None:
                optimizer_adamw = optim.AdamW([
                    {'params': model.transformer_decoder.parameters(),"lr": args.lr},
                    {'params': model.downsample_query.parameters(), "lr": args.lr},
                    {'params': model.merge_1.parameters(), "lr": args.lr},
                    {'params': sam_model.fusion.parameters(), "lr": args.lr},
                    {'params': sam_model.beta, "lr": args.lr},
                ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
            optimizer = optimizer_adamw
        else:  # 后使用SGD
            if optimizer_sgd is None:
                optimizer_sgd = optim.SGD([
                    {'params': model.transformer_decoder.parameters(),"lr": args.lr},
                    {'params': sam_model.fusion.parameters(), "lr": args.lr},
                    {'params': sam_model.beta, "lr": args.lr},
                ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            optimizer = optimizer_sgd
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
            
        trn_loss, trn_miou, trn_fb_iou ,trn_mPA = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True,dataroot='dataset/ROCK/rock_orgin')
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou ,val_mPA = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False,dataroot='dataset/ROCK/rock_orgin')

        # Save the best model
        if val_miou >= best_val_miou and val_mPA >= best_val_mPA:
            best_val_miou = val_miou
            best_val_mPA = val_mPA
            if utils.is_main_process():
                #记录VRP编码器
                Logger.save_model_miou(model, epoch, val_miou,val_mPA)
                #记录SAM模型
                Logger.save_sam_model_miou(sam_model, epoch, val_miou,val_mPA)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
        if epoch == args.epochs - 1:
            Logger.save_model_final(model, sam_model, epoch,val_miou,val_mPA)
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')