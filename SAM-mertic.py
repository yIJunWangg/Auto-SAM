import os
import argparse
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision import transforms

class Evaluator:
    @staticmethod
    def classify_prediction(pred_mask, gt_mask):
        """二值掩码的交并比计算"""
        pred = pred_mask.flatten()
        target = gt_mask.flatten().float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        # 避免除以0
        union = torch.clamp(union, min=1e-5)
        return intersection, union

class AverageMeter:
    """用于存储和计算指标的平均值"""
    def __init__(self, dataset):
        self.intersection_buf = []
        self.union_buf = []
        self.acc_buf = []
        self.loss_buf = []

    def update(self, inter, union, acc, loss):
        self.intersection_buf.append(inter)
        self.union_buf.append(union)
        self.acc_buf.append(acc)
        self.loss_buf.append(loss)

    def compute_iou(self):
        miou = np.mean([i/u for i,u in zip(self.intersection_buf, self.union_buf)])
        return miou

    def compute_mPA(self):
        mpa = np.mean(self.acc_buf)
        return mpa

def evaluate_model(pred_dir, gt_dir):
    """
    参数:
        pred_dir: 预测掩码的目录路径
        gt_dir: 真实掩码的目录路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化评估器
    average_meter = AverageMeter(None)
    
    # 获取文件列表
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    # 图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x > 0.5)  # 二值化处理
    ])
    
    for pred_file in tqdm.tqdm(pred_files):
        # 加载预测和真实掩码
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)  # 假设文件名相同
        
        # 转换为张量
        pred = transform(Image.open(pred_path)).float().to(device)
        gt = transform(Image.open(gt_path)).float().to(device)
        
        # 计算指标
        inter, union = Evaluator.classify_prediction(pred, gt)
        acc = (pred == gt).float().mean()
        
        average_meter.update(
            inter.cpu().numpy(),
            union.cpu().numpy(),
            acc.cpu().numpy(),
            0  # 不需要loss
        )
    
    # 计算最终指标
    miou = average_meter.compute_iou()
    mpa = average_meter.compute_mPA()
    
    print(f'mIoU: {miou:.4f}')
    print(f'mPA: {mpa:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='二值分割评估')
    parser.add_argument('--pred_dir', type=str, default='SAM_pred/', help='预测掩码目录')
    parser.add_argument('--gt_dir', type=str, default='GT/', help='真实掩码目录')
    args = parser.parse_args()
    
    # 验证路径存在
    assert os.path.exists(args.pred_dir), f"预测目录不存在: {args.pred_dir}"
    assert os.path.exists(args.gt_dir), f"真实掩码目录不存在: {args.gt_dir}"
    
    # 运行评估
    evaluate_model(args.pred_dir, args.gt_dir)
