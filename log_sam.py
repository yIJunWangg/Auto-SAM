import numpy as np
import torch

def compute_miou(pred_mask, gt_mask, num_classes=2):
    """
    计算二值分割的mIoU
    :param pred_mask: 模型预测的二值掩码（0/1）[H, W]
    :param gt_mask:  真实标注的二值掩码（0/1）[H, W]
    :param num_classes: 类别数（包括背景）
    :return: miou值
    """
    # 确保输入是numpy数组
    pred_mask = np.asarray(pred_mask)
    gt_mask = np.asarray(gt_mask)
    
    # 确保mask是二维的
    assert pred_mask.ndim == 2 and gt_mask.ndim == 2
    
    # 计算每个类别的IoU
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union == 0:  # 避免除零错误
            iou = 0.0
        else:
            iou = intersection / union
            
        iou_per_class.append(iou)
    
    return np.mean(iou_per_class)

def compute_mpa(pred_mask, gt_mask):
    """
    计算平均像素准确率（mPA）
    :param pred_mask: 模型预测的二值掩码（0/1）
    :param gt_mask:  真实标注的二值掩码（0/1）
    :return: mpa值
    """
    # 确保输入是numpy数组
    pred_mask = np.asarray(pred_mask)
    gt_mask = np.asarray(gt_mask)
    
    # 计算正确像素数
    correct = np.sum(pred_mask == gt_mask)
    total = gt_mask.size
    
    return correct / total

# 示例用法 -------------------------------------------------
if __name__ == "__main__":
    # 生成模拟数据（假设输入是H x W的numpy数组）
    H, W = 256, 256
    batch_size = 4
    
    # 生成随机预测结果（0或1）
    pred_masks = np.random.randint(0, 2, (batch_size, H, W))
    
    # 生成随机真实标签（0或1）
    gt_masks = np.random.randint(0, 2, (batch_size, H, W))
    
    # 计算每个样本的指标
    total_miou = 0.0
    total_mpa = 0.0
    
    for pred, gt in zip(pred_masks, gt_masks):
        total_miou += compute_miou(pred, gt)
        total_mpa += compute_mpa(pred, gt)
    
    # 计算平均指标
    final_miou = total_miou / batch_size
    final_mpa = total_mpa / batch_size
    
    # 打印结果
    print(f"Results over {batch_size} samples:")
    print(f"mIoU: {final_miou*100:.2f}%")
    print(f"mPA:  {final_mpa*100:.2f}%")