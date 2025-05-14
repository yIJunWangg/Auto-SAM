""" PASCAL-5i few-shot semantic segmentation dataset (Modified for Binary Segmentation) """
import sys
import os
sys.path.append('G:\wyj_big_project\VRP-SAM-main\VRP-SAM-main')
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetROCKRe(Dataset):
    def __init__(self, datapath, transform, split, shot, use_original_imgsize, mode):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.benchmark = 'rock'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.mode = mode

        # 确定数据子目录（train/test）
        # split_folder = 'train' if self.split == 'trn' else 'test'
        split_folder = 'train' if self.split == 'trn' else 'test'
        
        # 构建完整路径
        self.plus_dir = os.path.join(datapath, 'Demo', split_folder, 'XPL+')
        self.minus_dir = os.path.join(datapath, 'Demo', split_folder, 'PPL-')
        self.mask_dir = os.path.join(datapath, 'Demo', split_folder, 'Masks')

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)
        query_name, support_names, class_sample = self.sample_episode(idx, self.mode)
        
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
        
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), 
                                      query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(img) for img in support_imgs])
        
        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), 
                                 support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        
        batch = {
            'query_img': query_img,
            'query_mask': query_cmask,
            'query_name': query_name,
            'query_ignore_idx': query_ignore_idx,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': torch.stack(support_masks),
            'support_names': support_names,
            'support_ignore_idxs': torch.stack(support_ignore_idxs),
            'class_id': torch.tensor(class_sample)
        }
        return batch

    def extract_ignore_idx(self, mask, class_id):
        """Convert mask to binary (0 or 1) and extract boundary"""
        # Convert mask to binary (assuming 255 is foreground)
        binary_mask = (mask == 255).float()  # 255 -> 1, others -> 0
        boundary = (binary_mask > 0).float()  # Boundary is same as mask in binary case
        return binary_mask, boundary

    def load_frame(self, query_path, support_paths):
        # 读取query图像和mask
        query_img = Image.open(query_path).convert('RGB')
        query_mask = self.read_mask(query_path)
        
        # 读取support图像和mask
        support_imgs = [Image.open(path).convert('RGB') for path in support_paths]
        support_masks = [self.read_mask(path) for path in support_paths]
        
        return query_img, query_mask, support_imgs, support_masks, query_img.size

    def read_mask(self, img_path):
        """读取并二值化mask"""
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(base_name)[0] + '.png')
        mask = np.array(Image.open(mask_path))
        # 二值化处理：255 -> 1, 其他 -> 0
        mask = (mask == 255).astype(np.uint8)
        return torch.tensor(mask)

    def build_img_metadata(self):
        """获取XPL+目录下的所有图像路径"""
        valid_ext = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        images = sorted([os.path.join(self.plus_dir, f) 
                       for f in os.listdir(self.plus_dir)
                       if f.lower().endswith(valid_ext)])
        print(f'Total ({self.split}) images: {len(images)}')
        return images

    def sample_episode(self, idx, mode):
        """生成query-support对"""
        query_path = self.img_metadata[idx]
        
        # 对于二分类任务，class_sample固定为1（前景）
        class_sample = 1

        # 生成support路径
        if mode == 'equal':
            # 生成对应的PPL-路径
            base_name = os.path.basename(query_path)
            support_path = os.path.join(self.minus_dir, base_name)
            support_paths = [support_path]
        elif mode == 'random':
            # 随机选择同类别的其他样本
            support_idx = random.choice(range(len(self.img_metadata)))
            support_paths = [self.img_metadata[support_idx]]

        return query_path, support_paths, class_sample

    def build_class_ids(self):
        # 二分类任务，只有类别1（前景）
        class_ids_trn = [1]
        class_ids_val = [1]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata_classwise(self):
        # 二分类任务简化实现
        img_metadata_classwise = {
            1: [os.path.basename(f) for f in self.img_metadata]  # 所有图片都属于类别1
        }
        return img_metadata_classwise