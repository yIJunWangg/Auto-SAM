r""" PASCAL-5i few-shot semantic segmentation dataset """
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


class DatasetROCK(Dataset):
    def __init__(self, datapath, transform, split, shot, use_original_imgsize, mode):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.benchmark = 'rock'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.mode = mode

        # 确定数据子目录（train/test）
        split_folder = 'train' if self.split == 'trn' else 'test'
        
        # 构建完整路径
        self.plus_dir = os.path.join(datapath, 'ROCK/rock_orgin', split_folder, 'XPL+')
        self.minus_dir = os.path.join(datapath, 'ROCK/rock_orgin', split_folder, 'PPL-')
        self.mask_dir = os.path.join(datapath, 'ROCK/rock_orgin', split_folder, 'Masks')

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
            'query_mask': query_mask,
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
        boundary = (mask / 255).floor()
        mask[mask != class_id] = 0
        mask[mask == class_id] = 1
        return mask, boundary

    def load_frame(self, query_path, support_paths):
        # 读取query图像和mask
        query_img = Image.open(query_path)
        query_mask = self.read_mask(query_path)
        
        # 读取support图像和mask
        support_imgs = [Image.open(path) for path in support_paths]
        support_masks = [self.read_mask(path) for path in support_paths]
        
        return query_img, query_mask, support_imgs, support_masks, query_img.size

    def read_mask(self, img_path):
        """从XPL+路径生成对应的mask路径"""
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(base_name)[0] + '.png')
        return torch.tensor(np.array(Image.open(mask_path)))

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
        
        # 解析类别信息
        mask = self.read_mask(query_path)
        label_class = [c.item() for c in mask.unique() if c not in (0, 255)]
        class_sample = random.choice(label_class) if label_class else 1

        # 生成support路径
        if mode == 'equal':
            # 生成对应的PPL-路径
            base_name = os.path.basename(query_path)
            support_path = os.path.join(self.minus_dir, base_name)
            support_paths = [support_path]
        elif mode == 'random':
            # 随机选择同类别的其他样本（需要实现classwise元数据）
            pass  # 需要根据实际需求实现

        return query_path, support_paths, class_sample


    def build_class_ids(self):
        # nclass_trn = self.nclass // self.nfolds
        # class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        # class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids_trn = [1]
        class_ids_val = [1]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    # def build_img_metadata(self):

    #     def read_metadata(split):
    #         # 根据split参数选择对应的子文件夹
    #         split_folder = "train" if split == "train" else "test"  # 假设你的手动分好的两个文件夹名为train和test
    #         img_folder = os.path.join(self.plus_path, split_folder)
            
    #         # 验证文件夹是否存在
    #         if not os.path.exists(img_folder):
    #             raise FileNotFoundError(f"Split folder '{split_folder}' not found in {self.plus_path}")
            
    #         # 获取排序后的文件列表（按文件名排序）
    #         image_files = sorted([
    #             f for f in os.listdir(img_folder)
    #             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))  # 常见图片格式过滤
    #         ])
            
    #         return [os.path.join(img_folder, img_file) for img_file in image_files]


        img_metadata = []
        if self.split == 'trn' :  # For training, read image-metadata of "the other" folds
            img_metadata += read_metadata('train')
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    #获得各个类别的图片列表
    #暂时不改
    def build_img_metadata_classwise(self):
        if self.split == 'trn':
            split = 'train'
        else:
            split = 'val'
        fold_n_subclsdata = os.path.join('data/splits/lists/rock/fss_list/%s/sub_class_file_list_%d.txt' % (split,0))
            
        with open(fold_n_subclsdata, 'r') as f:
            fold_n_subclsdata = f.read()
            
        sub_class_file_list = eval(fold_n_subclsdata)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        return img_metadata_classwise