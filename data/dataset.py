r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.rock import DatasetROCK
from data.rockdataset import DatasetROCKRe
# from data.coco2pascal import DatasetCOCO2PASCAL


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            # 'pascal': DatasetPASCAL,
            # 'coco': DatasetCOCO,
            'rock': DatasetROCK,
            'rockre':DatasetROCKRe,
            
        }

        # cls.img_mean = [0.485, 0.456, 0.406]
        # cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        # if cls=='pascal' or cls=='coco':
        #     cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(cls.img_mean, cls.img_std)])
        # else:
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor()])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, mode ,shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        dataset = cls.datasets[benchmark](cls.datapath, transform=cls.transform, split=split, mode=mode, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        if split == 'trn':
            shuffle = True
        else:
            shuffle = False
        # pin_memory=True 数据一部分会被加载到cpu里，然后再传到GPU里，这样可以减少内存的占用,但占用cpu
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, pin_memory=False, num_workers=nworker)

        return dataloader
    
