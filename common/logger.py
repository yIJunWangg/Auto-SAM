r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
from .utils import is_main_process, save_on_master, reduce_metric


from collections import defaultdict
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'rock':
            self.nclass = 2

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

        # For mAP calculation
        self.predictions =  []# To store predictions per class
        self.ground_truths =  [] # To store ground truths per class

    def update(self, inter_b, union_b, class_id, loss, preds=None, targets=None):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)
        
        # Store predictions and ground truths for mAP calculation
        if preds is not None and targets is not None:
            for i in range(len(class_id)):
                self.predictions.append(preds[i].cpu().numpy())
                self.ground_truths.append(targets[i].cpu().numpy())

    def compute_iou(self):
        iou = self.intersection_buf.float() / torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100  
        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def compute_mPA(self):

        total_pixels = 0
        correct_pixels = 0

        for predictions, ground_truths in zip(self.predictions, self.ground_truths):
            predictions = predictions.ravel() 
            ground_truths = ground_truths.ravel()
            correct = np.sum(predictions == ground_truths)
            total = len(predictions.ravel())
            if total > 0:
                total_pixels += total
                correct_pixels += correct
        mpa = correct_pixels / total_pixels if total_pixels > 0 else 0
        return mpa*100

    def write_result(self, split, epoch):
        self.intersection_buf, self.union_buf = self.reduce_metrics([self.intersection_buf, self.union_buf], False)
        iou, fb_iou = self.compute_iou()
        mPA = self.compute_mPA()  # Compute mAP

        # loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
        if epoch != -1:
            loss_buf = torch.stack(self.loss_buf)
            loss_buf = self.reduce_metrics([loss_buf])[0]
            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        msg += 'mPA: %5.2f   ' % mPA  
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            mPA = self.compute_mPA() 

            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f  |  ' % fb_iou
            msg += 'mPA: %5.2f' % mPA  
            Logger.info(msg)

    def reduce_metrics(self, metrics, average=True):
        reduced_metrics = []
        for m in metrics:
            reduce_metric(m, average)
            reduced_metrics.append(m)
        return reduced_metrics




class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with VRP-SAM ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':==================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou,val_mpa):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f. mPA: %5.2f\n' % (epoch, val_miou,val_mpa))

    @classmethod
    def save_sam_model_miou(cls, model, epoch, val_miou,val_mpa):
        model_params = model.model_params()  # 获取整个参数字典
        torch.save({
            'fusion_state_dict': model_params['fusion'],  # 从字典中提取 fusion 的状态
            'beta_state_dict': model_params['beta']  # 从字典中提取 beta 的状态
        }, os.path.join(cls.logpath, 'best_model_sam.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f. mPA:%5.2f\n' % (epoch, val_miou,val_mpa))
    
    
    def save_model_final(cls, model,sam_model,epoch, val_miou,val_mpa):
        model_params = sam_model.model_params()  # 获取整个参数字典
        torch.save({
            'fusion_state_dict': model_params['fusion'],  # 从字典中提取 fusion 的状态
            'beta_state_dict': model_params['beta']  # 从字典中提取 beta 的状态
        }, os.path.join(cls.logpath, 'best_model_sam_final.pt'))
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model_final.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f. mPA:%5.2f\n' % (epoch, val_miou,val_mpa))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if [i for i in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] if i in k]:
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))


def custom_average_precision_score(targets, preds):
        """Compute average precision for binary classification (0/1)"""
        # Ensure both targets and preds are numpy arrays
        targets = np.array(targets)
        preds = np.array(preds)
        
        # True positives (TP), false positives (FP), false negatives (FN)
        acc1 = np.sum(np.logical_and(preds == 1, targets == 1))
        acc2 = np.sum(np.logical_and(preds == 0, targets == 0))
        
        return acc1 + acc2