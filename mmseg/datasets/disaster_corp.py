# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
import h5py
import numpy as np
import cv2
from .pipelines import Compose, LoadAnnotations
import pdb
from collections import OrderedDict
from mmseg.utils import get_root_logger
from mmcv.utils import print_log
from mmseg.models.losses import accuracy
from prettytable import PrettyTable

def gen_mask(img, loc=None):
    if loc is None:
        w, h = img.shape[:2]
        assert (img[0, 0] == img[w-1, h-1]).all()
        mask = img!=img[0,0]
    else:
        mask = img!=img[loc[0], loc[1]]
    if img.dtype == 'int8':
        mask[img==-128]=False
    if len(mask.shape)==3:
        mask = mask.any(-1)
    return mask

@DATASETS.register_module()
class DisasterCropDataset(CustomDataset):
    CLASSES = ('positive', 'negative', )
    PALETTE = [[255, 255, 255], [0, 0, 255]]

    def __init__(self,pipeline, h5_path, ann_path, test_mode=False, sub_size = 128, overlap_rate = 0.1):
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.h5_path=h5_path
        self.ann_path=ann_path
        self.sub_size = sub_size
        self.overlap_rate = overlap_rate
        self.slide = int(sub_size * overlap_rate)
        self.shift = 32
        self.load_annotations(self.h5_path,self.ann_path)

    def load_annotations(self, h5_path, ann_path):
        f = h5py.File(h5_path, 'r')
        self.key_list = []
        self.datas = []
        self.valid_mask = None
        for key in f.keys():
            im = f[key][:]
            mask = gen_mask(im)
            if self.valid_mask is None:
                self.valid_mask = mask
            else:
                self.valid_mask[self.valid_mask!=mask]=0
            self.key_list.append(key)
            self.datas.append(im[:,:,None])
        f.close()
        self.datas = np.concatenate(self.datas, axis=2)
        self.means = self.datas[self.valid_mask].mean(0)
        self.stds = self.datas[self.valid_mask].std(0)
        self.datas[self.valid_mask]= (self.datas[self.valid_mask]-self.means)/ self.stds
        self.datas[~self.valid_mask] = -1
        gt_semantic_seg = cv2.imread(ann_path, -1)
        gt_semantic_seg[gt_semantic_seg == 0] = 255
        gt_semantic_seg = gt_semantic_seg - 1
        gt_semantic_seg[gt_semantic_seg == 254] = 255
        self.anns = gt_semantic_seg
        self.height, self.weight = self.anns.shape
        shift_x = np.arange(np.ceil((self.weight-self.sub_size)/self.slide)+1)*self.slide
        shift_x[-1] = min(shift_x[-1], self.weight - self.sub_size)
        shift_y = np.arange(np.ceil((self.height-self.sub_size)/self.slide)+1)*self.slide
        shift_y[-1] = min(shift_y[-1], self.height - self.sub_size)
        lefts, ups = np.meshgrid(shift_x, shift_y)
        lefts, ups =lefts.reshape(-1).astype(np.int), ups.reshape(-1).astype(np.int) 
        self.img_infos=[]
        for i in range(len(lefts)):
            left = lefts[i]
            up=ups[i]
            if (self.anns[up:up+self.sub_size, left:left+self.sub_size]!=255).sum()>0:
                self.img_infos.append(dict(left=left, up=up))
        #     self.imgs.append(self.datas[up:up+self.sub_size, left:left+self.sub_size])
        #     self.gt_semantic_segs.append(self.anns[up:up+self.sub_size, left:left+self.sub_size])
        # pdb.set_trace()
        # for i in range(len(self.img_infos)):
        #     for j in range(len(self.key_list)):
        #         up = self.img_infos[i]['up']
        #         left = self.img_infos[i]['left']

        #         key = self.key_list[j]
        #         img = self.datas[up:up+self.sub_size, left:left+self.sub_size,j].copy()
        #         mask = img!=-1
        #         # pdb.set_trace()
        #         work_dir =  r'D:\liu601\project\mmsegmentation\work_dirs\show_data'
        #         if mask.sum()>0:
        #             img[mask] = (img[mask]-img[mask].min())/(img[mask].max() - img[mask].min())*255
        #         gt_semantic_seg=self.anns[up:up+self.sub_size, left:left+self.sub_size].copy()
        #         img[~mask] = 128
        #         img_uint8 = img[:,:,None].repeat(3,2).astype(np.uint8)
        #         img_uint8[gt_semantic_seg==0]=np.array([255,0,0])
        #         img_uint8[gt_semantic_seg==1]=np.array([255,255,0])
        #         path = work_dir + '/{}_{}.png'.format(i, key)
        #         cv2.imwrite(path, img_uint8)
        # pdb.set_trace()

        print_log(f'Loaded {len(self.img_infos)} images', logger=get_root_logger())

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
        
    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx])
        left = self.img_infos[idx]['left']+np.random.randint(-self.shift, self.shift)
        up = self.img_infos[idx]['up'] + np.random.randint(-self.shift, self.shift)
        results['seg_fields'] = []
        img = self.datas[up:up+self.sub_size, left:left+self.sub_size]
        shape = img.shape
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        results['pad_shape'] = shape
        results['scale_factor'] = 1.0
        
        results['gt_semantic_seg'] = self.anns[up:up+self.sub_size, left:left+self.sub_size]
        results['seg_fields'].append('gt_semantic_seg')
        return self.pipeline(results)
    def prepare_test_img(self, idx):
        results = dict(img_info=self.img_infos[idx])
        left = self.img_infos[idx]['left']
        up = self.img_infos[idx]['up']
        results['seg_fields'] = []
        img = self.datas[up:up+self.sub_size, left:left+self.sub_size]
        shape = img.shape
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        results['pad_shape'] = shape
        results['scale_factor'] = 1.0
        return self.pipeline(results)
    
    def evaluate(self,
                 results,
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.

            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        results = np.array(results)
        num_acc, num_sample = results.sum(0)
        eval_results = {}
        ret_metrics_summary=OrderedDict({'acc': np.round(num_acc/num_sample * 100, 2)})
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)
        for key, value in ret_metrics_summary.items():
            eval_results[key] = value
        return eval_results

    def pre_eval(self, preds, indices):
        pre_eval_results = []
        for pred, index in zip(preds, indices):
            img_info=self.img_infos[index]
            ann=self.anns[img_info['up']:img_info['up']+self.sub_size,img_info['left']:img_info['left']+self.sub_size]
            num_sample = (ann!=255).sum()
            num_acc = (pred[ann!=255]==ann[ann!=255]).sum()
            pre_eval_results.append([num_acc,num_sample])
        return pre_eval_results