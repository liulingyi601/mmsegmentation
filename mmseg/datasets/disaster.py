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
class DisasterDataset(CustomDataset):

    CLASSES = (
        'positive', 'negative', )

    PALETTE = [[255, 255, 255], [0, 0, 255]]

    def __init__(self,pipeline, h5_path, ann_path, test_mode=False):
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.h5_path=h5_path
        self.ann_path=ann_path
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
        self.img_infos = []
        self.img_infos.append(dict(left=0, up=0))
        # pdb.set_trace()
        #  cv2.copyMakeBorder(self.datas,0,1000,0,1000,cv2.BORDER_CONSTANT,value=-1)
        # import pdb
        # pdb.set_trace()
        # pdb.set_trace()
        # self.imgs = []
        # self.gt_semantic_segs=[]
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
        results['seg_fields'] = []
        img = self.datas[:,:].copy()
        shape = img.shape
        # print(shape)
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        results['pad_shape'] = shape
        results['scale_factor'] = 1.0
        
        results['gt_semantic_seg'] = self.anns[:,:].copy()
        results['seg_fields'].append('gt_semantic_seg')
        return self.pipeline(results)
    def prepare_test_img(self, idx):
        results = dict(img_info=self.img_infos[idx])
        results['seg_fields'] = []
        img = self.datas[:,:].copy()
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
    def format_results(self, results, imgfile_prefix, indices, **format_args):
        results_path = []
        for result, index in zip(results, indices):
            # pdb.set_trace()
            pre = result.argmax(0)
            pre[pre==1]=255
            pre[~self.valid_mask]=128
            pre = pre[:,:,None].repeat(3,2)
            pre[self.anns==0]=np.array([255,0,0])
            pre[self.anns==1]=np.array([0,0,255])
            path = imgfile_prefix + '/pre_{}.png'.format(index)
            cv2.imwrite(path, pre.astype(np.uint8))
            heat_map = cv2.applyColorMap((result[1]*255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_map[~self.valid_mask]=np.array([128,128,128])
            heat_map[self.anns==0]=np.array([255,0,0])
            heat_map[self.anns==1]=np.array([0,0,255])
            heat_path = imgfile_prefix + '/heat_{}.png'.format(index)
            cv2.imwrite(heat_path, heat_map)
            results_path.append(path)
            
            # pdb.set_trace()
        return results_path
    def pre_eval(self, logics, indices):
        pre_eval_results = []
        # pdb.set_trace()
        for logic, index in zip(logics, indices):
            pred = logic.argmax(0)
            ann=self.anns
            num_sample = (ann!=255).sum()
            num_acc = (pred[ann!=255]==ann[ann!=255]).sum()
            pre_eval_results.append([num_acc,num_sample])
            # pdb.set_trace()
        return pre_eval_results