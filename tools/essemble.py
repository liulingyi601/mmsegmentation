from mmseg.apis.inference import init_segmentor, inference_segmentor
from glob import glob
from mmseg.datasets import build_dataloader, build_dataset
import mmcv
import pdb
from mmseg.datasets.pipelines import Compose
import torch
from mmcv.parallel import collate, scatter

config = r'D:\liu601\project\mmsegmentation\config_disaster\deeplabv3plus_r18_128_200k.py'
model_dir = r'D:\liu601\project\mmsegmentation\work_dirs\deeplabv3plus_r18_128_200k'
cfg = mmcv.Config.fromfile(config)
model_list = glob(model_dir + '\*.pth')
dataset = build_dataset(cfg.data.test)
data = [dataset[0]]
data = collate([data], samples_per_gpu=1)
anns = dataset.anns
accs = []
heat_map = 0
for model_path in model_list:
    model = init_segmentor(config, model_path)
    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data[0])
        pre = result.argmax(1)[0]
        acc = (pre==anns)[anns!=255].sum()/(anns!=255).sum()
        accs.append(acc)
        heat_map += result*acc
heat_map=heat_map / sum(accs)
pre = heat_map.argmax(1)[0]
acc = (pre==anns)[anns!=255].sum()/(anns!=255).sum()
dataset.format_results(heat_map, model_dir, [0])
pdb.set_trace()