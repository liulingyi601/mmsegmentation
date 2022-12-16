_base_ = './deeplabv3plus_r50-d8_512x512_20k.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        # c1_in_channels=64,
        # c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='DisasterCropDataset'))
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.05)
