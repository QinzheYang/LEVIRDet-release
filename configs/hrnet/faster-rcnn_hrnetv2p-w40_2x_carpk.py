_base_ = './faster-rcnn_hrnetv2p-w40_2x_coco.py'

# Dataset settings
classes = ('car',)
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/CARPK/hrnet'
dataset_type = 'CocoDataset'
data_root = '/mnt/user/share/yangqinzhe/yqz-det/CARPK/'
backend_args = None

# Update model head class count
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline={{_base_.train_pipeline}},
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# learning policy
max_epochs = 48
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]
