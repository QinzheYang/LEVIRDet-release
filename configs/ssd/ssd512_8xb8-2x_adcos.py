_base_ = './ssd512_coco.py'

# Dataset settings
classes = ('plane', )
num_classes = len(classes)
metainfo = dict(classes=classes)

dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/adcos_dataset/'
backend_args = None
work_dir='/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/adcos_clean/ssd'
# Update head class count
model = dict(bbox_head=dict(num_classes=num_classes))

# Train/val dataloaders (follow SSD repeat-dataset style)
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train_clean.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_pipeline}},
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val_clean.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_clean.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]