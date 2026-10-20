"""Template used by tools/transfer_ckpt.py for a new fine-tuning run.

The transfer script writes a configured copy and reordered COCO annotation
copies. Their categories follow ``classes`` exactly; original data is untouched.
Run the generated configuration from the project root after checkpoint transfer.
"""

_base_ = './levirdetnet-30class.py'

import os

# tools/transfer_ckpt.py replaces these simple assignments in its generated copy.
classes = ('plane', 'ship')
data_root = './data/my_dataset'
train_ann_file = 'work_dirs/finetune/annotations/train.json'
val_ann_file = 'work_dirs/finetune/annotations/val.json'
train_image_dir = 'train/images/'
val_image_dir = 'val/images/'
load_from = 'work_dirs/finetune/init.pth'
work_dir = 'work_dirs/finetune'
max_epochs = 48
batch_size = 2
base_lr = 0.0001

num_workers = 2
metainfo = dict(classes=classes)
resume = False
launcher = 'none'
auto_scale_lr = dict(enable=False, base_batch_size=32)

# The transferred checkpoint already supplies the DINO and GSD parameters.
# Keep the original architecture, while replacing its class-dependent layers.
model = dict(
    backbone=dict(weights_path=None, init_cfg=None, finetune=True),
    bbox_head=dict(num_classes=len(classes)),
    gsd_cfg=dict(ckpt=''),
    data_preprocessor=dict(batch_augments=None),
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', weight=2.0, iou_mode='giou'),
            ])))

# dn_cfg.num_classes is populated by the detector; do not set it manually.
# Replace the 192-epoch EMA and augmentation switch hooks from the source config.
custom_hooks = [dict(type='SetEpochInfoHook')]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

# Absolute annotation paths avoid prepending data_root to generated JSON files.
# file_name in each COCO image record is relative to the corresponding image dir.
train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=os.path.abspath(train_ann_file),
        data_prefix=dict(img=train_image_dir),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline))
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=os.path.abspath(val_ann_file),
        data_prefix=dict(img=val_image_dir),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=os.path.abspath(val_ann_file),
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # One cosine period, starting at the base learning rate. This also
        # restores that rate when a tiny dataset has only one warmup iteration.
        type='CosineRestartLR',
        eta_min_ratio=0.1,
        begin=1,
        end=max_epochs,
        periods=[max_epochs - 1],
        restart_weights=[1],
        by_epoch=True,
        convert_to_iter_based=True),
]

# Replace inherited parameter-specific learning rates as well as the optimizer.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.000125),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            'backbone.dinov3': dict(lr_mult=0.1),
            'gsd_predictor': dict(lr_mult=0.01),
        }))
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=3,
        save_last=True,
        save_best='coco/bbox_mAP',
        rule='greater'))
