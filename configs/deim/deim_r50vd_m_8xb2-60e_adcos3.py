_base_ = './deim_r50vd_m_8xb2-60e_coco.py'

# Dataset settings
classes = ('plane', )
num_classes = len(classes)
metainfo = dict(classes=classes)
work_dir = '/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/adcos_clean/deim_60e_1024'
dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/adcos_dataset/'
backend_args = None

# Update model head class count

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/user/wanglubo/rtdetr-mmdet/configs/deim/resnet50vd_ssld_v2_pretrained_edfe4074.pth')),
    bbox_head=dict(num_classes=num_classes))
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train_clean.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=backend_args),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            backend_args=backend_args),
        pipeline={{_base_.train_pipeline}},
        deepcopy=False))

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

# keep at most 5 checkpoints and also save best checkpoint by bbox mAP
# NOTE: metric key usually is 'coco/bbox_mAP' for CocoMetric
# if your environment logs a different key, adjust `save_best` accordingly.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))


#resume=True


# learning policy
max_epochs = 60
train_cfg = dict(max_epochs=max_epochs)

train_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage2 = [
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False),
                dict(type='Expand', mean=[0, 0, 0]),
                dict(
                    type='RandomApply',
                    transforms=dict(
                        type='MinIoURandomCrop',
                        cover_all_box=False,
                        trials=40),
                    prob=0.8),
                dict(
                    type='FilterAnnotations',
                    min_gt_bbox_wh=(1, 1),
                    keep_empty=False),
                dict(type='Resize', scale=(1024, 1024), keep_ratio=False)
            ],
            [
                dict(
                    type='Mosaic',
                    img_scale=(640, 640),
                    center_ratio_range=(1.0, 1.0),
                    pad_val=0),
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.5, 1.5),
                    max_shear_degree=0,
                    border_val=(0, 0, 0),
                    center=None),
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False)
            ],
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage3 = [
    dict(
        type='PhotoMetricDistortion',
        hue_delta=12.75,
        clip_val=255,
        force_float32=False),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(
            type='MinIoURandomCrop', cover_all_box=False, trials=40),
        prob=0.8),
    *train_pipeline,
]
train_pipeline_stage4 = train_pipeline

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(type='BatchMixup', ratio_range=(0.45, 0.55), prob=0.5)
    ] + _base_.model.data_preprocessor.batch_augments,
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage3 = _base_.model.data_preprocessor
data_preprocessor_stage4 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)


stage2_switch_epoch = 4
stage3_switch_epoch = 34
stage4_switch_epoch = 58
custom_hooks = [
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_pipeline=train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_pipeline=train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage4)
]

param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=False, begin=0, end=2000),
    dict(
        type='CosineAnnealingLR',
        begin=stage3_switch_epoch,
        end=stage4_switch_epoch,
        by_epoch=True,
        eta_min_ratio=0.5,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR', by_epoch=True, factor=1, begin=stage4_switch_epoch)
]
