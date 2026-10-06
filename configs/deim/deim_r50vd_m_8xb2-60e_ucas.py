_base_ = './deim_r50vd_m_8xb2-60e_coco.py'

# Dataset settings
classes = ('plane','car' )
num_classes = len(classes)
metainfo = dict(classes=classes)
work_dir='/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/ucas/deim'
dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/ucas_o2h/'
backend_args = None

# Update model head class count
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/user/wanglubo/rtdetr-mmdet/configs/deim/resnet50vd_ssld_v2_pretrained_edfe4074.pth')),
    bbox_head=dict(num_classes=num_classes))


train_dataloader = dict(
    batch_size=2,
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
            ann_file='train.json',
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