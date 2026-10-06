_base_ = './centernet_r18_8xb16-crop512-140e_coco.py'

# Dataset settings
classes = ('ship', )
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/yangqinzhe/yqz-det/ckpt/vrhv/centernet_r18_8xb16-crop512-140e_vrhv'
dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/vrhv/'
backend_args = None

# Update model head class count

model = dict(
    backbone=dict(init_cfg=dict(checkpoint='/mnt/user/wanglubo/levirdet-mmdet/configs/centernet/resnet18-f37072fd.pth', type='Pretrained')),
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline={{_base_.train_pipeline}},
        backend_args=backend_args))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test_coco.json',
        data_prefix=dict(img='test/'),
        backend_args=backend_args))

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    ann_file=data_root + 'test_coco.json',
    metric='bbox',
    backend_args=backend_args)

# Validate every 10 epochs
train_cfg = dict(max_epochs=280, type='EpochBasedTrainLoop', val_interval=10)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=280,
        gamma=0.1,
        milestones=[
            180,
            240,
        ],
        type='MultiStepLR'),
]
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))