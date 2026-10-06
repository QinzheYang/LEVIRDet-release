_base_ = './fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# Dataset settings
classes = ('ship', )
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/yangqinzhe/yqz-det/ckpt/vrhv/fcos_r50-caffe_fpn_gn-head_1x_vrhv'
dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/vrhv/'
backend_args = None

# Update model head class count
model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        backend_args=backend_args))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test_coco.json',
        data_prefix=dict(img='test/'),
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'test_coco.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# Validate every 10 epochs


# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))
resume=True
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=10)
# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=240,
        by_epoch=True,
        milestones=[160, 220],
        gamma=0.1)
]