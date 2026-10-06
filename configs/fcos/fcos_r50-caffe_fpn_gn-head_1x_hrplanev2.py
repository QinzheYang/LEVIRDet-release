_base_ = './fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# Dataset settings
classes = ('plane', )
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/hrplanev2/fcos_r50-caffe_fpn_gn-head_1x_ucas'
dataset_type = 'CocoDataset'
data_root = '/mnt/user/share/yangqinzhe/yqz-det/HRPLANEv2/'
backend_args = None

# Update model head class count
model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
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
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=backend_args))

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox',
    backend_args=backend_args)

#resume = True

# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=10)
# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=120,
        by_epoch=True,
        milestones=[80, 110],
        gamma=0.1)
]