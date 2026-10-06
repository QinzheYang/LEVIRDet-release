_base_ = './retinanet_r50_fpn_1x_coco.py'

# Dataset settings
classes = ('plane',)
num_classes = len(classes)
metainfo = dict(classes=classes)
work_dir='/mnt/dataset/yangqinzhe/yqz-det/ckpt/skyfusion-plane/retinanet'
dataset_type = 'CocoDataset'
data_root = '/mnt/user/share/yangqinzhe/yqz-det/skyfusion-plane/'
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
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        backend_args=backend_args))

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox',
    backend_args=backend_args)

# Validate every 10 epochs


# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=10)