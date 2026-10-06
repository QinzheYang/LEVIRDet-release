_base_ = './yolof_r50-c5_8xb8-1x_coco.py'

# Dataset settings
classes = ('ship', )
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/yangqinzhe/yqz-det/ckpt/vrhv/yolof_r50-c5_8xb8-1x_vrhv'
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
        filter_cfg=dict(filter_empty_gt=False),
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
train_cfg = dict(val_interval=10,max_epochs=120)

# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))


optim_wrapper = dict(
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.3333333333333333)),
        norm_decay_mult=0.0),
    type='OptimWrapper')

