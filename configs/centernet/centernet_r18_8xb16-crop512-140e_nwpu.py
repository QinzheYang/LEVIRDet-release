_base_ = './centernet_r18_8xb16-crop512-140e_coco.py'

# Dataset settings
classes = ('airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle')
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/yangqinzhe/yqz-det/ckpt/NWPU/centernet_r18_8xb16-crop512-140e_vrhv'
dataset_type = 'CocoDataset'
data_root = '/mnt/user/share/yangqinzhe/yqz-det/NWPU/'
backend_args = None

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')

# Update model head class count
model = dict(bbox_head=dict(num_classes=num_classes))

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

train_cfg = dict(max_epochs=28, type='EpochBasedTrainLoop', val_interval=10)
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
train_cfg = dict(val_interval=10)

# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))