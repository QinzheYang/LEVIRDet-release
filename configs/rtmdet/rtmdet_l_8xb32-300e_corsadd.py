_base_ = './rtmdet_l_8xb32-300e_coco.py'

# Dataset settings
classes = ('plane',)
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir='/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/corsadd/rtmdet'
dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/corsadd_o2h/'
backend_args = None

# Update model head class count
model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=8,
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
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

resume = True

# Save after validation, keep at most 5 checkpoints, save best
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointAfterValHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'))

train_cfg = dict(val_interval=1)