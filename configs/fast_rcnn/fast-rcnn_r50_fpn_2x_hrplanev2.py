_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py'

# Dataset settings
classes = ('plane', )
num_classes = len(classes)
metainfo = dict(classes=classes)

work_dir = '/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/hrplanev2/fast-rcnn_r50_fpn_2x_vrhv'
dataset_type = 'CocoDataset'
data_root = '/mnt/user/share/yangqinzhe/yqz-det/HRPLANEv2/'
backend_args = None

# Update model head class count
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=8,
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
        test_mode=True,
        backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
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