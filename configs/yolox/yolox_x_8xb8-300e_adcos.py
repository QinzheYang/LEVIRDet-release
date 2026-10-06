_base_ = './yolox_x_8xb8-300e_coco.py'

# =======================
# 1) 数据集与类别
# =======================
# TODO: 把下面类别名改成你自己的真实类别名
classes = (
    'plane',
)
num_classes = len(classes)
metainfo = dict(classes=classes)

# 你的数据路径
data_root = '/mnt/dataset/share/yangqinzhe/yqz-det/adcos_dataset/'
work_dir='/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/adcos_clean/yolox'
# 采用 COCO 格式数据集
dataset_type = 'CocoDataset'
backend_args = None

# =======================
# 2) 模型类别数
# =======================
model = dict(
    bbox_head=dict(num_classes=num_classes)
)

# =======================
# 3) 训练集
# =======================
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train_clean.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=_base_.train_pipeline)

# =======================
# 4) DataLoader
# =======================
train_dataloader = dict(
    batch_size=4,         # 可按显存调小，比如 4 / 2
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
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
        pipeline=_base_.test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# =======================
# 5) 评估器
# =======================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator