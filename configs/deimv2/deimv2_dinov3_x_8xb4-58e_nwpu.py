_base_ = './deimv2_dinov3_x_8xb4-58e_coco.py'

# dataset settings
# NWPU-VHR-10 in COCO format:
#   - images: data/NWPU-VHR-10/imgs
#   - annotations: data/NWPU-VHR-10/annotations/*.json
work_dir = '/mnt/dataset/wanglubo/yqz-tmp/GID6/deimv2'
custom_imports = dict(
    imports=[
        'mmdet.models.detectors.deimv2',
        'mmdet.models.task_modules.assigners.match_cost'
    ],
    allow_failed_imports=False)

dataset_type = 'CocoDataset'
data_root = 'data/NWPU-VHR-10/'
classes = (
    'airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
    'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle')
metainfo = dict(classes=classes)

model = dict(bbox_head=dict(num_classes=len(classes)))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        dataset=dict(
            _delete_=True,
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/NWPU_instances_train.json',
            data_prefix=dict(img='imgs/'),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=1))))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/NWPU_instances_val.json',
        data_prefix=dict(img='imgs/'),
        metainfo=metainfo))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/NWPU_instances_val.json')
test_evaluator = val_evaluator