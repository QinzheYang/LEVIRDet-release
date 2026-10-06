_base_ = '../_base_/levirdetnet.py'

import os

# 30-class DFINEHead configuration, preserved from gsd_det (2).py.
# Model, optimizer and schedule are preserved; paths use the release layout.

classes = (
    'plane',
    'storage_tank',
    'tenniscourt',
    'baseball_diamond',
    'basketball_court',
    'ground_track_field',
    'car',
    'bridge',
    'harbor',
    'ship',
    'parking_lot',
    'overpass',
    'swimming-pool',
    'roundabout',
    'soccer-ball-field',
    'pylon',
    'stadium',
    'container',
    'container-crane',
    'windmill',
    'helipad',
    'rugby-count',
    'helicoper',
    'Expressway-toll-station',
    'chimney',
    'dam',
    'golffield',
    'trainstation',
    'Expressway-Service-area',
    'airport',
)
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(
        ema_type='ExpMomentumEMA',
        gamma=1000,
        momentum=0.0001,
        priority=49,
        restart_epoch=166,
        type='EMADynamicMomentumHook',
        update_buffers=True),
    dict(
        switch_epoch=13,
        switch_pipeline=[
            dict(
                transforms=[
                    [
                        dict(
                            clip_val=255,
                            force_float32=False,
                            hue_delta=12.75,
                            type='PhotoMetricDistortion'),
                        dict(mean=[
                            0,
                            0,
                            0,
                        ], type='Expand'),
                        dict(
                            prob=0.8,
                            transforms=dict(
                                cover_all_box=False,
                                trials=40,
                                type='MinIoURandomCrop'),
                            type='RandomApply'),
                        dict(
                            keep_empty=False,
                            min_gt_bbox_wh=(
                                1,
                                1,
                            ),
                            type='FilterAnnotations'),
                        dict(
                            keep_ratio=False,
                            scale=(
                                1024,
                                1024,
                            ),
                            type='Resize'),
                    ],
                    [
                        dict(
                            center_ratio_range=(
                                1.0,
                                1.0,
                            ),
                            img_scale=(
                                512,
                                512,
                            ),
                            pad_val=0,
                            type='Mosaic'),
                        dict(
                            border_val=(
                                0,
                                0,
                                0,
                            ),
                            center=None,
                            max_shear_degree=0,
                            scaling_ratio_range=(
                                0.5,
                                1.5,
                            ),
                            type='RandomAffine'),
                        dict(
                            clip_val=255,
                            force_float32=False,
                            hue_delta=12.75,
                            type='PhotoMetricDistortion'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
    dict(
        switch_epoch=96,
        switch_pipeline=[
            dict(
                clip_val=255,
                force_float32=False,
                hue_delta=12.75,
                type='PhotoMetricDistortion'),
            dict(mean=[
                0,
                0,
                0,
            ], type='Expand'),
            dict(
                prob=0.8,
                transforms=dict(
                    cover_all_box=False, trials=40, type='MinIoURandomCrop'),
                type='RandomApply'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
    dict(
        switch_epoch=166,
        switch_pipeline=[
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
    dict(
        switch_data_preprocessor=dict(
            batch_augments=[
                dict(
                    transforms=[
                        [
                            dict(
                                ratio_range=(
                                    0.45,
                                    0.55,
                                ), type='BatchMixup'),
                        ],
                        [
                            dict(
                                area_threshold=100,
                                expand_ratios=(
                                    0.1,
                                    0.25,
                                ),
                                num_objects=3,
                                prob=0.5,
                                ratio_range=(
                                    0.45,
                                    0.55,
                                ),
                                type='BatchCopyBlend',
                                with_expand=True),
                        ],
                    ],
                    type='BatchRandomChoice'),
                dict(
                    interpolations='nearest',
                    interval=1,
                    random_sizes=[
                        896,
                        960,
                        1024,
                        1024,
                        1024,
                        1088,
                        1152,
                    ],
                    type='BatchSyncRandomResize'),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_size_divisor=1,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='DetDataPreprocessor'),
        switch_epoch=13,
        type='DataPreprocessorSwitchHook'),
    dict(
        switch_data_preprocessor=dict(
            batch_augments=[
                dict(
                    area_threshold=100,
                    expand_ratios=(
                        0.1,
                        0.25,
                    ),
                    num_objects=3,
                    prob=0.5,
                    ratio_range=(
                        0.45,
                        0.55,
                    ),
                    type='BatchCopyBlend',
                    with_expand=True),
                dict(
                    interpolations='nearest',
                    interval=1,
                    random_sizes=[
                        896,
                        960,
                        1024,
                        1024,
                        1024,
                        1088,
                        1152,
                    ],
                    type='BatchSyncRandomResize'),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_size_divisor=1,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='DetDataPreprocessor'),
        switch_epoch=96,
        type='DataPreprocessorSwitchHook'),
    dict(
        switch_data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_size_divisor=1,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='DetDataPreprocessor'),
        switch_epoch=166,
        type='DataPreprocessorSwitchHook'),
]
data_preprocessor_stage2 = dict(batch_augments=[
    dict(
        transforms=[
            [
                dict(ratio_range=(
                    0.45,
                    0.55,
                ), type='BatchMixup'),
            ],
            [
                dict(
                    area_threshold=100,
                    expand_ratios=(
                        0.1,
                        0.25,
                    ),
                    num_objects=3,
                    prob=0.5,
                    ratio_range=(
                        0.45,
                        0.55,
                    ),
                    type='BatchCopyBlend',
                    with_expand=True),
            ],
        ],
        type='BatchRandomChoice'),
    dict(
        interpolations='nearest',
        interval=1,
        random_sizes=[
            896,
            960,
            1024,
            1024,
            1024,
            1088,
            1152,
        ],
        type='BatchSyncRandomResize'),
])
data_preprocessor_stage3 = dict(batch_augments=[
    dict(
        area_threshold=100,
        expand_ratios=(
            0.1,
            0.25,
        ),
        num_objects=3,
        prob=0.5,
        ratio_range=(
            0.45,
            0.55,
        ),
        type='BatchCopyBlend',
        with_expand=True),
    dict(
        interpolations='nearest',
        interval=1,
        random_sizes=[
            896,
            960,
            1024,
            1024,
            1024,
            1088,
            1152,
        ],
        type='BatchSyncRandomResize'),
])
# Run commands from the project root. Docker sets LEVIR_DATA_ROOT=/data.
data_root = os.environ.get('LEVIR_DATA_ROOT', '../02_release/')
pretrained = 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        max_keep_ckpts=10,
        rule='greater',
        save_best=[
            'coco/bbox_mAP',
        ],
        save_last=True,
        type='CheckpointHook'))
env_cfg = dict(dist_cfg=dict(timeout=6000))
filter_empty_imgs = True
keep_empty_imgs = False
launcher = 'pytorch'
max_epochs = 192
metainfo = dict(
    classes=(
        'plane',
        'storage_tank',
        'tenniscourt',
        'baseball_diamond',
        'basketball_court',
        'ground_track_field',
        'car',
        'bridge',
        'harbor',
        'ship',
        'parking_lot',
        'overpass',
        'swimming-pool',
        'roundabout',
        'soccer-ball-field',
        'pylon',
        'stadium',
        'container',
        'container-crane',
        'windmill',
        'helipad',
        'rugby-count',
        'helicoper',
        'Expressway-toll-station',
        'chimney',
        'dam',
        'golffield',
        'trainstation',
        'Expressway-Service-area',
        'airport',
    ))
model = dict(
    backbone=dict(weights_path=pretrained),
    bbox_head=dict(num_classes=30),
    data_preprocessor=dict(batch_augments=[
        dict(
            interpolations='nearest',
            interval=1,
            random_sizes=[
                896,
                960,
                1024,
                1024,
                1024,
                1088,
                1152,
            ],
            type='BatchSyncRandomResize'),
    ]),
    gsd_cfg=dict(
        agg='attn',
        backbone='resnet50',
        ckpt='gsd_fft/best.pt',
        default_gsd=1.0,
        embed_dim=256,
        enabled=True,
        fft_bins=64,
        fusion='gated',
        gsd_json='',
        hidden_dim=128,
        modulate='film',
        num_patches=5,
        online_trainable=True,
        patch_size=224,
        query_group_cfg=dict(
            bins=[
                300,
                600,
                800,
            ],
            density_alpha=0.2,
            density_weight=0.2,
            enabled=True,
            hidden_dim=128,
            loss_weight=0.05,
            query_per_gt=3.0,
            scale_alpha=0.2,
            scale_bonus_weight=8.0,
            scale_ref=0.08,
            small_obj_weight=15.0),
        scale=0.5,
        small_img_threshold=512,
        source='online',
        use_log_gsd=True),
    num_queries=800,
    train_cfg=dict(switch_assigner=dict(switch_epoch=149)),
    type='DEIMV2GSDGuided')
multiscale_sizes = [
    896,
    960,
    1024,
    1024,
    1024,
    1088,
    1152,
]
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(gsd_predictor=dict(lr_mult=0.01))))
param_scheduler = [
    dict(begin=0, by_epoch=False, end=2000, type='QuadraticWarmupLR'),
    dict(
        begin=96,
        by_epoch=True,
        convert_to_iter_based=True,
        end=166,
        eta_min_ratio=0.5,
        type='CosineAnnealingLR'),
    dict(begin=166, by_epoch=True, factor=1, type='ConstantLR'),
]
# Start a new run by default; pass --resume to continue an existing run.
resume = False
stage2_switch_epoch = 13
stage3_switch_epoch = 96
stage4_switch_epoch = 166
switch_assigner_epoch = 149
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test_30.json',
        data_prefix=dict(img='test/images/'),
        data_root=data_root,
        filter_cfg=dict(apply_in_test=True, max_num_instances=1000),
        metainfo=dict(
            classes=(
                'plane',
                'storage_tank',
                'tenniscourt',
                'baseball_diamond',
                'basketball_court',
                'ground_track_field',
                'car',
                'bridge',
                'harbor',
                'ship',
                'parking_lot',
                'overpass',
                'swimming-pool',
                'roundabout',
                'soccer-ball-field',
                'pylon',
                'stadium',
                'container',
                'container-crane',
                'windmill',
                'helipad',
                'rugby-count',
                'helicoper',
                'Expressway-toll-station',
                'chimney',
                'dam',
                'golffield',
                'trainstation',
                'Expressway-Service-area',
                'airport',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ]))
test_evaluator = dict(
    ann_file=os.path.join(data_root, 'annotations/test_30.json'))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=192, val_interval=5)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            _delete_=True,
            ann_file='annotations/train_30.json',
            data_prefix=dict(img='train/images/'),
            data_root=data_root,
            filter_cfg=dict(
                filter_empty_gt=True, max_num_instances=1000, min_size=1),
            metainfo=dict(
                classes=(
                    'plane',
                    'storage_tank',
                    'tenniscourt',
                    'baseball_diamond',
                    'basketball_court',
                    'ground_track_field',
                    'car',
                    'bridge',
                    'harbor',
                    'ship',
                    'parking_lot',
                    'overpass',
                    'swimming-pool',
                    'roundabout',
                    'soccer-ball-field',
                    'pylon',
                    'stadium',
                    'container',
                    'container-crane',
                    'windmill',
                    'helipad',
                    'rugby-count',
                    'helicoper',
                    'Expressway-toll-station',
                    'chimney',
                    'dam',
                    'golffield',
                    'trainstation',
                    'Expressway-Service-area',
                    'airport',
                )),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='CocoDataset'),
        pipeline=[
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ]))
train_pipeline = [
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(keep_ratio=False, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(
        transforms=[
            [
                dict(
                    clip_val=255,
                    force_float32=False,
                    hue_delta=12.75,
                    type='PhotoMetricDistortion'),
                dict(mean=[
                    0,
                    0,
                    0,
                ], type='Expand'),
                dict(
                    prob=0.8,
                    transforms=dict(
                        cover_all_box=False,
                        trials=40,
                        type='MinIoURandomCrop'),
                    type='RandomApply'),
                dict(
                    keep_empty=False,
                    min_gt_bbox_wh=(
                        1,
                        1,
                    ),
                    type='FilterAnnotations'),
                dict(keep_ratio=False, scale=(
                    1024,
                    1024,
                ), type='Resize'),
            ],
            [
                dict(
                    center_ratio_range=(
                        1.0,
                        1.0,
                    ),
                    img_scale=(
                        512,
                        512,
                    ),
                    pad_val=0,
                    type='Mosaic'),
                dict(
                    border_val=(
                        0,
                        0,
                        0,
                    ),
                    center=None,
                    max_shear_degree=0,
                    scaling_ratio_range=(
                        0.5,
                        1.5,
                    ),
                    type='RandomAffine'),
                dict(
                    clip_val=255,
                    force_float32=False,
                    hue_delta=12.75,
                    type='PhotoMetricDistortion'),
            ],
        ],
        type='RandomChoice'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage3 = [
    dict(
        clip_val=255,
        force_float32=False,
        hue_delta=12.75,
        type='PhotoMetricDistortion'),
    dict(mean=[
        0,
        0,
        0,
    ], type='Expand'),
    dict(
        prob=0.8,
        transforms=dict(
            cover_all_box=False, trials=40, type='MinIoURandomCrop'),
        type='RandomApply'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(keep_ratio=False, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage4 = [
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(keep_ratio=False, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test_30.json',
        data_prefix=dict(img='test/images/'),
        data_root=data_root,
        filter_cfg=dict(apply_in_test=True, max_num_instances=1000),
        metainfo=dict(
            classes=(
                'plane',
                'storage_tank',
                'tenniscourt',
                'baseball_diamond',
                'basketball_court',
                'ground_track_field',
                'car',
                'bridge',
                'harbor',
                'ship',
                'parking_lot',
                'overpass',
                'swimming-pool',
                'roundabout',
                'soccer-ball-field',
                'pylon',
                'stadium',
                'container',
                'container-crane',
                'windmill',
                'helipad',
                'rugby-count',
                'helicoper',
                'Expressway-toll-station',
                'chimney',
                'dam',
                'golffield',
                'trainstation',
                'Expressway-Service-area',
                'airport',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ]))
val_evaluator = dict(
    ann_file=os.path.join(data_root, 'annotations/test_30.json'))
work_dir = './work_dirs/levirdetnet-30class'
