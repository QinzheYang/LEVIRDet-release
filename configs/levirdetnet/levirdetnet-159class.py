_base_ = '../_base_/levirdetnet.py'

import os

# Strict 159-class release configuration.
# Released annotations already use the unified landing-ship label LL.
# The parent vehicle and its fine child car remain separate labels.

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(
        backend='nccl',
        timeout=6000,  # 100 min = 6000000 ms
    ),
)

# LEVIRDet-159 release dataset settings. Validation uses the test split.
work_dir = './work_dirs/levirdetnet-159class'
dataset_type = 'FineCocoDataset'
# Run commands from the project root. Docker sets LEVIR_DATA_ROOT=/data.
data_root = os.environ.get('LEVIR_DATA_ROOT', '../02_release/')
pretrained = 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
category_label_aliases = dict(car='vehicle')
source_class_aliases = dict(vehicle='car')

base_classes = (
    'plane', 'storage_tank', 'tenniscourt', 'baseball_diamond',
    'basketball_court', 'ground_track_field', 'vehicle', 'bridge', 'harbor',
    'ship', 'parking_lot', 'overpass', 'swimming-pool', 'roundabout',
    'soccer-ball-field', 'pylon', 'stadium', 'container', 'container-crane',
    'windmill', 'helipad', 'rugby-count', 'helicoper',
    'Expressway-toll-station', 'chimney', 'dam', 'golffield',
    'trainstation', 'Expressway-Service-area', 'airport')

plane_airliner_classes = (
    'A220', 'A321', 'A330', 'A350', 'ARJ21', 'Boeing737', 'Boeing747',
    'Boeing777', 'Boeing787', 'C919')
plane_notairliner_classes = (
    'A-10', 'A-26', 'B-1', 'B-1B', 'B-2', 'B-29', 'B-52', 'C-130',
    'C-135', 'C-17', 'C-21', 'C-5', 'E-3', 'E-8', 'F-15', 'F-16', 'F-22',
    'F-5', 'FA-18', 'KC-10', 'KC-135', 'P-3C', 'P-63', 'SU-24', 'SU-34',
    'SU-35', 'T-43', 'T-6', 'TU-160', 'TU-22', 'TU-95', 'U-2')
vehicle_fine_classes = (
    'car', 'bus', 'camping_car', 'Cargo Truck', 'dump truck', 'Excavator',
    'other', 'pickup', 'small-vehicle', 'tractor', 'Trailer',
    'Truck Tractor', 'van')
ship_civil_classes = (
    'bargePontoon', 'bulkCarrier', 'Car_carrier', 'coastGuard',
    'Container_Ship', 'Dock', 'dredgerReclamation', 'dredging', 'drill',
    'Engineering_Ship', 'Fishing_Vessel', 'Hovercraft', 'Large_sail_ship',
    'Liquid_Cargo_Ship', 'lpg', 'Merchant', 'offshore', 'oreCarrier',
    'passenger', 'RoRo', 'serviceCraft', 'Small_leisure_craft',
    'small_Ro-Ro_ferry', 'tiny_boat', 'tiny_ship', 'Tugboat', 'yacht')
ship_as_classes = ('Auxiliary Ships', 'Masyuu AS', 'Sanantonio AS')
ship_cruiser_classes = ('other cruiser', 'Ticonderoga')
ship_commander_classes = ('other Commander', 'USS Blue Ridge (LCC-19)')
ship_cv_classes = ('Enterprise', 'Midway', 'Nimitz', 'other Aircraft carrier')
ship_dd_classes = (
    'Arleigh Burke DD', 'Asagiri DD', 'Atago DD', 'Hatsuyuki DD',
    'Hyuga DDH', 'other Destroyer')
ship_ff_classes = ('Perry FF', 'Frigate')
ship_landing_classes = (
    'Austin LL', 'LHA LL', 'LSD_41 LL', 'Osumi LL', 'Wasp LL', 'LL',
    'other landing')
ship_lcs_classes = ('DULI', )

classes = base_classes + (
    'Airliner', *plane_airliner_classes, 'notairliner',
    *plane_notairliner_classes, 'other-airplane', *vehicle_fine_classes,
    'civil_ship', *ship_civil_classes, 'war', 'AOE', 'AS', *ship_as_classes,
    'C', *ship_cruiser_classes, 'commander', *ship_commander_classes, 'CV',
    *ship_cv_classes, 'DD', *ship_dd_classes, 'EPF', 'FF', *ship_ff_classes,
    'Landing', *ship_landing_classes, 'LCS', *ship_lcs_classes,
    'Medical ship', 'other Warship', 'patrolForce', 'Submarine', 'Test ship')

class_hierarchy = {
    'plane': ('Airliner', 'notairliner', 'other-airplane'),
    'Airliner': plane_airliner_classes,
    'notairliner': plane_notairliner_classes,
    'vehicle': vehicle_fine_classes,
    'ship': ('civil_ship', 'war'),
    'civil_ship': ship_civil_classes,
    'war': (
        'AOE', 'AS', 'C', 'commander', 'CV', 'DD', 'EPF', 'FF', 'Landing',
        'LCS', 'Medical ship', 'other Warship', 'patrolForce', 'Submarine',
        'Test ship'),
    'AS': ship_as_classes,
    'C': ship_cruiser_classes,
    'commander': ship_commander_classes,
    'CV': ship_cv_classes,
    'DD': ship_dd_classes,
    'FF': ship_ff_classes,
    'Landing': ship_landing_classes,
    'LCS': ship_lcs_classes,
}
#classes = (
#    'airplane','baseball_diamond','basketball_court','bridge','ground track field',
#    'harbor','parking_lot','ship','storage_tank','tennis_court','car','overpass',
#    'roundabout','swimming-pool','rugby-count','stadium','helicoper')

filter_empty_imgs=True
keep_empty_imgs=False

metainfo = dict(classes=classes, class_hierarchy=class_hierarchy)
multiscale_sizes = [896, 960, 1024, 1024, 1024, 1088, 1152]
max_epochs = 192
#load_from = 'epoch_117_hierarchy_pretrain_vehicle_ll.pth'
# Start a new run by default; pass --resume to continue an existing run.
resume = False
stage2_switch_epoch = 13
stage3_switch_epoch = 96
stage4_switch_epoch = 166
switch_assigner_epoch = 149
default_hooks = dict(
    checkpoint=dict(
        interval=1, type='CheckpointHook', by_epoch=True,
        max_keep_ckpts = 20, save_last = True,
        save_best = ['coco/weighted_bbox_mAP'],
        rule = 'greater'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

model = dict(
    backbone=dict(weights_path=pretrained),
    type='DEIMV2GSDGuided',
    num_queries=800,
    bbox_head=dict(
        type='HierarchicalDEIMV2Head',
        num_classes=len(classes),
        class_names=classes,
        class_hierarchy=class_hierarchy,
        # Fine labels supervise their full ancestor path. For coarse labels,
        # descendants are ignored rather than counted as false positives.
        descendant_negative_weight=0.0,
        # Siblings are still negatives, but weaker than cross-branch errors.
        sibling_negative_weight=0.25,
        family_negative_weight=0.5,
        collapse_tree_predictions=True,
        use_path_score=True,
        depth_score_factor=0.05),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(
                    type='HierarchicalFocalLossCost',
                    class_names=classes,
                    class_hierarchy=class_hierarchy,
                    weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]),
        switch_assigner=dict(
            switch_epoch=switch_assigner_epoch,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(
                        type='HierarchicalDEIMV2LossCost',
                        class_names=classes,
                        class_hierarchy=class_hierarchy,
                        iou_order_alpha=4.0,
                        weight=1.)
                ]))),
    gsd_cfg=dict(
        enabled=True,
        # `lookup`: read precomputed json; `online`: patch-based predictor
        source='online',
        # Optional offline lookup json for fallback when source='lookup'
        gsd_json='',
        default_gsd=1.0,
        use_log_gsd=True,
        # 'add' (residual) or 'film' (feature-wise scale/shift)
        modulate='film',
        scale=0.5,
        hidden_dim=128,
        # online predictor settings
        ckpt='gsd_fft/best.pt',
        backbone='resnet50',
        embed_dim=256,
        fft_bins=64,
        agg='attn',
        fusion='gated',
        patch_size=224,
        num_patches=5,
        small_img_threshold=512,
        # allow tiny-lr finetuning with detection loss
        online_trainable=True,
        query_group_cfg=dict(
            enabled=True,
            bins=[300, 600, 800],
            hidden_dim=128,
            scale_alpha=0.2,
            density_alpha=0.2,
            loss_weight=0.05,
            query_per_gt=3.0,
            small_obj_weight=15.0,
            density_weight=0.2,
            scale_bonus_weight=8.0,
            scale_ref=0.08)),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations='nearest',
                random_sizes=multiscale_sizes)
        ]))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(gsd_predictor=dict(lr_mult=0.01))))

train_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=keep_empty_imgs),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=keep_empty_imgs),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage2 = [
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False),
                dict(type='Expand', mean=[0, 0, 0]),
                dict(
                    type='RandomApply',
                    transforms=dict(
                        type='MinIoURandomCrop',
                        cover_all_box=False,
                        trials=40),
                    prob=0.8),
                dict(
                    type='FilterAnnotations',
                    min_gt_bbox_wh=(1, 1),
                    keep_empty=keep_empty_imgs),
                dict(type='Resize', scale=(1024, 1024), keep_ratio=False)
            ],
            [
                dict(
                    type='Mosaic',
                    img_scale=(512, 512),
                    center_ratio_range=(1.0, 1.0),
                    pad_val=0),
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.5, 1.5),
                    max_shear_degree=0,
                    border_val=(0, 0, 0),
                    center=None),
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False)
            ],
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=keep_empty_imgs),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage3 = [
    dict(
        type='PhotoMetricDistortion',
        hue_delta=12.75,
        clip_val=255,
        force_float32=False),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(
            type='MinIoURandomCrop', cover_all_box=False, trials=40),
        prob=0.8),
    *train_pipeline,
]
train_pipeline_stage4 = train_pipeline

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(
            type='BatchRandomChoice',
            transforms=[
                [dict(type='BatchMixup', ratio_range=(0.45, 0.55))],
                [
                    dict(
                        type='BatchCopyBlend',
                        area_threshold=100,
                        num_objects=3,
                        with_expand=True,
                        expand_ratios=(0.1, 0.25),
                        ratio_range=(0.45, 0.55),
                        prob=0.5)
                ],
            ]),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=multiscale_sizes)
    ],
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage3 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(
            type='BatchCopyBlend',
            area_threshold=100,
            num_objects=3,
            with_expand=True,
            expand_ratios=(0.1, 0.25),
            ratio_range=(0.45, 0.55),
            prob=0.5),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=multiscale_sizes)
    ],
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage4 = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/train_159.json',
            data_prefix=dict(img='train/images/'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            metainfo=metainfo,
            category_label_aliases=category_label_aliases,
            ignore_crowd=False,
            filter_cfg=dict(filter_empty_gt=filter_empty_imgs, min_size=1,max_num_instances=1000)),
        pipeline=train_pipeline,
        deepcopy=False))


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test_159.json',
        data_prefix=dict(img='test/images/'),
        metainfo=metainfo,
        category_label_aliases=category_label_aliases,
        ignore_crowd=False,
        filter_cfg=dict(max_num_instances=1000, apply_in_test=True),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HierarchicalCocoMetric',
    ann_file=None,
    metric='bbox',
    classwise=True,
    class_names=classes,
    class_hierarchy=class_hierarchy,
    base_classes=base_classes,
    coarse_30_weight=0.7,
    coarse_all_weight=0.3)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=5)#######
param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=False, begin=0, end=2000),
    dict(
        type='CosineAnnealingLR',
        begin=stage3_switch_epoch,
        end=stage4_switch_epoch,
        by_epoch=True,
        eta_min_ratio=0.5,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=stage4_switch_epoch)
]

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_pipeline=train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_pipeline=train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage4)
]
