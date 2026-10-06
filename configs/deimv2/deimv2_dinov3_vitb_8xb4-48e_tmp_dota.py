_base_ = './deimv2_dinov3_x_8xb4-58e_coco.py'

# We use DINOv3 as backbone, you can download them following the guide
# in [DINOv3](https://github.com/facebookresearch/dinov3).
pretrained = '/mnt/user/wanglubo/rtdetr-mmdet/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

# Save checkpoint every epoch, no validation.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_last=True),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

switch_assigner_epoch = 37
resume = True
work_dir = '/mnt/dataset/share/yangqinzhe/yqz-det/ckpt/dota/deimv2-vit-tmp_gsd'

dataset_type = 'CocoDataset'
data_root = '/mnt/dataset/wanglubo/yqz-tmp/DOTA/'
classes = (
    'airport','baseball-diamond','basketball-court','bridge','container-crane',
    'ground-track-field','harbor','helicopter','helipad','large-vehicle',
    'plane','roundabout','ship','small-vehicle','soccer-ball-field',
    'storage-tank','swimming-pool','tennis-court')
metainfo = dict(classes=classes)

model = dict(
    backbone=dict(
        name='dinov3_vitb16', weights_path=pretrained, conv_inplane=128),
    bbox_head=dict(num_classes=len(classes)),
    train_cfg=dict(switch_assigner=dict(switch_epoch=switch_assigner_epoch)))

backbone_lr_mult = 0.01
custom_keys = {
    'in_proj_bias': dict(decay_mult=0),
    'backbone.dinov3': dict(lr_mult=backbone_lr_mult),
    'backbone.dinov3.norm.weight': dict(
        lr_mult=backbone_lr_mult, decay_mult=0),
    'backbone.dinov3.norm.bias': dict(lr_mult=backbone_lr_mult, decay_mult=0),
    'backbone.dinov3.patch_embed.proj.bias': dict(
        lr_mult=backbone_lr_mult, decay_mult=0),
}
custom_keys.update({
    f'backbone.dinov3.blocks.{bid}.{name}':
    dict(lr_mult=backbone_lr_mult, decay_mult=0)
    for name in [
        'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
        'attn.qkv.bias', 'attn.qkv.bias_mask', 'attn.proj.bias',
        'mlp.fc1.bias', 'mlp.fc2.bias'
    ] for bid in range(12)
})
custom_keys.update({
    f'decoder.layers.{lid}.norms.{i}.scale': dict(decay_mult=0)
    for lid in range(_base_.num_layers) for i in range(3)
})

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(_delete_=True, **custom_keys)))

max_epochs = 48
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)

# Disable validation/testing loops completely.

# Disable validation during training, but keep test dataloader for inferencer metadata loading.
val_cfg = None
val_dataloader = None
val_evaluator = None

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline={{_base_.test_pipeline}}))

test_evaluator = dict(ann_file=data_root + 'train.json')


stage2_switch_epoch = 4
stage3_switch_epoch = 24
stage4_switch_epoch = 42
custom_hooks = [
    dict(type='SetEpochInfoHook'),  # for DEIMV2 assigner switch
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage4)
]

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

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        dataset=dict(
            type=dataset_type,
            _delete_=True,
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=1))))