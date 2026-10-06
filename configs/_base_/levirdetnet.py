# Shared LEVIR-DetNet runtime, DINOv3 backbone and optimizer defaults.
# Expanded from the original DEIMv2 base; no dependency on configs/deimv2.
# Change task-specific settings in configs/levirdetnet/*.py.

act_cfg = dict(inplace=True, type='SiLU')
auto_scale_lr = dict(base_batch_size=32, enable=False)
backbone_lr_mult = 0.02
backbone_norm_multi = dict(decay_mult=1.0, lr_mult=0.1)
backend_args = None
base_dim = 256
base_size_repeat = 3
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(
        ema_type='ExpMomentumEMA',
        gamma=1000,
        momentum=0.0001,
        priority=49,
        restart_epoch=50,
        type='EMADynamicMomentumHook',
        update_buffers=True),
    dict(
        switch_epoch=4,
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
                                640,
                                640,
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
                                320,
                                320,
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
        switch_epoch=29,
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
                640,
                640,
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
        switch_epoch=50,
        switch_pipeline=[
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=False, scale=(
                640,
                640,
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
                        480,
                        512,
                        544,
                        576,
                        608,
                        640,
                        640,
                        640,
                        672,
                        704,
                        736,
                        768,
                        800,
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
        switch_epoch=4,
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
                        480,
                        512,
                        544,
                        576,
                        608,
                        640,
                        640,
                        640,
                        672,
                        704,
                        736,
                        768,
                        800,
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
        switch_epoch=29,
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
        switch_epoch=50,
        type='DataPreprocessorSwitchHook'),
]
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1),
    'backbone.dinov3':
    dict(lr_mult=0.02),
    'backbone.dinov3.blocks.0.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.0.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.1.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.10.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.11.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.2.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.3.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.4.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.5.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.6.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.7.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.8.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.attn.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.attn.qkv.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.attn.qkv.bias_mask':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.mlp.fc1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.mlp.fc2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.norm1.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.norm1.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.norm2.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.blocks.9.norm2.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.norm.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.norm.weight':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.dinov3.patch_embed.proj.bias':
    dict(decay_mult=0, lr_mult=0.02),
    'backbone.stages.0.blocks.0.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.layers.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.layers.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.layers.2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.layers.3.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.layers.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.layers.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.layers.2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.layers.3.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.1.downsample.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.0.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.0.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.1.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.1.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.2.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.2.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.3.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.layers.3.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.0.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.0.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.1.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.1.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.2.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.2.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.3.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.layers.3.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.0.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.0.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.1.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.1.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.2.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.2.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.3.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.layers.3.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.2.downsample.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.aggregation.0.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.aggregation.1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.0.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.0.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.1.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.1.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.2.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.2.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.3.conv1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.layers.3.conv2.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stages.3.downsample.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stem.stem1.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stem.stem2a.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stem.stem2b.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stem.stem3.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.stem.stem4.bn':
    dict(decay_mult=1.0, lr_mult=0.1),
    'decoder.layers.0.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.0.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.0.norms.2.scale':
    dict(decay_mult=0),
    'decoder.layers.1.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.1.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.1.norms.2.scale':
    dict(decay_mult=0),
    'decoder.layers.2.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.2.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.2.norms.2.scale':
    dict(decay_mult=0),
    'decoder.layers.3.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.3.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.3.norms.2.scale':
    dict(decay_mult=0),
    'decoder.layers.4.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.4.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.4.norms.2.scale':
    dict(decay_mult=0),
    'decoder.layers.5.norms.0.scale':
    dict(decay_mult=0),
    'decoder.layers.5.norms.1.scale':
    dict(decay_mult=0),
    'decoder.layers.5.norms.2.scale':
    dict(decay_mult=0),
    'in_proj_bias':
    dict(decay_mult=0)
})
data_preprocessor_stage2 = dict(
    batch_augments=[
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
                480,
                512,
                544,
                576,
                608,
                640,
                640,
                640,
                672,
                704,
                736,
                768,
                800,
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
    type='DetDataPreprocessor')
data_preprocessor_stage3 = dict(
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
                480,
                512,
                544,
                576,
                608,
                640,
                640,
                640,
                672,
                704,
                736,
                768,
                800,
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
    type='DetDataPreprocessor')
data_preprocessor_stage4 = dict(
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
    type='DetDataPreprocessor')
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointAfterValHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_idx = -1
interpolations = [
    'nearest',
    'bilinear',
    'bicubic',
    'area',
    'lanczos',
]
layer_scale = 1.0
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 58
model = dict(
    as_two_stage=True,
    backbone=dict(
        conv_inplane=64,
        finetune=True,
        freeze_at=0,
        freeze_norm=True,
        hidden_dim=256,
        init_cfg=dict(
            checkpoint=
            'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth',
            type='Pretrained'),
        interaction_indexes=[
            5,
            8,
            11,
        ],
        name='dinov3_vits16plus',
        return_idx=[
            1,
            2,
            3,
        ],
        type='DINOv3STAs',
        use_lab=False,
        weights_path=
        '/mnt/user/wanglubo/rtdetr-mmdet/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    ),
    bbox_head=dict(
        eval_idx=-1,
        layer_scale=1.0,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=1.0,
            gamma=1.5,
            iou_weighted=True,
            loss_weight=1.0,
            type='DEIMMalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        loss_ld=dict(
            T=5,
            loss_weight=1.5,
            reduction='none',
            type='KnowledgeDistillationKLDivLoss'),
        num_classes=80,
        reg_act_cfg=dict(inplace=True, type='SiLU'),
        reg_max=32,
        reg_scale=4,
        sync_cls_avg_factor=True,
        type='DFINEHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interpolations='nearest',
                interval=1,
                random_sizes=[
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    640,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
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
    decoder=dict(
        eval_idx=-1,
        layer_cfg=dict(
            cross_attn_cfg=dict(
                dropout=0.0,
                embed_dims=256,
                num_levels=3,
                num_points=(
                    3,
                    6,
                    3,
                )),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        layer_scale=1.0,
        lqe_act_cfg=dict(inplace=True, type='SiLU'),
        num_layers=6,
        post_norm_cfg=None,
        ref_act_cfg=dict(inplace=True, type='SiLU'),
        ref_hidden_dim=256,
        ref_num_layers=3,
        reg_max=32,
        reg_scale=4,
        return_intermediate=True,
        update_query_pos=False),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fpn_cfg=dict(
            expansion=1.25,
            fuse_type='sum',
            in_channels=[
                256,
                256,
                256,
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_csp_blocks=4,
            out_channels=256,
            type='DEIMV2FPN'),
        in_channels=[
            256,
            256,
            256,
        ],
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(type='GELU'),
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_encoder_layers=1,
        use_encoder_idx=[
            -1,
        ]),
    eval_idx=-1,
    neck=None,
    num_queries=300,
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner'),
        switch_assigner=dict(
            assigner=dict(
                match_costs=[
                    dict(
                        iou_order_alpha=4.0, type='DEIMV2LossCost',
                        weight=1.0),
                ],
                type='HungarianAssigner'),
            switch_epoch=45)),
    type='DEIMV2',
    with_box_refine=True)
num_blocks_list = (
    1,
    1,
    3,
    1,
)
num_layers = 6
num_points = (
    3,
    6,
    3,
)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.000125),
    paramwise_cfg=dict(
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys=dict({
            'backbone.dinov3':
            dict(lr_mult=0.02),
            'backbone.dinov3.blocks.0.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.0.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.1.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.10.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.11.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.2.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.3.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.4.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.5.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.6.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.7.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.8.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.attn.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.attn.qkv.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.attn.qkv.bias_mask':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.mlp.fc1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.mlp.fc2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.norm1.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.norm1.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.norm2.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.blocks.9.norm2.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.norm.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.norm.weight':
            dict(decay_mult=0, lr_mult=0.02),
            'backbone.dinov3.patch_embed.proj.bias':
            dict(decay_mult=0, lr_mult=0.02),
            'decoder.layers.0.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.0.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.0.norms.2.scale':
            dict(decay_mult=0),
            'decoder.layers.1.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.1.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.1.norms.2.scale':
            dict(decay_mult=0),
            'decoder.layers.2.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.2.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.2.norms.2.scale':
            dict(decay_mult=0),
            'decoder.layers.3.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.3.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.3.norms.2.scale':
            dict(decay_mult=0),
            'decoder.layers.4.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.4.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.4.norms.2.scale':
            dict(decay_mult=0),
            'decoder.layers.5.norms.0.scale':
            dict(decay_mult=0),
            'decoder.layers.5.norms.1.scale':
            dict(decay_mult=0),
            'decoder.layers.5.norms.2.scale':
            dict(decay_mult=0),
            'in_proj_bias':
            dict(decay_mult=0)
        }),
        norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=2000, type='QuadraticWarmupLR'),
    dict(
        begin=29,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min_ratio=0.5,
        type='CosineAnnealingLR'),
    dict(begin=50, by_epoch=True, factor=1, type='ConstantLR'),
]
pretrained = '/mnt/user/wanglubo/rtdetr-mmdet/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
reg_max = 32
reg_scale = 4
resume = False
stage2_num_epochs = 8
stage2_switch_epoch = 4
stage3_switch_epoch = 29
stage4_switch_epoch = 50
switch_assigner_epoch = 45
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
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
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        640,
        640,
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
train_cfg = dict(max_epochs=58, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=4,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train2017.json',
            backend_args=None,
            data_prefix=dict(img='train2017/'),
            data_root='data/coco/',
            filter_cfg=None,
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='CocoDataset'),
        deepcopy=False,
        pipeline=[
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=False, scale=(
                640,
                640,
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
        type='MultiImageMixDataset'),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(keep_ratio=False, scale=(
        640,
        640,
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
                    640,
                    640,
                ), type='Resize'),
            ],
            [
                dict(
                    center_ratio_range=(
                        1.0,
                        1.0,
                    ),
                    img_scale=(
                        320,
                        320,
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
        640,
        640,
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
        640,
        640,
    ), type='Resize'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
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
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
