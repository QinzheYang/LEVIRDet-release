#!/usr/bin/env python3
"""Environment-neutral inference for trusted local LEVIR checkpoints.

Called by demo/image_demo.py in Docker and Conda environments alike.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
from datetime import datetime, timezone
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import time


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm'}



def resolve_config_path(model, project_root):
    """Resolve only a missing bare .py name; preserve MMDet aliases and URLs."""
    if model is None or not model.lower().endswith('.py'):
        return model
    path = Path(model).expanduser()
    if path.is_file():
        return str(path)
    if path.name != model or '/' in model or '\\' in model:
        return model
    matches = sorted((Path(project_root) / 'configs').rglob(model))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f'Configuration {model!r} was not found under configs/.')
    raise ValueError('Ambiguous configuration basename; use a full relative path: '
                     + ', '.join(str(path) for path in matches))


def load_levir_config(model):
    """Leave unrelated architectures and checkpoint-only MMDet calls unchanged."""
    if model is None or not str(model).lower().endswith('.py') or not Path(model).is_file():
        return None
    from mmengine.config import Config
    cfg = Config.fromfile(str(model))
    if str(cfg.get('model', {}).get('type', '')).rsplit('.', 1)[-1] == 'DEIMV2GSDGuided':
        return cfg
    return None


def validate_levir_options(init_args, call_args):
    defaults = dict(texts=None, custom_entities=False, chunked_size=-1, tokens_positive=None)
    unsupported = [key for key, default in defaults.items() if call_args.get(key, default) != default]
    if unsupported:
        raise ValueError('LEVIR does not support these text/prompt options: ' +
                         ', '.join('--' + key.replace('_', '-') for key in unsupported))
    if not init_args.get('weights'):
        raise ValueError('LEVIR inference requires --weights with a complete detector checkpoint.')
    if call_args.get('batch_size', 1) < 1 or not 0 <= call_args.get('pred_score_thr', 0.3) <= 1:
        raise ValueError('batch-size must be positive; pred-score-thr must be between 0 and 1')


@contextlib.contextmanager
def inactive_query_guard(model, torch, image_paths, report):
    """Repair only the diagnosed inactive reference Inf * 0 pattern."""
    name = '_apply_query_grouping'
    original = getattr(model, name, None)
    if original is None:
        yield
        return
    had_instance_value = name in model.__dict__
    instance_value = model.__dict__.get(name)
    corrections = report.setdefault('query_padding_corrections', {})

    def checked(*positional, **keywords):
        decoder = keywords['decoder_inputs_dict'] if 'decoder_inputs_dict' in keywords else positional[0]
        before = decoder['reference_points']
        result = original(*positional, **keywords)
        after = decoder['reference_points']
        bad = torch.isnan(after)
        if bool(bad.any()):
            q_count = model._last_group_queries.to(after.device)
            inactive = torch.arange(after.shape[1], device=after.device)[None, :] >= q_count[:, None]
            expected = torch.isinf(before[:, :after.shape[1]]) & inactive[:, :, None]
            if not torch.equal(bad, expected):
                raise RuntimeError('Unexpected reference NaN outside the diagnosed inactive inf*0 pattern.')
            counts = bad.flatten(1).sum(1).tolist()
            if len(counts) != len(image_paths):
                raise RuntimeError('Query guard image-path batch alignment failed.')
            for path, count in zip(image_paths, counts):
                if count:
                    key = str(path)
                    corrections[key] = corrections.get(key, 0) + int(count)
            # Preserve valid +Inf refs: the decoder sigmoid maps them to one.
            decoder['reference_points'] = after.masked_fill(bad, 0)
        return result

    setattr(model, name, checked)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(model, name, instance_value)
        else:
            delattr(model, name)


def saved_state_semantics(checkpoint, cfg):
    """Describe the audited save-hook convention without exchanging tensors."""
    from mmengine.hooks.ema_hook import EMAHook
    from mmdet.engine.hooks.ema_dynamic_momentum_hook import EMADynamicMomentumHook

    hooks = [dict(hook) for hook in cfg.get('custom_hooks', [])
             if str(hook.get('type', '')).rsplit('.', 1)[-1]
             in ('EMAHook', 'EMADynamicMomentumHook')]
    snapshot = checkpoint.get('meta', {}).get('cfg')
    source = checkpoint.get('state_dict', {})
    ema = checkpoint.get('ema_state_dict', {})
    normalized = {key.removeprefix('module.'): value for key, value in source.items()}
    wrapped = {key[7:]: value for key, value in ema.items() if key.startswith('module.')}
    matching = bool(normalized) and set(normalized) == set(wrapped) and all(
        getattr(value, 'shape', None) == getattr(wrapped[key], 'shape', None)
        for key, value in normalized.items())
    save_source = inspect.getsource(EMAHook.before_save_checkpoint)
    swap_source = inspect.getsource(EMAHook._swap_ema_state_dict)
    dynamic_source = inspect.getsource(EMADynamicMomentumHook.before_save_checkpoint)
    convention = (
        "checkpoint['ema_state_dict'] = self.ema_model.state_dict()" in save_source
        and 'self._swap_ema_state_dict(checkpoint)' in save_source
        and "ema_state[k] = model_state[k[7:]]" in swap_source
        and "model_state[k[7:]] = tmp" in swap_source
        and 'super().before_save_checkpoint(runner, checkpoint)' in dynamic_source)
    verified = bool(isinstance(snapshot, str) and snapshot.strip() and hooks and matching and convention)
    return {
        'ema_used': True if verified else None,
        'extra_ema_swap': False,
        'saved_state_dict_semantics': (
            'EMA deployment parameters written by the checkpoint save hook' if verified
            else 'unverified; state_dict loaded unchanged'),
        'ema_semantics_validation': {
            'verified': verified,
            'saved_configuration_has_ema_hook': bool(hooks),
            'saved_ema_hooks': hooks,
            'ema_wrapper_and_main_tensor_keys_shapes_match': matching,
            'installed_save_hook_exchange_convention_matches_reviewed_source': convention,
            'interpretation': 'The save hook places EMA in state_dict and original model weights in ema_state_dict.module.*.',
        },
    }


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def list_images(path):
    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name.casefold())
    else:
        raise FileNotFoundError(path)
    if not images:
        raise ValueError('No supported images found (directory input is non-recursive).')
    stems = {}
    for image in images:
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image extension: {image}')
        key = image.stem.casefold()
        if key in stems:
            raise ValueError(f'Prediction JSON names collide: {stems[key]} and {image.name}')
        stems[key] = image.name
    return images


@contextlib.contextmanager
def singleton_prior(model, torch, policy, report):
    name = '_extract_density_scale_prior'
    original = getattr(model, name, None)
    if original is None:
        yield
        return
    had_instance_value = name in model.__dict__
    instance_value = model.__dict__.get(name)

    def checked(memory, spatial_shapes):
        if not torch.isfinite(memory).all().item():
            raise RuntimeError('Encoder memory has NaN/Inf.')
        if memory.shape[0] == 1:
            if policy == 'error':
                raise RuntimeError('This model has undefined sample std at batch=1. Use --single-image-prior zero for a neutral centered prior.')
            result = memory.new_zeros((1, 2))
            report['singleton_prior_zero_calls'] += 1
        else:
            result = original(memory, spatial_shapes)
        if not torch.isfinite(result).all().item():
            raise RuntimeError('Density/scale prior has NaN/Inf.')
        report['prior_calls_checked'] += 1
        return result

    setattr(model, name, checked)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(model, name, instance_value)
        else:
            delattr(model, name)


def run_levir_demo(init_args, call_args, requested):
    """Use MMDet's normal CLI flags with the checkpoint's complete saved model."""
    validate_levir_options(init_args, call_args)
    args = SimpleNamespace(**dict(call_args, **init_args))
    args.config = Path(args.model)
    args.inputs = Path(args.inputs)
    args.weights = Path(args.weights)
    args.single_image_prior = 'zero'
    images = list_images(args.inputs.resolve())
    for path in (args.config, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.out_dir).resolve() if args.out_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    # Avoid silently replacing previous predictions or run metadata.
    planned = [output / 'run_metadata.json', output / 'effective_config.py'] if output else []
    for image in images:
        if output and not args.no_save_vis:
            planned.append(output / 'vis' / image.name)
        if output and not args.no_save_pred:
            planned.append(output / 'preds' / (image.stem + '.json'))
    occupied = [str(path) for path in planned if path.exists()]
    if occupied:
        raise FileExistsError('Use a fresh output directory; existing outputs: ' + ', '.join(occupied[:5]))

    report = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'requested_config': str(args.config), 'checkpoint': str(args.weights),
        'checkpoint_bytes': args.weights.stat().st_size,
        'input': str(args.inputs), 'input_count': len(images), 'processed_images': 0,
        'batch_size': args.batch_size, 'resize': [1024, 1024],
        'single_image_prior': args.single_image_prior,
        'singleton_prior_zero_calls': 0, 'prior_calls_checked': 0,
        'model_source_modified': False, 'weights_source': 'state_dict', 'ema_used': None,
        'pred_score_thr_visualization_only': args.pred_score_thr,
        'prediction_labels': 'zero-based indices into classes',
        'prediction_boxes': 'xyxy in original-image pixel coordinates',
        'no_save_vis': args.no_save_vis, 'no_save_pred': args.no_save_pred,
        'metrics_computed': False,
    }
    start = time.monotonic()
    try:
        import torch
        from mmengine.config import Config
        from mmdet.apis import DetInferencer
        from mmdet.utils import register_all_modules

        if str(args.device).split(':', 1)[0] != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('LEVIR inference requires an available CUDA device (for example --device cuda:0).')
        torch.cuda.set_device(torch.device(args.device))
        register_all_modules(init_default_scope=True)
        checkpoint = torch.load(str(args.weights), map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError('Expected a checkpoint dictionary.')
        state = checkpoint.get('state_dict', checkpoint)
        meta = checkpoint.get('meta', {})
        dataset_meta = {str(k).lower(): v for k, v in meta.get('dataset_meta', {}).items()}
        saved_classes = dataset_meta.get('classes', meta.get('CLASSES'))
        snapshot = meta.get('cfg')
        if isinstance(snapshot, str) and snapshot.strip():
            cfg = Config.fromstring(snapshot, '.py')
            report['effective_config_source'] = 'checkpoint.meta.cfg'
            for key in ('type',):
                if requested.model[key] != cfg.model[key]:
                    raise ValueError('Requested configuration and checkpoint have different detector architectures.')
            if requested.model.backbone.get('name') != cfg.model.backbone.get('name'):
                raise ValueError('Requested configuration and checkpoint have different backbone names.')
        else:
            cfg = requested.copy()
            report['effective_config_source'] = 'requested_config (checkpoint has no saved configuration)'
        classes = tuple(cfg.get('classes', cfg.get('metainfo', {}).get('classes', ())))
        if not classes or len(classes) != cfg.model.bbox_head.num_classes:
            raise ValueError('Effective configuration has inconsistent class metadata and detection head.')
        if saved_classes is not None and tuple(saved_classes) != classes:
            raise ValueError('Checkpoint class names/order differ from effective configuration.')
        report.update(
            requested_class_count=len(requested.get('classes', ())),
            requested_head=requested.model.bbox_head.type,
            class_count=len(classes), classes=classes,
            model=cfg.model.type, backbone=cfg.model.backbone.name,
            head=cfg.model.bbox_head.type,
            checkpoint_epoch=meta.get('epoch'),
            checkpoint_has_ema='ema_state_dict' in checkpoint,
            gpu=torch.cuda.get_device_name(torch.device(args.device)),
        )
        report.update(saved_state_semantics(checkpoint, cfg))
        # The complete detector checkpoint includes these parameters. Do not
        # read obsolete initialization paths or download separate pretraining.
        cfg.model.backbone.weights_path = None
        cfg.model.gsd_cfg.ckpt = ''
        cfg.model.init_cfg = None
        cfg.model.backbone.init_cfg = None
        cfg.load_from = None
        cfg.resume = False
        cfg.work_dir = str(output) if output else '.'
        cfg.visualizer = dict(type='DetLocalVisualizer', name='levir_image_demo', vis_backends=[])
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
            dict(type='PackDetInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')),
        ]
        cfg.test_dataloader.dataset.pipeline = pipeline
        cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
        report['external_initialization_checkpoints_used'] = False

        # Drop optimizer/EMA payloads before constructing the inference model.
        del checkpoint
        gc.collect()

        class StrictInferencer(DetInferencer):
            def _load_weights_to_model(self, model, checkpoint, cfg):
                normalized = state
                if any(key.startswith('module.') for key in state):
                    normalized = OrderedDict((key.removeprefix('module.'), value) for key, value in state.items())
                    if len(normalized) != len(state):
                        raise ValueError('Checkpoint keys collide after removing module prefix.')
                target = model.state_dict()
                missing = sorted(set(target) - set(normalized))
                extra = sorted(set(normalized) - set(target))
                conflicts = [key for key in set(target) & set(normalized)
                             if not isinstance(normalized[key], torch.Tensor)
                             or tuple(target[key].shape) != tuple(normalized[key].shape)]
                report['checkpoint_audit'] = dict(model_keys=len(target), checkpoint_keys=len(normalized),
                                                  missing_keys=missing, unexpected_keys=extra,
                                                  shape_conflicts=sorted(conflicts))
                if missing or extra or conflicts:
                    raise ValueError('Checkpoint does not strictly match the effective model; see run_metadata.json.')
                model.load_state_dict(normalized, strict=True)
                model.dataset_meta = dict(dataset_meta, classes=classes)
                configured_palette = cfg.get('metainfo', {}).get('palette')
                model.dataset_meta['palette'] = (args.palette if args.palette != 'none' else
                                                  configured_palette or dataset_meta.get('palette', 'random'))
                report['checkpoint_audit']['strict_load_passed'] = True

            def forward(self, inputs, **kwargs):
                predictions = super().forward(inputs, **kwargs)
                for prediction in predictions:
                    pred = prediction.pred_instances
                    if not torch.isfinite(pred.bboxes).all().item() or not torch.isfinite(pred.scores).all().item():
                        raise RuntimeError('Predicted boxes or scores contain NaN/Inf.')
                    if len(pred.labels) and ((pred.labels < 0).any().item() or (pred.labels >= len(classes)).any().item()):
                        raise RuntimeError('Predicted label outside the checkpoint class list.')
                return predictions

        inferencer = StrictInferencer(model=cfg.to_dict(), weights=None,
                                      device=args.device, palette=args.palette, show_progress=False)
        del state
        gc.collect()
        if output:
            cfg.dump(str(output / 'effective_config.py'))
        print(json.dumps({key: report[key] for key in ('effective_config_source', 'requested_class_count', 'class_count', 'head', 'single_image_prior')}, ensure_ascii=False), flush=True)
        torch.cuda.synchronize()
        report['model_load_seconds'] = round(time.monotonic() - start, 3)
        inference_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        current_paths = []
        with singleton_prior(inferencer.model, torch, args.single_image_prior, report), \
                inactive_query_guard(inferencer.model, torch, current_paths, report):
            for offset in range(0, len(images), args.batch_size):
                paths = images[offset:offset + args.batch_size]
                current_paths[:] = paths
                result = inferencer([str(path) for path in paths], batch_size=args.batch_size,
                                    out_dir=str(output) if output else '', pred_score_thr=args.pred_score_thr,
                                    no_save_vis=args.no_save_vis, no_save_pred=args.no_save_pred,
                                    return_vis=False, return_datasamples=False, show=args.show,
                                    print_result=args.print_result)
                # This local Inferencer otherwise retains all directory images.
                del result
                report['processed_images'] += len(paths)
                print(f"Processed {report['processed_images']}/{len(images)}", flush=True)
        torch.cuda.synchronize()
        report.update(status='passed', inference_seconds=round(time.monotonic() - inference_start, 3),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        return 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['total_seconds'] = round(time.monotonic() - start, 3)
        if output:
            dump_json(output / 'run_metadata.json', report)
        print(json.dumps({'status': report['status'], 'processed_images': report['processed_images'],
                          'output': str(output)}, ensure_ascii=False), flush=True)

