"""Run with unittest; runtime guard cases require torch, but never CUDA."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'levir_demo_under_test', ROOT / 'mmdet/apis/levir_image_demo.py')
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)

try:
    import torch
except ImportError:
    torch = None


class RoutingTests(unittest.TestCase):
    def test_unique_basename_and_ambiguous_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'configs/deimv2/model.py'
            first.parent.mkdir(parents=True)
            first.write_text('model = dict(type="DEIMV2GSDGuided")')
            self.assertEqual(demo.resolve_config_path('model.py', root), str(first))
            second = root / 'configs/other/model.py'
            second.parent.mkdir()
            second.write_text('')
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                demo.resolve_config_path('model.py', root)
            with self.assertRaises(FileNotFoundError):
                demo.resolve_config_path('missing.py', root)

    def test_alias_url_checkpoint_and_explicit_path_are_preserved(self):
        for value in ('rtmdet-s', None, 'weights.pth', 'https://example.org/model.py',
                      'configs/missing.py', r'configs\missing.py'):
            self.assertEqual(demo.resolve_config_path(value, '.'), value)
        self.assertIsNone(demo.load_levir_config(None))
        self.assertIsNone(demo.load_levir_config('rtmdet-s'))

    def test_route_only_local_levir_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.py'
            path.touch()
            fake_module = types.ModuleType('mmengine.config')
            cfg = {'model': {'type': 'DEIMV2GSDGuided'}}
            fake_module.Config = types.SimpleNamespace(fromfile=lambda _: cfg)
            with patch.dict('sys.modules', {'mmengine.config': fake_module}):
                self.assertIs(demo.load_levir_config(str(path)), cfg)
                cfg['model']['type'] = 'mmdet.DEIMV2GSDGuided'
                self.assertIs(demo.load_levir_config(str(path)), cfg)
                cfg['model']['type'] = 'RTMDet'
                self.assertIsNone(demo.load_levir_config(str(path)))

    def test_prompt_options_are_rejected_and_standard_options_accepted(self):
        init = {'weights': 'epoch_117.pth'}
        call = dict(batch_size=1, pred_score_thr=0.3, show=True, print_result=True,
                    no_save_pred=True, no_save_vis=True)
        demo.validate_levir_options(init, call)
        for key, value in (('texts', 'plane'), ('custom_entities', True),
                           ('chunked_size', 2), ('tokens_positive', -1)):
            with self.assertRaisesRegex(ValueError, key.replace('_', '-')):
                demo.validate_levir_options(init, dict(call, **{key: value}))
        with self.assertRaisesRegex(ValueError, '--weights'):
            demo.validate_levir_options({}, call)


@unittest.skipIf(torch is None, 'torch is unavailable')
class GuardTests(unittest.TestCase):
    def make_model(self, counts):
        class Model:
            def __init__(self):
                self._last_group_queries = torch.tensor(counts)

            def _apply_query_grouping(self, decoder_inputs_dict):
                refs = decoder_inputs_dict['reference_points']
                active = torch.arange(refs.shape[1])[None, :] < self._last_group_queries[:, None]
                decoder_inputs_dict['reference_points'] = refs * active[:, :, None]
                return 'original_return'

            def _extract_density_scale_prior(self, memory, spatial_shapes):
                return memory[:, :2] + 3
        return Model()

    def test_exact_inactive_infinity_repaired_valid_infinity_preserved(self):
        model = self.make_model([1, 2])
        refs = torch.tensor([[[float('inf'), 1.], [float('inf'), 2.]],
                             [[1., 2.], [float('inf'), 3.]]])
        decoder = {'reference_points': refs}
        report = {}
        with demo.inactive_query_guard(model, torch, ['a.png', 'b.png'], report):
            self.assertEqual(model._apply_query_grouping(decoder), 'original_return')
        expected = torch.tensor([[[float('inf'), 1.], [0., 0.]],
                                 [[1., 2.], [float('inf'), 3.]]])
        self.assertTrue(torch.equal(decoder['reference_points'], expected))
        self.assertEqual(report['query_padding_corrections'], {'a.png': 1})
        self.assertNotIn('_apply_query_grouping', model.__dict__)

    def test_finite_control_is_exact(self):
        model = self.make_model([1])
        refs = torch.tensor([[[1., 2.], [3., 4.]]])
        expected = {'reference_points': refs.clone()}
        model._apply_query_grouping(expected)
        decoder = {'reference_points': refs.clone()}
        report = {}
        with demo.inactive_query_guard(model, torch, ['a'], report):
            model._apply_query_grouping(decoder_inputs_dict=decoder)
        self.assertTrue(torch.equal(decoder['reference_points'], expected['reference_points']))
        self.assertEqual(report['query_padding_corrections'], {})

    def test_unexpected_nan_rejected_and_instance_method_restored(self):
        model = self.make_model([1])
        original = model._apply_query_grouping
        model._apply_query_grouping = original
        decoder = {'reference_points': torch.tensor([[[float('nan'), 1.], [2., 3.]]])}
        with self.assertRaisesRegex(RuntimeError, 'Unexpected reference NaN'):
            with demo.inactive_query_guard(model, torch, ['a'], {}):
                model._apply_query_grouping(decoder)
        self.assertIs(model.__dict__['_apply_query_grouping'], original)

    def test_query_batch_alignment_failure_does_not_leak_wrapper(self):
        model = self.make_model([1])
        decoder = {'reference_points': torch.tensor([[[1., 1.], [float('inf'), 3.]]])}
        with self.assertRaisesRegex(RuntimeError, 'alignment'):
            with demo.inactive_query_guard(model, torch, [], {}):
                model._apply_query_grouping(decoder)
        self.assertNotIn('_apply_query_grouping', model.__dict__)

    def test_singleton_prior_zero_and_multi_image_prior_unchanged(self):
        model = self.make_model([1])
        report = dict(singleton_prior_zero_calls=0, prior_calls_checked=0)
        two = torch.tensor([[1., 2.], [3., 4.]])
        expected = model._extract_density_scale_prior(two, None)
        with demo.singleton_prior(model, torch, 'zero', report):
            self.assertTrue(torch.equal(model._extract_density_scale_prior(two[:1], None),
                                        torch.zeros((1, 2))))
            self.assertTrue(torch.equal(model._extract_density_scale_prior(two, None), expected))
        self.assertEqual(report, dict(singleton_prior_zero_calls=1, prior_calls_checked=2))
        self.assertNotIn('_extract_density_scale_prior', model.__dict__)

    def test_nonfinite_memory_is_not_silently_zeroed(self):
        model = self.make_model([1])
        report = dict(singleton_prior_zero_calls=0, prior_calls_checked=0)
        with self.assertRaisesRegex(RuntimeError, 'Encoder memory'):
            with demo.singleton_prior(model, torch, 'zero', report):
                model._extract_density_scale_prior(torch.tensor([[float('nan'), 1.]]), None)
        self.assertNotIn('_extract_density_scale_prior', model.__dict__)


if __name__ == '__main__':
    unittest.main()
