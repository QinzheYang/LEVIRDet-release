"""Small local protocol tests; no Docker, GPU, or model downloads required."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

spec = importlib.util.spec_from_file_location('fast_demo_server', Path(__file__).with_name('server.py'))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for name in ('config.py', 'weights.pth', 'demo/image_demo.py'):
            target = root / name
            target.parent.mkdir(exist_ok=True)
            target.write_text('test fixture')
        manifest = dict(default_model='30', models=[dict(id='30', label='30 classes', num_classes=30,
                                                       config='config.py', asset='weights30')],
                        assets=[dict(id='weights30', path='weights.pth')])
        self.token = 'test-session-token-' + 'x' * 32
        self.state = server.DemoState(root, root / 'outputs', self.token, manifest, runner=self.fake_runner)
        self.httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        self.httpd.demo_state = self.state
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.httpd.server_port}'
        data = io.BytesIO()
        Image.new('RGB', (16, 16), '#5a887c').save(data, format='PNG')
        self.image = data.getvalue()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, method, path, data=None, token=True, headers=None):
        h = dict(headers or {})
        if token:
            h['X-Levir-Token'] = self.token
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode()
            h['Content-Type'] = 'application/json'
        request = Request(self.url + path, data=data, method=method, headers=h)
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            body = response.read()
            return response.status, json.loads(body) if 'json' in response.headers.get('Content-Type', '') else body

    def create(self, names=('image.png',)):
        return self.request('POST', '/api/jobs', dict(model='30', files=[dict(name=n, relative_path=n,
                                                                           size=len(self.image)) for n in names]))

    def fake_runner(self, job):
        folder = job['_folder'] / 'predictions'
        (folder / 'vis').mkdir(parents=True)
        (folder / 'preds').mkdir()
        for item in job['files']:
            (folder / 'vis' / item['stored_name']).write_bytes(self.image)
            (folder / 'preds' / (item['id'] + '.json')).write_text('{"labels":[],"scores":[],"bboxes":[]}')
        self.state.refresh_results(job)

    def test_local_auth_and_origin(self):
        self.assertEqual(self.request('GET', '/health', token=False)[0], 200)
        self.assertEqual(self.request('GET', '/api/status', token=False)[0], 401)
        self.assertEqual(self.request('GET', '/api/status', headers={'Origin': 'https://example.org'})[0], 403)
        self.assertEqual(self.request('GET', '/api/status', headers={'Host': 'attacker.example'})[0], 403)
        self.assertEqual(self.request('GET', '/api/status')[1]['default_model'], '30')

    def test_invalid_inputs_and_traversal(self):
        for name in ('../secret.png', 'x\\secret.png', 'bad.svg'):
            self.assertEqual(self.create((name,))[0], 400)
        self.assertEqual(self.request('POST', '/api/jobs', dict(model='160', files=[]))[0], 400)
        self.assertEqual(self.request('GET', '/../../assets.json')[0], 404)
        self.assertEqual(self.request('POST', '/api/jobs', [1, 2])[0], 400)

    def test_upload_retry_collision_progress_and_media(self):
        status, job = self.create(('same.png', 'same.png'))
        self.assertEqual(status, 201)
        self.assertNotEqual(job['files'][0]['stored_name'], job['files'][1]['stored_name'])
        base = '/api/jobs/' + job['id']
        self.assertEqual(self.request('POST', base + '/start', {})[0], 409)
        for item in job['files']:
            target = base + '/files/' + item['id']
            self.assertEqual(self.request('PUT', target, self.image[:-1])[0], 400)
            self.assertEqual(self.request('PUT', target, self.image)[0], 200)
            self.assertEqual(self.request('PUT', target, self.image)[0], 200)
        self.assertEqual(self.request('POST', base + '/start', {})[0], 202)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.request('GET', base)[1]
            if result['status'] == 'done':
                break
            time.sleep(.02)
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['completed'], 2)
        for item in result['files']:
            self.assertEqual(self.request('GET', item['prediction_url'])[1]['labels'], [])
            status, preview = self.request('GET', item['visualization_url'])
            self.assertEqual(status, 200)
            self.assertEqual(Image.open(io.BytesIO(preview)).size, (16, 16))
        self.assertTrue((self.state.get_job(job['id'])['_folder'] / 'manifest.json').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
