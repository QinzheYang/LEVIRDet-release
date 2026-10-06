#!/usr/bin/env python3
"""Local-only browser UI around the release's unchanged image_demo entry point."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm'}
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_FILES = 10000
MAX_PIXELS = 100_000_000
HERE = Path(__file__).resolve().parent


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class DemoState:
    def __init__(self, project, output, token, manifest=None, runner=None):
        self.project = Path(project).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.manifest = manifest or json.loads((HERE / 'assets.json').read_text(encoding='utf-8'))
        self.models = {m['id']: m for m in self.manifest['models']}
        self.assets = {a['id']: a for a in self.manifest['assets']}
        self.jobs = {}
        self.latest = None
        self.running = None
        self.lock = threading.RLock()
        self.process = None
        self.runner = runner or self.run_inference
        self.host_output = os.environ.get('LEVIR_DEMO_HOST_OUTPUT', str(self.output))

    def public_job(self, job):
        with self.lock:
            result = {k: copy.deepcopy(v) for k, v in job.items() if not k.startswith('_')}
            for item in result['files']:
                prefix = f"/api/jobs/{job['id']}/files/{item['id']}"
                item['input_url'] = prefix + '/input' if item['uploaded'] else None
                item['visualization_url'] = prefix + '/visualization' if item['status'] == 'done' else None
                item['prediction_url'] = prefix + '/prediction' if item['status'] == 'done' else None
            return result

    def get_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ApiError('This session was not found. Select your images again.', 404)
            return self.jobs[job_id]

    def create_job(self, payload):
        if not isinstance(payload, dict):
            raise ApiError('Expected a JSON object.')
        model = str(payload.get('model', self.manifest['default_model']))
        if model not in self.models:
            raise ApiError('Choose the 30-class or 159-class model.')
        threshold = payload.get('threshold', 0.3)
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ApiError('Confidence threshold must be between 0 and 1.')
        files = payload.get('files')
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
            raise ApiError(f'Select between 1 and {MAX_FILES:,} supported images.')
        checked = []
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                raise ApiError('Invalid image description.')
            name = item.get('name', '')
            relative = item.get('relative_path', name)
            size = item.get('size', 0)
            if (not isinstance(name, str) or not name or len(name) > 255
                    or '/' in name or '\\' in name or any(ord(c) < 32 for c in name)):
                raise ApiError('Invalid image filename.')
            if not isinstance(relative, str) or len(relative) > 4096:
                raise ApiError('Invalid relative filename.')
            suffix = Path(name).suffix.lower()
            if suffix not in SUFFIXES:
                raise ApiError(f'Unsupported image: {name}')
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_FILE_BYTES:
                raise ApiError(f'{name}: each image must be nonempty and at most 512 MiB.')
            checked.append(dict(id=f'{index + 1:06d}', index=index, name=name,
                                relative_path=relative, size=size,
                                stored_name=f'{index + 1:06d}{suffix}',
                                uploaded=False, status='waiting'))
        total_size = sum(f['size'] for f in checked)
        if shutil.disk_usage(self.output).free < total_size * 2 + 512 * 1024 * 1024:
            raise ApiError('Not enough free disk space for the input copies and results.')
        with self.lock:
            if self.running:
                raise ApiError('An inference is already running. Wait for it to finish.', 409)
            job_id = uuid.uuid4().hex
            dirname = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{model}class_' + job_id[:8]
            folder = self.output / dirname
            (folder / 'input').mkdir(parents=True)
            separator = '\\' if '\\' in self.host_output else '/'
            job = dict(id=job_id, model=model, threshold=float(threshold),
                       status='uploading', stage='Preparing images',
                       completed=0, total=len(checked), files=checked, error=None,
                       output_dir=self.host_output.rstrip('/\\') + separator + dirname,
                       created_at=datetime.now().isoformat(timespec='seconds'),
                       _folder=folder, _uploading=set())
            self.jobs[job_id] = job
            self.latest = job_id
            self.persist(job)
            return self.public_job(job)

    def persist(self, job):
        folder = job['_folder']
        temporary = folder / 'manifest.json.tmp'
        temporary.write_text(json.dumps(self.public_job(job), ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(folder / 'manifest.json')

    def file_item(self, job, file_id):
        for item in job['files']:
            if item['id'] == file_id:
                return item
        raise ApiError('Image not found.', 404)

    def upload(self, job_id, file_id, stream, length):
        job = self.get_job(job_id)
        with self.lock:
            item = self.file_item(job, file_id)
            if job['status'] != 'uploading' or file_id in job['_uploading']:
                raise ApiError('This image cannot be uploaded again.', 409)
            if length != item['size']:
                raise ApiError('Image size does not match the selected file.')
            job['_uploading'].add(file_id)
        path = job['_folder'] / 'input' / item['stored_name']
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            remaining = length
            with temporary.open('xb') as destination:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ApiError('Upload interrupted. Please select the images and try again.')
                    destination.write(block)
                    remaining -= len(block)
            from PIL import Image
            with Image.open(temporary) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ApiError(f'{item["name"]}: image exceeds the 100-megapixel demo limit.')
                image.verify()
            temporary.replace(path)
            with self.lock:
                item.update(uploaded=True, status='ready')
                job.update(error=None, stage='Preparing images')
                self.persist(job)
            return dict(uploaded=True, file_id=file_id)
        except Exception as error:
            with self.lock:
                job.update(stage='Upload failed; retry or select new images', error=str(error))
                self.persist(job)
            if isinstance(error, ApiError):
                raise
            raise ApiError(f'Cannot read {item["name"]} as an image: {error}') from error
        finally:
            with self.lock:
                job['_uploading'].discard(file_id)
            if temporary.exists():
                temporary.unlink()

    def start(self, job_id):
        job = self.get_job(job_id)
        with self.lock:
            if self.running or job['status'] != 'uploading':
                raise ApiError('An inference is already running or this session has ended.', 409)
            if not all(item['uploaded'] for item in job['files']):
                raise ApiError('Wait for all images to finish uploading.', 409)
            model = self.models[job['model']]
            for relative in (model['config'], self.assets[model['asset']]['path'], 'demo/image_demo.py'):
                if not (self.project / relative).is_file():
                    raise ApiError(f'Missing project file: {relative}. Close this window and run the launcher again.')
            self.running = job_id
            self.latest = job_id
            job.update(status='running', stage='Loading model')
            self.persist(job)
            threading.Thread(target=self.execute, args=(job,), daemon=True).start()
            return self.public_job(job)

    def execute(self, job):
        try:
            self.runner(job)
            with self.lock:
                job.update(status='done', stage='Inference complete', completed=job['total'])
        except Exception as error:
            with self.lock:
                message = str(error)
                if 'out of memory' in message.lower():
                    message = 'The GPU ran out of memory. Close other GPU applications and try again. Details: ' + message
                job.update(status='failed', stage='Inference failed', error=message[-4000:])
        finally:
            with self.lock:
                self.running = None
                self.process = None
                self.persist(job)

    def refresh_results(self, job):
        folder = job['_folder'] / 'predictions'
        complete = 0
        with self.lock:
            for item in job['files']:
                pred = folder / 'preds' / (item['id'] + '.json')
                vis = folder / 'vis' / item['stored_name']
                if pred.is_file() and vis.is_file():
                    # A parse succeeds only after the prediction JSON has been fully written.
                    try:
                        json.loads(pred.read_text(encoding='utf-8'))
                    except (OSError, ValueError):
                        continue
                    item['status'] = 'done'
                    complete += 1
            job['completed'] = complete
            if complete:
                job['stage'] = f'Processing images: {complete}/{job["total"]}'

    def run_inference(self, job):
        model = self.models[job['model']]
        folder = job['_folder']
        output = folder / 'predictions'
        command = [sys.executable, '-u', str(self.project / 'demo/image_demo.py'),
                   str(folder / 'input'), model['config'], '--weights',
                   self.assets[model['asset']]['path'], '--out-dir', str(output),
                   '--device', 'cuda:0', '--batch-size', '1', '--pred-score-thr', str(job['threshold'])]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1')
        tail = []
        with (folder / 'inference.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=self.project, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1)
            with self.lock:
                self.process = process
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail = (tail + [line.rstrip()])[-25:]
                if line.startswith('{'):
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        payload = {}
                    if 'class_count' in payload and payload['class_count'] != model['num_classes']:
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError('The selected checkpoint has the wrong class count. Run the launcher to verify the weights.')
                if line.startswith('Processed '):
                    self.refresh_results(job)
            if process.wait() != 0:
                detail = next((line for line in reversed(tail) if line.strip()), 'Unknown error')
                raise RuntimeError('Inference stopped. Details are saved in inference.log. ' + detail[-1200:])
        self.refresh_results(job)
        metadata = json.loads((output / 'run_metadata.json').read_text(encoding='utf-8'))
        if (metadata.get('status') != 'passed' or metadata.get('class_count') != model['num_classes']
                or not metadata.get('checkpoint_audit', {}).get('strict_load_passed')
                or job['completed'] != job['total']):
            raise RuntimeError('The inference result did not pass completeness or checkpoint validation. See inference.log.')

    def media(self, job_id, file_id, kind):
        job = self.get_job(job_id)
        item = self.file_item(job, file_id)
        folder = job['_folder']
        if kind == 'input' and item['uploaded']:
            path = folder / 'input' / item['stored_name']
        elif kind == 'visualization' and item['status'] == 'done':
            path = folder / 'predictions' / 'vis' / item['stored_name']
        elif kind == 'prediction' and item['status'] == 'done':
            path = folder / 'predictions' / 'preds' / (item['id'] + '.json')
        else:
            raise ApiError('This result is not ready yet.', 404)
        if kind == 'prediction':
            return path.read_bytes(), 'application/json'
        # Convert only browser previews; original inputs and detector outputs stay on disk.
        from PIL import Image
        with Image.open(path) as image:
            image.thumbnail((2400, 2400))
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='JPEG', quality=92)
            return buffer.getvalue(), 'image/jpeg'


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    @property
    def state(self):
        return self.server.demo_state

    def log_message(self, format, *args):
        # Do not put authentication headers in application logs.
        sys.stderr.write('%s %s\n' % (self.log_date_time_string(), format % args))

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def send_bytes(self, body, mime='application/json', status=200):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

    def authorize(self):
        host = self.headers.get('Host', '')
        try:
            hostname = urlsplit('http://' + host).hostname
        except ValueError:
            hostname = None
        if hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ApiError('Only localhost access is allowed.', 403)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + host:
            raise ApiError('Cross-origin requests are not allowed.', 403)
        if not hmac.compare_digest(self.headers.get('X-Levir-Token', ''), self.state.token):
            raise ApiError('Session key is missing or expired. Reopen the page using the launcher.', 401)

    def read_json(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as error:
            raise ApiError('Invalid Content-Length.') from error
        if not 0 <= length <= 4 * 1024 * 1024:
            raise ApiError('Request is too large.', 413)
        try:
            return json.loads(self.rfile.read(length) or b'{}')
        except ValueError as error:
            raise ApiError('Invalid JSON request.') from error

    def dispatch(self):
        path = urlsplit(self.path).path
        if self.command == 'GET' and path == '/health':
            return self.send_json(dict(ok=True, application='LEVIRDetNet'))
        if path.startswith('/api/'):
            self.authorize()
        elif self.command == 'GET' and path in ('/', '/index.html', '/app.js', '/style.css'):
            name = 'index.html' if path == '/' else path[1:]
            mime = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript; charset=utf-8',
                    'style.css': 'text/css; charset=utf-8'}[name]
            return self.send_bytes((HERE / 'static' / name).read_bytes(), mime)
        else:
            raise ApiError('Not found.', 404)
        if self.command == 'GET' and path == '/api/status':
            state = self.state
            latest = state.jobs.get(state.latest)
            return self.send_json(dict(models=[dict(id=m['id'], label=m['label']) for m in state.models.values()],
                                       default_model=state.manifest['default_model'], output_root=state.host_output,
                                       active_job=state.public_job(latest) if latest else None))
        if self.command == 'POST' and path == '/api/jobs':
            return self.send_json(self.state.create_job(self.read_json()), 201)
        match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})(?:/(start|files)(?:/([0-9]{6})(?:/(input|visualization|prediction))?)?)?', path)
        if not match:
            raise ApiError('Not found.', 404)
        job_id, action, file_id, kind = match.groups()
        if self.command == 'GET' and action is None:
            return self.send_json(self.state.public_job(self.state.get_job(job_id)))
        if self.command == 'POST' and action == 'start':
            self.read_json()
            return self.send_json(self.state.start(job_id), 202)
        if self.command == 'PUT' and action == 'files' and file_id and not kind:
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError as error:
                raise ApiError('Invalid Content-Length.') from error
            if not 0 < length <= MAX_FILE_BYTES:
                raise ApiError('Invalid image size.', 413)
            return self.send_json(self.state.upload(job_id, file_id, self.rfile, length))
        if self.command == 'GET' and action == 'files' and kind:
            body, mime = self.state.media(job_id, file_id, kind)
            return self.send_bytes(body, mime)
        raise ApiError('Method not allowed.', 405)

    def handle_request(self):
        try:
            self.dispatch()
        except ApiError as error:
            self.close_connection = True
            self.send_json(dict(error=str(error)), error.status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as error:
            self.close_connection = True
            self.log_error('%s: %s', type(error).__name__, error)
            self.send_json(dict(error=str(error)), 500)

    do_GET = handle_request
    do_POST = handle_request
    do_PUT = handle_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output-root', type=Path, default=Path('fast_demo_outputs'))
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    args = parser.parse_args()
    token = os.environ.get('LEVIR_DEMO_TOKEN')
    if not token or len(token) < 24:
        parser.error('Start this app with launch_fast_demo.cmd or launch_fast_demo.sh (session key required).')
    state = DemoState(args.project_root, args.output_root, token)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.demo_state = state
    print(f'LEVIRDetNet Fast Demo ready on port {httpd.server_port}. Outputs: {state.host_output}', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if state.process and state.process.poll() is None:
            state.process.terminate()
        httpd.server_close()


if __name__ == '__main__':
    main()
