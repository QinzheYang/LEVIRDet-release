import argparse
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import queue
import struct
import time
import zlib
from collections import Counter
from pathlib import Path


SPLIT_PREFIXES = ('train', 'val', 'test')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit every image referenced by COCO JSON using the same '
            'MMCV/OpenCV decoder used by LoadImageFromFile.'))
    parser.add_argument('coco_root', type=Path)
    parser.add_argument(
        '--ann-files',
        nargs='*',
        type=Path,
        help='Annotation JSON files. Defaults to annotations/*.json.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Maximum seconds for one image before its worker is killed.')
    parser.add_argument('--progress-interval', type=int, default=250)
    parser.add_argument('--slow-top', type=int, default=100)
    parser.add_argument(
        '--report',
        type=Path,
        help='Output JSON report. Defaults to annotations/image_audit.json.')
    parser.add_argument(
        '--manifest',
        type=Path,
        help='Output SHA-256 manifest. Defaults to annotations/images.sha256.')
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def infer_split(annotation_path):
    stem = annotation_path.stem.lower()
    for split in SPLIT_PREFIXES:
        if stem == split or stem.startswith(f'{split}_'):
            return split
    return None


def collect_records(root, annotation_paths):
    records_by_path = {}
    annotation_stats = []
    for annotation_path in annotation_paths:
        try:
            with annotation_path.open('r', encoding='utf-8-sig') as file:
                coco = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            annotation_stats.append({
                'path': str(annotation_path),
                'error': repr(error),
            })
            continue

        images = coco.get('images')
        if not isinstance(images, list):
            continue
        split = infer_split(annotation_path)
        if split is None:
            raise ValueError(
                f'Cannot infer train/val/test split from {annotation_path}')
        image_root = root / split / 'imgs'
        annotation_stats.append({
            'path': str(annotation_path),
            'split': split,
            'images': len(images),
        })
        for image in images:
            relative_path = Path(split) / 'imgs' / image['file_name']
            absolute_path = (root / relative_path).resolve()
            width = int(image['width'])
            height = int(image['height'])
            old = records_by_path.get(absolute_path)
            if old is not None and old[2:] != (width, height):
                raise ValueError(
                    f'Conflicting declared dimensions for {absolute_path}')
            records_by_path[absolute_path] = (
                str(absolute_path), relative_path.as_posix(), width, height)
    return list(records_by_path.values()), annotation_stats


def check_png_chunks(content):
    signature = b'\x89PNG\r\n\x1a\n'
    if not content.startswith(signature):
        raise ValueError('invalid PNG signature')
    offset = len(signature)
    chunks = []
    saw_ihdr = False
    saw_iend = False
    while offset < len(content):
        if offset + 12 > len(content):
            raise ValueError(f'truncated PNG chunk header at byte {offset}')
        length = struct.unpack('>I', content[offset:offset + 4])[0]
        chunk_type = content[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(content):
            raise ValueError(
                f'truncated {chunk_type!r} chunk at byte {offset}')
        chunk_data = content[offset + 8:offset + 8 + length]
        stored_crc = struct.unpack('>I', content[offset + 8 + length:end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xffffffff
        if actual_crc != stored_crc:
            name = chunk_type.decode('ascii', errors='replace')
            raise ValueError(
                f'PNG CRC mismatch in {name} chunk at byte {offset}')
        chunks.append(chunk_type.decode('ascii', errors='replace'))
        if chunk_type == b'IHDR':
            if saw_ihdr or offset != 8 or length != 13:
                raise ValueError('invalid PNG IHDR placement or length')
            saw_ihdr = True
        if chunk_type == b'IEND':
            if length != 0:
                raise ValueError('invalid PNG IEND length')
            saw_iend = True
            offset = end
            break
        offset = end
    if not saw_ihdr or not saw_iend:
        raise ValueError('PNG is missing IHDR or IEND')
    if offset != len(content):
        raise ValueError(f'PNG has {len(content) - offset} trailing bytes')
    return ','.join(chunks)


def check_container(content, suffix):
    if suffix == '.png':
        return check_png_chunks(content)
    if suffix in ('.jpg', '.jpeg'):
        if not content.startswith(b'\xff\xd8'):
            raise ValueError('invalid JPEG SOI marker')
        if content.rstrip().endswith(b'\xff\xd9') is False:
            raise ValueError('JPEG is missing its final EOI marker')
        return 'SOI,EOI'
    raise ValueError(f'unsupported image suffix {suffix!r}')


def audit_one(record):
    import mmcv
    import numpy as np

    absolute, relative, expected_width, expected_height = record
    path = Path(absolute)
    started = time.perf_counter()
    content = path.read_bytes()
    read_finished = time.perf_counter()
    suffix = path.suffix.lower()
    container = check_container(content, suffix)
    verify_finished = time.perf_counter()
    image = mmcv.imfrombytes(content, flag='color', backend='cv2')
    decode_finished = time.perf_counter()
    if image is None:
        raise ValueError('mmcv.imfrombytes returned None')
    expected_shape = (expected_height, expected_width, 3)
    if image.shape != expected_shape:
        raise ValueError(
            f'decoded shape={image.shape}, expected={expected_shape}')
    if image.dtype != np.uint8:
        raise ValueError(f'decoded dtype={image.dtype}, expected=uint8')
    if not image.flags.c_contiguous:
        raise ValueError('decoded image is not C-contiguous')
    pixel_min = int(image.min())
    pixel_max = int(image.max())
    pixel_mean = float(image.mean())
    finished = time.perf_counter()
    return {
        'path': relative,
        'suffix': suffix,
        'size_bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest(),
        'container': container,
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'pixel_min': pixel_min,
        'pixel_max': pixel_max,
        'pixel_mean': round(pixel_mean, 6),
        'read_ms': round((read_finished - started) * 1000, 3),
        'verify_ms': round((verify_finished - read_finished) * 1000, 3),
        'decode_ms': round((decode_finished - verify_finished) * 1000, 3),
        'total_ms': round((finished - started) * 1000, 3),
    }


def worker_main(worker_id, task_queue, result_queue):
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    while True:
        item = task_queue.get()
        if item is None:
            return
        index, record = item
        result_queue.put((
            'start', worker_id, index, {
                'path': record[1],
                'started': time.time(),
            }))
        try:
            result = audit_one(record)
            result_queue.put(('done', worker_id, index, result))
        except BaseException as error:
            result_queue.put((
                'failed', worker_id, index, {
                    'path': record[1],
                    'error_type': type(error).__name__,
                    'error': str(error),
                }))


def launch_worker(context, worker_id, task_queue, result_queue):
    process = context.Process(
        target=worker_main,
        args=(worker_id, task_queue, result_queue),
        daemon=True)
    process.start()
    return process


def run_audit(records, workers, timeout, progress_interval, slow_top):
    context = mp.get_context('spawn')
    task_queue = context.Queue()
    result_queue = context.Queue()
    for index, record in enumerate(records):
        task_queue.put((index, record))
    for _ in range(workers):
        task_queue.put(None)

    processes = {
        worker_id: launch_worker(
            context, worker_id, task_queue, result_queue)
        for worker_id in range(workers)
    }
    active = {}
    started_indices = set()
    finished_indices = set()
    failures = []
    manifest = {}
    suffix_counts = Counter()
    shape_counts = Counter()
    pixel_range_counts = Counter()
    timings = Counter()
    slowest = []
    start_time = time.time()
    last_reported = 0

    def finish_failure(index, path, error_type, error):
        if index in finished_indices:
            return
        finished_indices.add(index)
        failures.append({
            'path': path,
            'error_type': error_type,
            'error': error,
        })

    while len(finished_indices) < len(records):
        try:
            message = result_queue.get(timeout=0.25)
        except queue.Empty:
            message = None
        if message is not None:
            kind, worker_id, index, payload = message
            if kind == 'start':
                started_indices.add(index)
                active[worker_id] = {
                    'index': index,
                    'path': payload['path'],
                    'started': payload['started'],
                }
            elif kind == 'done':
                active.pop(worker_id, None)
                if index not in finished_indices:
                    finished_indices.add(index)
                    manifest[payload['path']] = payload['sha256']
                    suffix_counts[payload['suffix']] += 1
                    shape_counts[str(payload['shape'])] += 1
                    pixel_range_counts[
                        f'{payload["pixel_min"]}:{payload["pixel_max"]}'] += 1
                    for key in ('read_ms', 'verify_ms', 'decode_ms',
                                'total_ms'):
                        timings[key] += payload[key]
                    entry = (payload['total_ms'], payload['path'], payload)
                    if len(slowest) < slow_top:
                        heapq.heappush(slowest, entry)
                    elif entry[0] > slowest[0][0]:
                        heapq.heapreplace(slowest, entry)
            elif kind == 'failed':
                active.pop(worker_id, None)
                finish_failure(
                    index, payload['path'], payload['error_type'],
                    payload['error'])

        now = time.time()
        for worker_id, state in list(active.items()):
            if now - state['started'] <= timeout:
                continue
            process = processes[worker_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            finish_failure(
                state['index'], state['path'], 'TimeoutError',
                f'image audit exceeded {timeout} seconds')
            active.pop(worker_id, None)
            if len(started_indices) < len(records):
                processes[worker_id] = launch_worker(
                    context, worker_id, task_queue, result_queue)

        for worker_id, process in list(processes.items()):
            if process.is_alive() or worker_id in active:
                continue
            if process.exitcode not in (0, None):
                if len(started_indices) < len(records):
                    processes[worker_id] = launch_worker(
                        context, worker_id, task_queue, result_queue)

        completed = len(finished_indices)
        if (completed and progress_interval > 0
                and completed // progress_interval >
                last_reported // progress_interval):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f'Audited {completed}/{len(records)} images; '
                f'failed={len(failures)}; rate={completed / elapsed:.2f}/s',
                flush=True)
            last_reported = completed

        if not any(process.is_alive() for process in processes.values()):
            pending = [
                index for index in range(len(records))
                if index not in finished_indices
            ]
            for index in pending:
                finish_failure(
                    index, records[index][1], 'WorkerPoolError',
                    'all audit workers exited before processing this image')

    for process in processes.values():
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)

    succeeded = len(records) - len(failures)
    timing_averages = {
        key: round(value / succeeded, 3) if succeeded else None
        for key, value in timings.items()
    }
    return {
        'total_images': len(records),
        'succeeded': succeeded,
        'failed': len(failures),
        'elapsed_seconds': round(time.time() - start_time, 3),
        'suffix_counts': dict(suffix_counts),
        'shape_counts': dict(shape_counts),
        'pixel_range_counts': dict(pixel_range_counts),
        'average_timings_ms': timing_averages,
        'failures': failures,
        'slowest': [
            entry[2] for entry in sorted(slowest, reverse=True)
        ],
    }, manifest


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least one')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    root = args.coco_root.resolve()
    annotation_paths = args.ann_files
    if not annotation_paths:
        annotation_paths = sorted((root / 'annotations').glob('*.json'))
    else:
        annotation_paths = [
            path if path.is_absolute() else root / path
            for path in annotation_paths
        ]
    records, annotation_stats = collect_records(root, annotation_paths)
    records.sort(key=lambda item: item[1])
    if args.limit:
        records = records[:args.limit]
    print(
        f'Collected {len(records)} unique images from '
        f'{len(annotation_stats)} COCO files.',
        flush=True)

    report, manifest = run_audit(
        records, args.workers, args.timeout, args.progress_interval,
        args.slow_top)
    report.update({
        'coco_root': str(root),
        'annotation_files': annotation_stats,
        'decoder': "mmcv.imfrombytes(flag='color', backend='cv2')",
        'workers': args.workers,
        'per_image_timeout_seconds': args.timeout,
    })

    report_path = args.report or root / 'annotations' / 'image_audit.json'
    manifest_path = args.manifest or root / 'annotations' / 'images.sha256'
    if not report_path.is_absolute():
        report_path = root / report_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_text = ''.join(
        f'{digest}  {path}\n' for path, digest in sorted(manifest.items()))
    manifest_bytes = manifest_text.encode('utf-8')
    report['manifest_path'] = str(manifest_path)
    report['manifest_entries'] = len(manifest)
    report['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8'))
    atomic_write(manifest_path, manifest_bytes)

    print(f'Report: {report_path}')
    print(f'Manifest: {manifest_path}')
    print(
        f'Result: total={report["total_images"]}, '
        f'succeeded={report["succeeded"]}, failed={report["failed"]}')
    if report['failures']:
        for failure in report['failures'][:20]:
            print(
                f'FAILED {failure["path"]}: '
                f'{failure["error_type"]}: {failure["error"]}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
