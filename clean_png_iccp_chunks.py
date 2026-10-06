#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from tqdm import tqdm


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def parse_args():
    ap = argparse.ArgumentParser("Clean PNG chunks (remove iCCP and optionally sRGB/gAMA/cHRM/etc.)")
    ap.add_argument("--src_dir", type=str, required=True, help="Folder containing png files")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder (default: <src_dir>__clean)")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan subfolders")
    ap.add_argument("--inplace", action="store_true", help="Clean in-place (overwrite original)")
    ap.add_argument("--backup", action="store_true", help="When --inplace, keep a .bak backup")
    ap.add_argument("--remove", nargs="*", default=["iCCP"],
                    help="Chunk types to remove (default: iCCP). Example: --remove iCCP sRGB gAMA cHRM")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output file exists")
    return ap.parse_args()


def iter_pngs(folder, recursive=False):
    folder = os.path.abspath(folder)
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".png"):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and fn.lower().endswith(".png"):
                yield p


def safe_relpath(path, start):
    try:
        return os.path.relpath(path, start)
    except Exception:
        return os.path.basename(path)


def clean_png_chunks_bytes(in_path: str, out_path: str, remove_types: set):
    """
    Remove specified chunk types by copying all other chunks verbatim.
    Does NOT decode image. Keeps original CRCs for kept chunks (valid).
    """
    removed = {t: 0 for t in remove_types}

    with open(in_path, "rb") as f:
        sig = f.read(8)
        if sig != PNG_SIG:
            return False, removed, "not_png_signature"

        data = f.read()

    # parse chunks from memory
    # chunk format: length(4) type(4) data(length) crc(4)
    pos = 0
    out = bytearray()
    out += PNG_SIG

    # Safety: avoid infinite loops
    max_len = len(data)

    found_iend = False

    while pos + 12 <= max_len:
        length_bytes = data[pos:pos+4]
        ctype = data[pos+4:pos+8]
        if len(length_bytes) < 4 or len(ctype) < 4:
            break

        length = int.from_bytes(length_bytes, "big")
        chunk_total = 12 + length  # len + type + data + crc

        if pos + chunk_total > max_len:
            # truncated/broken file
            return False, removed, "truncated_chunk"

        chunk = data[pos:pos+chunk_total]
        ctype_str = ctype.decode("ascii", errors="ignore")

        if ctype_str in remove_types:
            removed[ctype_str] = removed.get(ctype_str, 0) + 1
            # skip this chunk
        else:
            out += chunk

        pos += chunk_total

        if ctype == b"IEND":
            found_iend = True
            break

    if not found_iend:
        return False, removed, "no_iend"

    # write output (atomic replace if same path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as w:
        w.write(out)
    os.replace(tmp_path, out_path)

    return True, removed, ""


def main():
    args = parse_args()
    src_dir = os.path.abspath(args.src_dir)

    if not os.path.isdir(src_dir):
        raise RuntimeError(f"src_dir not found: {src_dir}")

    remove_types = set(args.remove or [])
    # chunk type must be 4 chars
    for t in list(remove_types):
        if len(t) != 4:
            raise RuntimeError(f"Invalid chunk type '{t}'. Must be 4 chars like iCCP, sRGB, gAMA, cHRM.")

    if args.inplace:
        out_dir = src_dir
    else:
        out_dir = os.path.abspath(args.out_dir) if args.out_dir else (src_dir + "__clean")
        os.makedirs(out_dir, exist_ok=True)

    files = list(iter_pngs(src_dir, recursive=args.recursive))
    stats = {
        "total": len(files),
        "ok": 0,
        "skipped_exists": 0,
        "failed": 0,
        "removed_chunks": {t: 0 for t in remove_types},
        "fail_reasons": {}
    }

    for p in tqdm(files, desc="Cleaning PNG", ncols=100):
        rel = safe_relpath(p, src_dir)
        if args.inplace:
            out_path = p
            if args.backup:
                bak = p + ".bak"
                if not os.path.exists(bak):
                    try:
                        # make backup once
                        with open(p, "rb") as f_in, open(bak, "wb") as f_out:
                            f_out.write(f_in.read())
                    except Exception:
                        pass
        else:
            out_path = os.path.join(out_dir, rel)

        if (not args.inplace) and args.skip_existing and os.path.isfile(out_path):
            stats["skipped_exists"] += 1
            continue

        ok, removed, reason = clean_png_chunks_bytes(p, out_path, remove_types)
        if ok:
            stats["ok"] += 1
            for k, v in removed.items():
                stats["removed_chunks"][k] = stats["removed_chunks"].get(k, 0) + v
        else:
            stats["failed"] += 1
            stats["fail_reasons"][reason] = stats["fail_reasons"].get(reason, 0) + 1

    print("\n=== DONE ===")
    print(f"src_dir     : {src_dir}")
    print(f"out_dir     : {out_dir}  (inplace={args.inplace})")
    print(f"recursive   : {args.recursive}")
    print(f"remove      : {sorted(list(remove_types))}")
    print(f"total/png   : {stats['total']}")
    print(f"ok          : {stats['ok']}")
    print(f"skipped     : {stats['skipped_exists']}")
    print(f"failed      : {stats['failed']}")
    print("removed_chunks:")
    for k, v in sorted(stats["removed_chunks"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")
    if stats["fail_reasons"]:
        print("fail_reasons:")
        for k, v in sorted(stats["fail_reasons"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()