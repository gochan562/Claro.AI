"""Safe persistence helpers: per-experiment directories, no user paths."""
import hashlib
import json
import os
import re

import joblib

ID_RE = re.compile(r'^(mlw|exp)_[a-f0-9]{12}$')


def check_id(value, prefix):
    if not isinstance(value, str) or not ID_RE.match(value) or not value.startswith(prefix + '_'):
        raise ValueError('Invalid id: %r.' % (value,))
    return value


def safe_join(base, *parts):
    """Join under base; reject traversal/absolute segments. Returns abspath."""
    base = os.path.abspath(base)
    for p in parts:
        if not isinstance(p, str) or not p or os.path.isabs(p) or '..' in p.split(os.sep):
            raise ValueError('Unsafe path segment: %r.' % (p,))
    out = os.path.abspath(os.path.join(base, *parts))
    if out != base and not out.startswith(base + os.sep):
        raise ValueError('Path escapes base directory.')
    return out


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def dump_joblib(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    if not os.path.exists(path):
        raise ValueError('Artifact not found: %s.' % os.path.basename(path))
    return joblib.load(path)


def sha256_of_obj(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def sha256_of_file(path, limit_bytes=None):
    h = hashlib.sha256()
    left = limit_bytes
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536 if left is None else min(65536, left))
            if not chunk:
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
                if left <= 0:
                    break
    return h.hexdigest()
