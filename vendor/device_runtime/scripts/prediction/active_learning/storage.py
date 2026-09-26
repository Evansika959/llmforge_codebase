"""Atomic commits, integrity checks and OS-released locks for resumable rounds."""
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_bundle(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_configs(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def locked(path):
    """The file can remain after a crash; only the OS lock indicates ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process owns {path}; do not delete a live lock') from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid())+'\n')
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
