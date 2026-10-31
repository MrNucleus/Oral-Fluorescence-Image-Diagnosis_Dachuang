"""Bounded retry for transient Windows file locks; telemetry is best effort."""
import json
import os
from pathlib import Path
import tempfile
import time
import warnings


def replace_with_retry(source, target, attempts=10):
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(.02 * 2**attempt, .25))


def atomic_write(path, writer):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(fd)
    temporary=Path(name)
    try:
        writer(temporary)
        temporary.chmod(0o600)
        replace_with_retry(temporary,path)
    finally:
        # Only our own disposable temporary is removed; historical files stay intact.
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_json(path,value):
    path=Path(path)
    payload=json.dumps(value,ensure_ascii=False,indent=2)
    try:
        atomic_write(path,lambda target:target.write_text(payload,encoding='utf-8'))
        return True
    except OSError as error:
        if path.name!='live_progress.json':
            raise
        warnings.warn(f'Live progress update skipped after I/O retries: {error}',RuntimeWarning)
        return False


def save_checkpoint(value,path):
    import torch
    atomic_write(path,lambda target:torch.save(value,target))
