import os
import shutil
from contextlib import contextmanager
from tempfile import mkdtemp, mkstemp

@contextmanager
def tmpfile(dir='/tmp'):
    fid, fname = mkstemp(dir=dir)
    try:
        os.close(fid)
        yield fname
    finally:
        os.unlink(fname)

@contextmanager
def tmpdir(dir='/tmp'):
    dirname = mkdtemp(dir=dir)
    try:
        yield dirname
    finally:
        shutil.rmtree(dirname)