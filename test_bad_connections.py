import subprocess
import pandas as pd
import importlib
import tempfile
import pickle
from pathlib import Path
import pytest

def run_single(n_ranks, script, tmpdir, compress, source_set):
    subprocess.run(['mpirun', '-np', str(n_ranks), 'python', script, tmpdir, str(compress), source_set],
                                                   capture_output=True,
                                                   check=True)
    res = {}
    for rank in range(n_ranks):
        with open(Path(tmpdir) / f'{n_ranks}-{rank}', 'rb') as pkl:
            res[rank] = pickle.load(pkl)
    return res

script = 'run_bad_connections.py'

#@pytest.mark.parametrize('compress, source_set', [[True, 'good'], [True, 'bad']])
def test_bug(compress=True, source_set='bad'):
    procs = [1, 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        res = {n_ranks: run_single(n_ranks, script, tmpdir, compress, source_set) for n_ranks in procs}

    mod = importlib.import_module(script.split('.')[0])
    mod.checker(res)

    
