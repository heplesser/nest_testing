# must be run with export PYTEST_QUIET=1

import subprocess
import pandas as pd
import json
import importlib

def de_json(res):
    print('*' * 50)
    print(res)
    print('*' * 50)
    rr = {}
    for rj in res.split(b'\n'):
        if rj:
            r = json.loads(rj)
            rr[r['rank']] = pd.read_json(r['res'], precise_float=True)
    return rr

def test_bug():
    script = 'def_bug_mini.py'
    procs = [1, 2]

    res = {n_ranks: de_json(subprocess.run(['mpirun', '-np', str(n_ranks), 'python', script],
                                               capture_output=True,
                                               check=True).stdout)
               for n_ranks in procs}

    mod = importlib.import_module(script.split('.')[0])
    mod.checker(res)

    
