# must be run with export PYTEST_QUIET=1

import subprocess
import pandas as pd
import importlib
import tempfile
import pickle
from pathlib import Path
from mpi4py import MPI
import pickle
import sys

import nest

print(__file__)

def run_single(n_ranks, script, tmpdir):
    try:
        sr = subprocess.run(['mpirun', '-np', str(n_ranks), 'python', script, tmpdir],
                                capture_output=True,
                                check=True)
    except subprocess.CalledProcessError as err:
        print(50*'*')
        print(err.cmd)
        print(50*'=')
        print(err.stdout)
        print(50*'=')
        print(err.stderr)
        print(50*'^')
        sys.exit(-2)
        
    print(50*'*')
    print(n_ranks, sr.returncode)
    print(50*'=')
    print(sr.stdout)
    print(50*'-')
    print(sr.stderr)
    print(50*'*')
    res = {}
    for rank in range(n_ranks):
        with open(Path(tmpdir) / f'{n_ranks}-{rank}', 'rb') as pkl:
            res[rank] = pickle.load(pkl)
    return res
        
def test_bug():
    procs = [1, 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        res = {n_ranks: run_single(n_ranks, __file__, tmpdir) for n_ranks in procs}

    for k, v in res.items():
        print(k, type(v))
    rpc = {}
    for nranks, rr in res.items():
        ar = pd.concat(rr.values(), ignore_index=True)
        ar.sort_values(['times', 'senders'], inplace=True, ignore_index=True)
        rpc[nranks] = ar
    nn = list(rpc.keys())
    ref = rpc[nn[0]]
    for n in nn[1:]:
        pd.testing.assert_frame_equal(ref, rpc[n])

        
def runner():
    resolution = 0.1
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.resolution = resolution
    nest.total_num_virtual_procs = 4

    p1 = nest.Create('parrot_neuron', 50)
    sg = nest.Create('spike_generator', params={'spike_times': [0.1]})
    sr = nest.Create('spike_recorder')

    sources = [10, 49, 41, 41, 9, 44, 19, 46, 9, 25, 11, 33, 46, 37,
               36, 27, 45, 29, 15, 27, 21, 50, 27, 38, 3, 5, 38, 3,
               41, 49, 42, 37, 36, 45, 5, 3, 21, 29, 9, 30, 34, 40,
               35, 44, 41, 48, 44, 27, 36, 47]

    nest.Connect(sg, p1, syn_spec={'delay': resolution, 'receptor_type': 0})

    for (idx, n_src) in enumerate(sources):
        nest.Connect(p1[n_src-1], p1[idx], syn_spec={'delay': resolution})

    nest.Connect(p1, sr, syn_spec={'delay': resolution})
    
    nest.Simulate(0.4)

    return pd.DataFrame.from_records(sr.events)


if __name__ == '__main__':
    res = runner()

    with open(Path(sys.argv[1])/f'{MPI.COMM_WORLD.size}-{MPI.COMM_WORLD.rank}', 'wb') as pkl:
        pickle.dump(res, pkl, pickle.HIGHEST_PROTOCOL)

    

    
