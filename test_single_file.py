import subprocess
import pandas as pd
import importlib
import tempfile
import pickle
from pathlib import Path
import pytest

@pytest.mark.parametrize('compress, source_set', [[True, 'good'], [True, 'bad']])
def test_bug(compress, source_set):
    procs = [1, 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        for n_ranks in procs:
            run_single(n_ranks, tmpdir, compress, source_set)
        checker(procs, tmpdir)

def run_single(n_ranks, tmpdir, *test_args):
    subprocess.run(['mpirun', '-np', str(n_ranks), 'python', __file__, tmpdir, *(str(arg) for arg in test_args)],
                    check=True, capture_output=True)
    
def runner(outdir, source_selection, compress):

    bad_sources = [10, 49, 41, 41, 9, 44, 19, 46, 9, 25, 11, 33, 46, 37,
                       36, 27, 45, 29, 15, 27, 21, 50, 27, 38, 3, 5, 38, 3,
                       41, 49, 42, 37, 36, 45, 5, 3, 21, 29, 9, 30, 34, 40,
                       35, 44, 41, 48, 44, 27, 36, 47]
    good_sources = [45, 50, 37, 13, 47, 29, 9, 46, 15, 10, 15, 38, 34,
                            29, 47, 45, 14, 23, 35, 1, 44, 3, 20, 46, 46, 13, 3,
                            49, 3, 48, 5, 9, 15, 28, 30, 25, 40, 30, 16, 3, 40,
                            40, 24, 40, 40, 17, 50, 32, 43, 42 ]

    sources = good_sources if source_selection == 'good' else bad_sources
    
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.total_num_virtual_procs = 4
    nest.use_compressed_spikes = compress == 'True'
    
    p1 = nest.Create('parrot_neuron', 50)
    sg = nest.Create('spike_generator', params={'spike_times': [0.1]})
    sr = nest.Create('spike_recorder')

    for (idx, n_src) in enumerate(sources):
        nest.Connect(p1[n_src-1], p1[idx], syn_spec={'delay': nest.resolution})

    nest.Connect(sg, p1, syn_spec={'delay': nest.resolution, 'receptor_type': 0})
    nest.Connect(p1, sr, syn_spec={'delay': nest.resolution})
    
    nest.Simulate(0.3)  # must be at least 0.3 to ensure spikes are delivered 
    
    with open(Path(outdir)/f'{nest.num_processes}-{nest.Rank()}', 'wb') as pkl:
        pickle.dump(pd.DataFrame.from_records(sr.events), pkl, pickle.HIGHEST_PROTOCOL)


def checker(n_procs, tmpdir):
    rpc = {}
    for n_ranks in n_procs:
        ar = pd.concat(pickle.load(open(Path(tmpdir) / f'{n_ranks}-{rank}', 'rb')) for rank in range(n_ranks))
        ar.sort_values(['times', 'senders'], inplace=True, ignore_index=True)
        rpc[n_ranks] = ar

    nn = list(rpc.keys())
    ref = rpc[nn[0]]
    for n in nn[1:]:
        pd.testing.assert_frame_equal(ref, rpc[n])


if __name__ == '__main__':
    import nest
    import sys
    runner(*sys.argv[1:])
