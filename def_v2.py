import pandas as pd
import nest
from mpi4py import MPI
import pickle
from pathlib import Path
import sys

nest.set_verbosity('M_ERROR')

resolution = 0.1


bad_sources = [10, 49, 41, 41, 9, 44, 19, 46, 9, 25, 11, 33, 46, 37,
               36, 27, 45, 29, 15, 27, 21, 50, 27, 38, 3, 5, 38, 3,
               41, 49, 42, 37, 36, 45, 5, 3, 21, 29, 9, 30, 34, 40,
               35, 44, 41, 48, 44, 27, 36, 47]
good_sources = [45, 50, 37, 13, 47, 29, 9, 46, 15, 10, 15, 38, 34,
                29, 47, 45, 14, 23, 35, 1, 44, 3, 20, 46, 46, 13, 3,
                49, 3, 48, 5, 9, 15, 28, 30, 25, 40, 30, 16, 3, 40,
                40, 24, 40, 40, 17, 50, 32, 43, 42 ]

    
def runner(sources, compress):
    nest.ResetKernel()
    nest.resolution = resolution
    nest.total_num_virtual_procs = 4
    nest.use_compressed_spikes = compress
    nest.buffer_growth_extra = 0
    
    p1 = nest.Create('parrot_neuron', 50)
    sg = nest.Create('spike_generator', params={'spike_times': [0.1]})
    sr = nest.Create('spike_recorder')

    for (idx, n_src) in enumerate(sources):
        nest.Connect(p1[n_src-1], p1[idx], syn_spec={'delay': resolution})

    nest.Connect(sg, p1, syn_spec={'delay': resolution, 'receptor_type': 0})
    nest.Connect(p1, sr, syn_spec={'delay': resolution})
    
    nest.Simulate(0.2)
    for k, v in nest.GetKernelStatus().items():
        if k.startswith('buffer_'):
            print(f'{k:25s}:{v:6}')
    
    return pd.DataFrame.from_records(sr.events)


def checker(res):
    rpc = {}
    for nranks, rr in res.items():
        ar = pd.concat(rr.values(), ignore_index=True)
        ar.sort_values(['times', 'senders'], inplace=True, ignore_index=True)
        rpc[nranks] = ar
    nn = list(rpc.keys())
    ref = rpc[nn[0]]
    for n in nn[1:]:
        pd.testing.assert_frame_equal(ref, rpc[n])


if __name__ == '__main__':
    print(sys.argv)
    assert len(sys.argv) == 4
    outdir = sys.argv[1]
    compress = sys.argv[2] == 'True'
    print(compress)
    source_set = good_sources if sys.argv[3] == 'good' else bad_sources
    res = runner(source_set, compress)

    with open(Path(outdir)/f'{MPI.COMM_WORLD.size}-{MPI.COMM_WORLD.rank}', 'wb') as pkl:
        pickle.dump(res, pkl, pickle.HIGHEST_PROTOCOL)

    
