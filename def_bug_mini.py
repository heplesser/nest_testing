import pandas as pd
import nest
import json
from mpi4py import MPI

nest.set_verbosity('M_ERROR')

resolution = 0.1

def runner():
    nest.ResetKernel()
    nest.resolution = resolution
    nest.total_num_virtual_procs = 4

    p1 = nest.Create('parrot_neuron', 50)
    sg = nest.Create('spike_generator', params={'spike_times': [0.1]})
    #p2 = nest.Create('parrot_neuron', 50)
    sr = nest.Create('spike_recorder')

    sources = [10, 49, 41, 41, 9, 44, 19, 46, 9, 25, 11, 33, 46, 37,
               36, 27, 45, 29, 15, 27, 21, 50, 27, 38, 3, 5, 38, 3,
               41, 49, 42, 37, 36, 45, 5, 3, 21, 29, 9, 30, 34, 40,
               35, 44, 41, 48, 44, 27, 36, 47]

    nest.Connect(sg, p1, syn_spec={'delay': resolution, 'receptor_type': 0})

    for (idx, n_src) in enumerate(sources):
        nest.Connect(p1[n_src-1], p1[idx], syn_spec={'delay': resolution})

    # nest.Connect(p1, p1, 'one_to_one', syn_spec={'delay': resolution})

    nest.Connect(p1, sr, syn_spec={'delay': resolution})
    
    nest.Simulate(0.4)

    return pd.DataFrame.from_records(sr.events)


def checker(res):
    rpc = {}
    for nranks, rr in res.items():
        ar = pd.concat(rr.values(), ignore_index=True)
        ar.sort_values(['times', 'senders'], inplace=True, ignore_index=True)
        rpc[nranks] = ar
    print(rpc)
    nn = list(rpc.keys())
    ref = rpc[nn[0]]
    for n in nn[1:]:
        pd.testing.assert_frame_equal(ref, rpc[n])

if __name__ == '__main__':
    res = runner()
    #print(res)
    res.set_index(['times', 'senders']).sort_index().to_csv(f'dm_{MPI.COMM_WORLD.size:02d}_{MPI.COMM_WORLD.rank:02d}.csv')
    #print(json.dumps({'rank': MPI.COMM_WORLD.rank, 'res': res.to_json(double_precision=15)}))

    
