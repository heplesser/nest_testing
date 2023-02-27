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

    bad_conns = [[10, 1],
 [44, 6],
 [46, 8],
 [46, 13],
 [36, 15],
 [50, 22],
 [38, 24],
 [38, 27],
 [42, 31],
 [36, 33],
 [30, 40],
 [34, 41],
 [40, 42],
 [44, 44],
 [48, 46],
 [44, 47],
 [36, 49]]
    
    nest.Connect(sg, p1, syn_spec={'delay': resolution, 'receptor_type': 0})

    for src, tgt in bad_conns:
        nest.Connect([src], [tgt], syn_spec={'delay': resolution})

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
    res.set_index(['times', 'senders']).sort_index().to_csv(f'dmbc_{MPI.COMM_WORLD.size:02d}_{MPI.COMM_WORLD.rank:02d}.csv')
    #print(json.dumps({'rank': MPI.COMM_WORLD.rank, 'res': res.to_json(double_precision=15)}))

    
