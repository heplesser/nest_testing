import pandas as pd
import nest
import json
from mpi4py import MPI

nest.set_verbosity('M_ERROR')

resolution = 0.1

def runner():
    NE = 4
    simtime = 0.3
    neuron_params = {'tau_m': 20.,
                     'tau_syn_ex': 0.5,
                     'tau_syn_in': 0.5,
                     't_ref': 2.0,
                     'E_L': 0.0,
                     'V_th': 20.0,
                     'V_reset': 0.0,
                     'C_m': 250.,
                     'I_e': 600.}
    sources = [10, 49, 41, 41, 9, 44, 19, 46, 9, 25, 11, 33, 46, 37,
               36, 27, 45, 29, 15, 27, 21, 50, 27, 38, 3, 5, 38, 3,
               41, 49, 42, 37, 36, 45, 5, 3, 21, 29, 9, 30, 34, 40,
               35, 44, 41, 48, 44, 27, 36, 47]

    nest.ResetKernel()
    nest.resolution = resolution
    nest.total_num_virtual_procs = 4
    
    E_neurons = nest.Create('iaf_psc_alpha', n=NE, params=neuron_params)
    for nrn in E_neurons:
        nrn.V_m = neuron_params['V_th']

        
    print('VVVVV', E_neurons.vp)
    print(MPI.COMM_WORLD.rank, 'TTTTT', E_neurons.thread)

    #sr = nest.Create('spike_recorder', params={'time_in_steps': True})
    mm = nest.Create('multimeter', params={'record_from': ['I_syn_ex', 'V_m'], 'interval': 0.1})
    for (idx, n_src) in enumerate(sources):
        nest.Connect([n_src], [idx+1], 'one_to_one', {'synapse_model': 'static_synapse',
                                                      'weight': [20.],
                                                      'delay': [0.1]})

    #nest.Connect(E_neurons, sr)
    nest.Connect(mm, E_neurons)
    
    nest.Simulate(simtime+resolution)

    return pd.DataFrame.from_records(mm.events)


def checker(res):
    rpc = {}
    for nranks, rr in res.items():
        ar = pd.concat(rr.values(), ignore_index=True)
        # ar['Time'] = resolution * ar['times'] - ar['offsets']
        ar.sort_values(['times', 'senders'], inplace=True, ignore_index=True)
        rpc[nranks] = ar
    print(rpc)
    nn = list(rpc.keys())
    ref = rpc[nn[0]]
    for n in nn[1:]:
        pd.testing.assert_frame_equal(ref, rpc[n])

if __name__ == '__main__':
    res = runner()
    res.set_index(['times', 'senders']).sort_index().to_csv(f'dbvpa_{MPI.COMM_WORLD.size:02d}_{MPI.COMM_WORLD.rank:02d}.csv')
    #print(json.dumps({'rank': MPI.COMM_WORLD.rank, 'res': res.to_json(double_precision=15)}))

    
