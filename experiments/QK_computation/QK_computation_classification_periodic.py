'''
Quantum kernels for the periodic classification experiment (Section IV A, Figure 9).

For every number of oscillations omega, number of qubits D and training-set size in the config, the script
generates the training set (seed 0) and the test set (seed 1) with Eq. (6), rescales the data by b * pi and
computes the train (n_train x n_train) and train-test (n_train x n_test) IQP kernels for every entanglement
pattern, number of repetitions r and bandwidth b.

Usage (from the repository root):
    PYTHONPATH=. python experiments/QK_computation/QK_computation_classification_periodic.py \
        --config configuration_files/config_classification_periodic.json

Output, in <Data_dir>/Kernels/Sample_<n>/Oscillations_<omega>/<D>_qubits/:
    data.csv                              features 0..D-1, label y, train/test flags
    IQP_<ent>_reps_<r>/qk_tr_<b>.npz      train kernel (scipy sparse format)
    IQP_<ent>_reps_<r>/qk_ts_<b>.npz      train-test kernel
'''

import numpy as np
import argparse
import json
import os
import pandas as pd
import scipy.sparse as sparse

from src.QK import QuantumKernel
from src.data import compute_periodic_data_classification


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_classification_periodic.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(params_dir) as f:
    params = json.load(f)

output_path=params['Data_dir']
os.makedirs(output_path+'/Kernels',exist_ok=True)


############################################################################################
train_seed = 0
test_seed = 1

for oscillations in params['oscillations']:
    for n_qubits in params['n_qubits']:
        for n_samples in params['n_samples_train']:
            print('==============Computing kernel for {} oscillations with {} qubits==============='.format(oscillations,n_qubits),flush=True)

            # Generate the data
            x_train, y_train = compute_periodic_data_classification(oscillations=oscillations, sample_size=n_samples, seed=train_seed, number_of_features=n_qubits)
            x_test, y_test = compute_periodic_data_classification(oscillations=oscillations, sample_size=params['n_samples_test'], seed=test_seed, number_of_features=n_qubits)

            # Save the data in a df
            df_train=pd.DataFrame(x_train)
            df_train['y']=y_train
            df_train['train'] = 1
            df_train['test'] = 0

            df_test=pd.DataFrame(x_test)
            df_test['y']=y_test
            df_test['train'] = 0
            df_test['test'] = 1

            df = pd.concat([df_train, df_test], ignore_index=True)

            # Compute kernel
            for ent in params['entanglement']:
                for rep in params['n_repeats']:
                    kernel_dir = output_path+'/Kernels/Sample_{}/Oscillations_{}/{}_qubits/IQP_{}_reps_{}'.format(
                        n_samples, oscillations, n_qubits, ent, rep)
                    os.makedirs(kernel_dir, exist_ok=True)

                    for b in params['bandwidth']:
                        #perform bandwidth adjustment
                        x_train_scaled = x_train * b * np.pi
                        x_test_scaled = x_test * b * np.pi

                        qk = QuantumKernel(n_qubits=n_qubits, ent_type=ent, n_repeats=rep)
                        K_train = qk.kernel_matrix(x_train_scaled, x_train_scaled)
                        K_test = qk.kernel_matrix(x_train_scaled, x_test_scaled)
                        print('Quantum Kernel with {} samples {} ent , {} qubits, {} oscillation, {} reps, {} bandwidth'.format(
                            n_samples, ent, n_qubits, oscillations, rep, b), flush=True)

                        # Save the kernel
                        sparse.save_npz(kernel_dir+'/qk_tr_{}.npz'.format(b), sparse.csr_matrix(K_train))
                        sparse.save_npz(kernel_dir+'/qk_ts_{}.npz'.format(b), sparse.csr_matrix(K_test))

            #save the data
            df.to_csv(output_path+'/Kernels/Sample_{}/Oscillations_{}/{}_qubits/data.csv'.format(n_samples,oscillations,n_qubits),index=False)
