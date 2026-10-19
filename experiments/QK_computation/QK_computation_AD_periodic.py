'''
Quantum kernels for the periodic anomaly-detection experiment (Section IV A, Figure 10).

For every number of oscillations omega, number of qubits D and dataset size in the config, the script generates
the periodic AD dataset (inliers labelled 0 by Eq. (6) plus 20% uniform outliers, seed 42), rescales the data by
b * pi and computes the IQP kernel on all points for every entanglement pattern, repetitions r and bandwidth b.

Usage (from the repository root):
    PYTHONPATH=. python experiments/QK_computation/QK_computation_AD_periodic.py \
        --config configuration_files/config_AD_periodic.json

Output, in <Data_dir>/Kernels/Sample_<n>/Oscillations_<omega>/<D>_qubits/:
    data.csv                           features and label (0 = inlier, 1 = outlier)
    K_train_<ent>_rep_<r>_b_<b>.npy    kernel matrix
Kernels containing NaN are skipped and listed in <Data_dir>/nan_ad.txt.
'''

import numpy as np
import argparse
import json
import os

from src.QK import QuantumKernel
from src.data import create_periodic_dataset_AD


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_AD_periodic.json',
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
print(output_path+'/Kernels')

############################################################################################
# log of the kernels containing NaN values
nan_log = os.path.join(output_path, 'nan_ad.txt')
with open(nan_log,'w') as f:
    f.write('')

count=0

for oscillations in params['oscillations']:
    for n_qubits in params['n_qubits']:
        for n_samples in params['n_samples']:
            print('==============Computing kernel for {} oscillations with {} qubits==============='.format(oscillations,n_qubits),flush=True)
            data_dir = output_path+'/Kernels/Sample_{}/Oscillations_{}/{}_qubits/'.format(n_samples, oscillations, n_qubits)
            os.makedirs(data_dir, exist_ok=True)

            # Generate the data
            df = create_periodic_dataset_AD(oscillations = oscillations, sample_size = n_samples, number_of_features = n_qubits)
            assert not df.isnull().values.any(), 'NaN values detected in the data'

            #save the data
            df.to_csv(data_dir+'data.csv',index=False)

            #extract features and labels
            x_train = df.iloc[:,:n_qubits].values

            # Compute kernel
            for ent in params['entanglement']:
                for rep in params['n_repeats']:
                    for b in params['bandwidth']:
                        #perform bandwidth adjustment
                        x_train_scaled = x_train * b * np.pi

                        K_train =QuantumKernel(n_qubits=n_qubits, ent_type=ent,n_repeats=rep).kernel_matrix(x_train_scaled,x_train_scaled)
                        print('Kernel shape:', K_train.shape,flush=True)
                        if np.isnan(K_train).any():
                            print('Nan detected in kernel',flush=True)
                            with open(nan_log,'a') as f:
                                f.write('Sample: {}, Oscillations: {}, Qubits: {}, Entanglement: {}, Reps: {}, Bandwidth: {}\n'.format(
                                    n_samples, oscillations, n_qubits, ent, rep, b))
                            count+=1
                            continue

                        # Save the kernel
                        np.save(data_dir+'K_train_{}_rep_{}_b_{}.npy'.format(ent,rep,b),K_train)

print('Total number of NaN kernels:', count,flush=True)
