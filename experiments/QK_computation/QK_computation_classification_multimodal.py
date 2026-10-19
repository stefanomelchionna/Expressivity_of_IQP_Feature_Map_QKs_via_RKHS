'''
Quantum kernels for the multimodal classification experiment (Section IV B, Figure 11(a)).

For every training-set size, number of centres C and number of qubits D in the config, the script generates
two multimodal classes (Gaussian mixtures with C centres each, Eq. (7)), splits them into train and test sets
(n_samples_test points, random_state 42), min-max scales the features to [0, b * pi] (scaler fitted on train + test)
and computes the train and train-test IQP kernels for every entanglement pattern, repetitions r and bandwidth b.

Usage (from the repository root):
    PYTHONPATH=. python experiments/QK_computation/QK_computation_classification_multimodal.py \
        --config configuration_files/config_classification_multimodal.json

Output, in <output_path>/Kernels/Sample_<n>/<C>_clusters/<D>_qubits/:
    data.csv, train_data.csv, test_data.csv   features Feature_1..Feature_D and label
    IQP_<ent>_reps_<r>/qk_tr_<b>.npz          train kernel (scipy sparse format)
    IQP_<ent>_reps_<r>/qk_ts_<b>.npz          train-test kernel
'''

import numpy as np
import pandas as pd
from scipy import sparse
import os
import argparse
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.QK import QuantumKernel
from src.data import create_blob_multimodal_classification


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_classification_multimodal.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(params_dir) as f:
    params = json.load(f)

output_path=params['output_path']
os.makedirs(output_path+'/Kernels',exist_ok=True)

############################################################################################
for n_sample in params['n_samples']:
    for n_clusters in params['n_clusters']:
        for n_qubits in params['n_qubits']:
            print('==============Computing kernel for {} samples with {} clusters and {} qubits==============='.format(n_sample,n_clusters,n_qubits),flush=True)
            data_dir = output_path+'/Kernels/Sample_{}/{}_clusters/{}_qubits/'.format(n_sample,n_clusters,n_qubits)
            os.makedirs(data_dir,exist_ok=True)

            # Generate the data (train + test) and save it
            df=create_blob_multimodal_classification(n_features=n_qubits, sample_size=n_sample+params['n_samples_test'], number_of_blobs=n_clusters)
            df.to_csv(data_dir+'data.csv',index=False)

            # train and test split
            X=df.drop(columns=['label']).values
            y=df['label'].values
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=params['n_samples_test'], random_state=42)
            #save the data with train and test split
            df_train = pd.DataFrame(x_train, columns=[f'Feature_{i+1}' for i in range(n_qubits)])
            df_train['label'] = y_train
            df_train.to_csv(data_dir+'train_data.csv',index=False)
            df_test = pd.DataFrame(x_test, columns=[f'Feature_{i+1}' for i in range(n_qubits)])
            df_test['label'] = y_test
            df_test.to_csv(data_dir+'test_data.csv',index=False)

            # Compute kernel
            for ent in params['entanglement']:
                for rep in params['n_repeats']:
                    kernel_dir = data_dir+'IQP_{}_reps_{}/'.format(ent,rep)
                    os.makedirs(kernel_dir,exist_ok=True)
                    for b in params['bandwidth']:
                        # bandwidth: features scaled to [0, b * pi]
                        scaler = MinMaxScaler(feature_range = (0,np.pi*b))
                        scaler.fit(X)
                        X_train_scaled = scaler.transform(x_train)
                        X_test_scaled = scaler.transform(x_test)

                        print('Computing kernel for {} entanglement with {} reps and bandwidth {} '.format(ent,rep,b),flush=True)
                        qk = QuantumKernel(n_qubits=n_qubits, ent_type=ent, n_repeats=rep)
                        t_start = datetime.datetime.now()
                        K_train = qk.kernel_matrix(X_train_scaled,X_train_scaled)
                        K_test = qk.kernel_matrix(X_train_scaled,X_test_scaled)
                        print('Time to compute kernels: {}'.format(datetime.datetime.now()-t_start),flush=True)

                        #save kernel
                        sparse.save_npz(kernel_dir+'qk_tr_{}.npz'.format(b),sparse.csr_matrix(K_train))
                        sparse.save_npz(kernel_dir+'qk_ts_{}.npz'.format(b),sparse.csr_matrix(K_test))
                        print('Kernel saved')
