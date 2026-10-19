'''
Quantum kernels for the multimodal anomaly-detection experiment (Section IV B, Figure 11(b)).

For every dataset size, number of centres C and number of qubits D in the config, the script generates the
multimodal AD dataset (Gaussian mixture with C centres plus 20% uniform outliers, min-max scaled to [0, 1], seed 42),
scales the features to [0, b * pi] and computes the IQP kernel on all points for every entanglement pattern,
repetitions r and bandwidth b.

Usage (from the repository root):
    PYTHONPATH=. python experiments/QK_computation/QK_computation_AD_multimodal.py \
        --config configuration_files/config_AD_multimodal.json

Output, in <output_path>/Kernels/Sample_<n>/<C>_clusters/<D>_qubits/:
    data.csv                          features Feature_1..Feature_D and label (0 = inlier, 1 = outlier)
    IQP_<ent>_reps_<r>/qk_tr_<b>.npz  kernel matrix (scipy sparse format)
'''

import numpy as np
from scipy import sparse
import os
import argparse
import json
import datetime
from sklearn.preprocessing import MinMaxScaler

from src.QK import QuantumKernel
from src.data import create_blob_with_outliers


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_AD_multimodal.json',
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

            # Generate the data and save it
            df=create_blob_with_outliers(n_features=n_qubits, sample_size=n_sample, number_of_blobs=n_clusters, ratio_outliers=0.2)
            df.to_csv(data_dir+'data.csv',index=False)
            #Extract X (all but label column)
            X=df.drop(columns=['label']).values

            # Compute kernel
            for ent in params['entanglement']:
                for rep in params['n_repeats']:
                    kernel_dir = data_dir+'IQP_{}_reps_{}/'.format(ent,rep)
                    os.makedirs(kernel_dir,exist_ok=True)
                    for b in params['bandwidth']:
                        # bandwidth: features scaled to [0, b * pi]
                        scaler = MinMaxScaler(feature_range = (0,np.pi*b))
                        X_scaled = scaler.fit_transform(X)

                        print('Computing kernel for {} entanglement with {} reps and bandwidth {} '.format(ent,rep,b),flush=True)
                        t_start = datetime.datetime.now()
                        K =QuantumKernel(n_qubits=n_qubits, ent_type=ent,n_repeats=rep).kernel_matrix(X_scaled,X_scaled)
                        print('Time to compute kernel: {}'.format(datetime.datetime.now()-t_start),flush=True)

                        #save kernel
                        sparse.save_npz(kernel_dir+'qk_tr_{}.npz'.format(b),sparse.csr_matrix(K))
                        print('Kernel saved')
