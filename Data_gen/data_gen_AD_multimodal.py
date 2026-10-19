'''
Generates and saves the datasets of the multimodal anomaly detection (Section IV B) experiment: Gaussian mixture with C centres plus 20% uniform outliers, min-max scaled to [0, 1] (seed 42).

The kernel scripts in experiments/QK_computation/ regenerate exactly the same data with the same generators and seeds
(src/data.py), so this step is optional: it only exports the datasets, e.g. for inspection or for other models.

Usage (from the repository root):
    PYTHONPATH=. python Data_gen/data_gen_AD_multimodal.py --config config/config_AD_multimodal.json

Output: <output dir>/Kernels/Sample_<n>/<C>_clusters/<D>_qubits/data.csv (Feature_1..Feature_D, label: 0 = inlier, 1 = outlier)
'''

import argparse
import json
import os

from src.data import create_blob_with_outliers

ap = argparse.ArgumentParser()
ap.add_argument('-config', '--config_file',
                 default='config/config_AD_multimodal.json',
                 required=False,
                 help='json file with experiments info path')

args = vars(ap.parse_args())
params_dir = args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(params_dir) as f:
    params = json.load(f)

output_path = params['output_path']
os.makedirs(output_path + '/Kernels', exist_ok=True)

############################################################################################
for n_sample in params['n_samples']:
    for n_clusters in params['n_clusters']:
        for n_qubits in params['n_qubits']:
            print('==============Generating data for {} samples with {} clusters and {} qubits==============='.format(
                n_sample, n_clusters, n_qubits), flush=True)

            out_dir = output_path + '/Kernels/Sample_{}/{}_clusters/{}_qubits/'.format(
                n_sample, n_clusters, n_qubits)
            os.makedirs(out_dir, exist_ok=True)

            # Generate the data
            df = create_blob_with_outliers(
                n_features=n_qubits, sample_size=n_sample, number_of_blobs=n_clusters, ratio_outliers=0.2
            )
            df.to_csv(out_dir + 'data.csv', index=False)
            print('Data saved to {}'.format(out_dir + 'data.csv'))
