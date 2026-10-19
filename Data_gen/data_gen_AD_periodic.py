'''
Generates and saves the datasets of the periodic anomaly detection (Section IV A) experiment: inliers labelled 0 by Eq. (6) plus 20% uniform outliers (seed 42).

The kernel scripts in experiments/QK_computation/ regenerate exactly the same data with the same generators and seeds
(src/data.py), so this step is optional: it only exports the datasets, e.g. for inspection or for other models.

Usage (from the repository root):
    PYTHONPATH=. python Data_gen/data_gen_AD_periodic.py --config config/config_AD_periodic.json

Output: <output dir>/Kernels/Sample_<n>/Oscillations_<omega>/<D>_qubits/data.csv (Feature_1..Feature_D, label: 0 = inlier, 1 = outlier)
'''

import argparse
import json
import os

from src.data import create_periodic_dataset_AD

# Load hyperparameters from json
ap = argparse.ArgumentParser()
ap.add_argument('-config', '--config_file',
                 default='config/config_AD_periodic.json',
                 required=False,
                 help='json file with experiments info path')

args = vars(ap.parse_args())
params_dir = args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(params_dir) as f:
    params = json.load(f)

output_path = params['Data_dir']
os.makedirs(output_path + '/Kernels', exist_ok=True)

############################################################################################
for oscillations in params['oscillations']:
    for n_qubits in params['n_qubits']:
        for n_samples in params['n_samples']:
            print('==============Generating data for {} oscillations with {} qubits==============='.format(
                oscillations, n_qubits), flush=True)

            out_dir = output_path + '/Kernels/Sample_{}/Oscillations_{}/{}_qubits/'.format(
                n_samples, oscillations, n_qubits)
            os.makedirs(out_dir, exist_ok=True)

            # Generate the data
            df = create_periodic_dataset_AD(
                oscillations=oscillations, sample_size=n_samples, number_of_features=n_qubits
            )
            assert not df.isnull().values.any(), 'NaN values detected in the data'

            df.to_csv(out_dir + 'data.csv', index=False)
            print('Data saved to {}'.format(out_dir + 'data.csv'))
