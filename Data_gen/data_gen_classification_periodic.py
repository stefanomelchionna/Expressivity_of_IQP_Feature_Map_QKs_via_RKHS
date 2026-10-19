'''
Generates and saves the datasets of the periodic classification (Section IV A) experiment: training set (seed 0) and test set (seed 1) of Eq. (6).

The kernel scripts in experiments/QK_computation/ regenerate exactly the same data with the same generators and seeds
(src/data.py), so this step is optional: it only exports the datasets, e.g. for inspection or for other models.

Usage (from the repository root):
    PYTHONPATH=. python Data_gen/data_gen_classification_periodic.py --config config/config_classification_periodic.json

Output: <output dir>/Kernels/Sample_<n>/Oscillations_<omega>/<D>_qubits/data.csv (features 0..D-1, label y, train/test flags)
'''

import argparse
import json
import os

import pandas as pd

from src.data import compute_periodic_data_classification

# Load hyperparameters from json
ap = argparse.ArgumentParser()
ap.add_argument('-config', '--config_file',
                 default='config/config_classification_periodic.json',
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
train_seed = 0
test_seed = 1

for oscillations in params['oscillations']:
    for n_qubits in params['n_qubits']:
        for n_samples in params['n_samples_train']:
            print('==============Generating data for {} oscillations with {} qubits==============='.format(
                oscillations, n_qubits), flush=True)

            # Generate the data
            x_train, y_train = compute_periodic_data_classification(
                oscillations=oscillations, sample_size=n_samples, seed=train_seed, number_of_features=n_qubits
            )
            x_test, y_test = compute_periodic_data_classification(
                oscillations=oscillations, sample_size=params['n_samples_test'], seed=test_seed,
                number_of_features=n_qubits
            )

            # Save the data in a df
            df_train = pd.DataFrame(x_train)
            df_train['y'] = y_train
            df_train['train'] = 1
            df_train['test'] = 0

            df_test = pd.DataFrame(x_test)
            df_test['y'] = y_test
            df_test['train'] = 0
            df_test['test'] = 1

            df = pd.concat([df_train, df_test], ignore_index=True)

            # make directory (same structure as before, without the IQP/ent/rep subfolders)
            out_dir = output_path + '/Kernels/Sample_{}/Oscillations_{}/{}_qubits/'.format(
                n_samples, oscillations, n_qubits)
            os.makedirs(out_dir, exist_ok=True)

            df.to_csv(out_dir + 'data.csv', index=False)
            print('Data saved to {}'.format(out_dir + 'data.csv'))
