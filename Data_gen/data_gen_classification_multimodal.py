'''
Generates and saves the datasets of the multimodal classification (Section IV B) experiment: two Gaussian-mixture classes with C centres each (Eq. (7)), split into train/test (random_state 42).

The kernel scripts in experiments/QK_computation/ regenerate exactly the same data with the same generators and seeds
(src/data.py), so this step is optional: it only exports the datasets, e.g. for inspection or for other models.

Usage (from the repository root):
    PYTHONPATH=. python Data_gen/data_gen_classification_multimodal.py --config config/config_classification_multimodal.json

Output: <output dir>/Kernels/Sample_<n>/<C>_clusters/<D>_qubits/data.csv, train_data.csv, test_data.csv (Feature_1..Feature_D, label)
'''

import argparse
import json
import os

import pandas as pd
import sklearn.model_selection

from src.data import create_blob_multimodal_classification

ap = argparse.ArgumentParser()
ap.add_argument('-config', '--config_file',
                 default='config/config_classification_multimodal.json',
                 required=False,
                 help='json file with experiments info path')

args = vars(ap.parse_args())
params_dir = args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(params_dir) as f:
    params = json.load(f)

output_path = params['output_path']
os.makedirs(output_path, exist_ok=True)

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
            df = create_blob_multimodal_classification(
                n_features=n_qubits, sample_size=n_sample + params['n_samples_test'], number_of_blobs=n_clusters
            )
            df.to_csv(out_dir + 'data.csv', index=False)

            # train and test split
            X = df.drop(columns=['label']).values
            y = df['label'].values
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=params['n_samples_test'], random_state=42
            )

            df_train = pd.DataFrame(x_train, columns=[f'Feature_{i + 1}' for i in range(n_qubits)])
            df_train['label'] = y_train
            df_train.to_csv(out_dir + 'train_data.csv', index=False)

            df_test = pd.DataFrame(x_test, columns=[f'Feature_{i + 1}' for i in range(n_qubits)])
            df_test['label'] = y_test
            df_test.to_csv(out_dir + 'test_data.csv', index=False)

            print('Data saved to {}'.format(out_dir))
