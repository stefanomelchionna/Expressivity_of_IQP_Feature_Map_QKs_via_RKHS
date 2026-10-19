'''
Evaluation of the periodic anomaly-detection experiment (Section IV A, Figure 10).

For every quantum kernel written by experiments/QK_computation/QK_computation_AD_periodic.py, a One-Class SVM
(nu = 0.2) with precomputed kernel is fitted on all points and its predictions on the same points are scored against
the true labels (in-sample protocol: no labels are used in training). The same is done with the classical RBF kernel
with default gamma = 1/(D * Var(X)).

Usage (from the repository root):
    PYTHONPATH=. python experiments/evaluation/AD_periodic_eval.py \
        --config configuration_files/config_AD_periodic_eval.json

Output: <Output_dir>/AD_periodic_results.csv, one row per (kernel configuration, K_type).
'''

import numpy as np
import glob
import pandas as pd
import argparse
import json
import os

from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.utils import Compute_rbf_kernel, compute_gd, Kernel_concentration


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_AD_periodic_eval.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
config=args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(config) as f:
    params = json.load(f)
print(params)

out_dir=params["Output_dir"]
os.makedirs(out_dir,exist_ok=True)


def ocsvm_metrics(kernel, labels):
    """Fit a One-Class SVM on `kernel` and score its predictions (anomaly = 1) against `labels`."""
    oc_svm = OneClassSVM(kernel='precomputed', nu=0.2)
    oc_svm.fit(kernel)
    scores = oc_svm.score_samples(kernel)
    predictions = ((oc_svm.predict(kernel) - 1) * (-0.5)).astype(int)   # +1 (inlier) -> 0, -1 (outlier) -> 1
    return {
        "F1": f1_score(labels, predictions),
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions),
        "Recall": recall_score(labels, predictions),
        "Avg_anomaly_score_on_true_labels": np.mean(scores[labels == 1]),
        "Avg_anomaly_score": np.mean(scores),
        "Variance_anomaly_score": np.var(scores),
    }


#############################################################################################################################
#                                                   Evaluation                                                              #
#############################################################################################################################

kernel_dir = params["Kernel_dir"]
results=[]

# <Kernel_dir>/Sample_<n>/Oscillations_<omega>/<D>_qubits/K_train_<ent>_rep_<r>_b_<b>.npy
for i in sorted(glob.glob(os.path.join(kernel_dir, '*', '*', '*', 'K_train_*.npy'))):
    path_parts = os.path.normpath(i).split(os.sep)
    print('Processing file:', i, flush=True)

    sampling = int(path_parts[-4].split('_')[1])
    o = int(path_parts[-3].split('_')[1])
    n_qubits = int(path_parts[-2].split('_')[0])
    ent_type = path_parts[-1].split('_')[2]
    reps = int(path_parts[-1].split('_')[4])
    bandwidth = float(path_parts[-1].split('_')[6].split('.npy')[0])

    #load data
    data = pd.read_csv(os.path.join(os.path.dirname(i), 'data.csv'))
    labels=data['label'].to_numpy()

    #Compute Classical Kernel
    c_kernel=Compute_rbf_kernel(data.iloc[:,:-1],data.iloc[:,:-1])
    c_conc=Kernel_concentration(c_kernel)

    #Load quantum kernel
    q_kernel=np.load(i)
    qk_conc=Kernel_concentration(q_kernel)

    gd=compute_gd(c_kernel,q_kernel)

    # perform AD and save results
    results.append( {
        "K_type":"Quantum",
        "Sampling":sampling,
        "oscillations":o,
        "N_qubits":n_qubits,
        "Ent_type":ent_type,
        "Reps":reps,
        "Bandwidth":bandwidth,
        "K_concentration":qk_conc,
        "GD":gd,
        **ocsvm_metrics(q_kernel, labels),
        })

    results.append( {
        "K_type":"Classical",
        "Sampling":sampling,
        'oscillations':o,
        "N_qubits":n_qubits,
        "Ent_type":None,
        "Reps":None,
        "Bandwidth":bandwidth,
        "K_concentration":c_conc,
        "GD":gd,
        **ocsvm_metrics(c_kernel, labels),
        })

#Save results
pd.DataFrame(results).to_csv(os.path.join(out_dir,'AD_periodic_results.csv'),index=False)
