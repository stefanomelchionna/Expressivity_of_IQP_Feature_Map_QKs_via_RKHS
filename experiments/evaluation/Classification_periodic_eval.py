'''
Evaluation of the periodic classification experiment (Section IV A, Figure 9).

For every quantum kernel written by experiments/QK_computation/QK_computation_classification_periodic.py, an SVM
with precomputed kernel is trained on the training kernel and evaluated on the test set. The same is done with the
classical RBF kernel with default gamma = 1/(D * Var(X_train)).

Usage (from the repository root):
    PYTHONPATH=. python experiments/evaluation/Classification_periodic_eval.py \
        --config configuration_files/config_classification_periodic_eval.json

Output: <Output_dir>/classification_periodic_results.csv, one row per (kernel configuration, K_type).
'''

import numpy as np
import glob
import pandas as pd
import scipy.sparse as sparse
import argparse
import json
import os

from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score

from src.utils import Compute_rbf_kernel, compute_gd


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_classification_periodic_eval.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
config=args['config_file']

###########Load hyperparameters from json################
print('Loading Parameters')
with open(config) as f:
    params = json.load(f)

out_dir=params["Output_dir"]
os.makedirs(out_dir,exist_ok=True)

#############################################################################################################################
#                                                   Evaluation                                                              #
#############################################################################################################################

kernel_dir=params["Kernel_dir"]
results=[]

# <Kernel_dir>/Sample_<n>/Oscillations_<omega>/<D>_qubits/IQP_<ent>_reps_<r>/qk_tr_<b>.npz
for i in sorted(glob.glob(os.path.join(kernel_dir, '*', '*', '*', '*', 'qk_tr_*.npz'))):
    path = os.path.normpath(i).split(os.sep)
    sampling = int(path[-5].split('_')[1])
    o = int(path[-4].split('_')[1])
    n_qubits = int(path[-3].split('_')[0])
    ent_type = path[-2].split('_')[1]
    reps = int(path[-2].split('_')[3])
    bandwidth = float(path[-1].split('_')[2].split('.npz')[0])

    #load data (train and test)
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(i)), 'data.csv'))
    X_train=data[data['train']==1].iloc[:,:n_qubits]
    X_test=data[data['train']==0].iloc[:,:n_qubits]
    y_train=data[data['train']==1].loc[:,'y']
    y_test=data[data['train']==0].loc[:,'y']

    #Compute Classical Kernel
    c_kernel_tr=Compute_rbf_kernel(X_train,X_train)
    c_kernel_ts=Compute_rbf_kernel(X_test,X_train)

    #Load quantum kernels: train (n_train x n_train) and train-test (n_train x n_test)
    q_kernel_tr=sparse.load_npz(i).toarray()
    q_kernel_ts=sparse.load_npz(os.path.join(os.path.dirname(i), path[-1].replace('qk_tr_', 'qk_ts_'))).toarray()

    #check for nan in kernel
    if np.isnan(q_kernel_tr).any():
        print('Quantum Kernel with {} samples {} ent , {} qubits, {} oscillation, {} reps, {} bandwidth contains NaN, skipped'.format(
            sampling,ent_type,n_qubits,o,reps, bandwidth))
        continue

    gd=compute_gd(c_kernel_tr,q_kernel_tr)

    #PERFORM CLASSIFICATION
    qsvc=SVC(kernel='precomputed',random_state=42)
    csvc=SVC(kernel='precomputed',random_state=42)
    qsvc.fit(q_kernel_tr,y_train)
    csvc.fit(c_kernel_tr,y_train)

    #Predict
    y_pred_q_tr=qsvc.predict(q_kernel_tr)
    y_pred_q=qsvc.predict(q_kernel_ts.T)
    y_pred_c_tr=csvc.predict(c_kernel_tr)
    y_pred_c=csvc.predict(c_kernel_ts)

    #Compute metrics
    results.append({
            'K_type':'Quantum',
            'bandwidth':bandwidth,
            'oscillations':o,
            'ent_type':ent_type,
            'sampling':sampling,
            'n_qubits':n_qubits,
            'reps':reps,
            'gd':gd,
            'acc_tr':accuracy_score(y_train,y_pred_q_tr),
            'acc_ts':accuracy_score(y_test,y_pred_q),
            'f1_tr':f1_score(y_train,y_pred_q_tr,average='weighted'),
            'f1_ts':f1_score(y_test,y_pred_q,average='weighted'),
            'prec_tr':precision_score(y_train,y_pred_q_tr,average='weighted'),
            'prec_ts':precision_score(y_test,y_pred_q,average='weighted'),
        })

    results.append({
            'K_type':'Classical',
            'bandwidth':bandwidth,
            'oscillations':o,
            'ent_type':None,
            'sampling':sampling,
            'n_qubits':n_qubits,
            'reps':None,
            'gd':gd,
            'acc_tr':accuracy_score(y_train,y_pred_c_tr),
            'acc_ts':accuracy_score(y_test,y_pred_c),
            'f1_tr':f1_score(y_train,y_pred_c_tr,average='weighted'),
            'f1_ts':f1_score(y_test,y_pred_c,average='weighted'),
            'prec_tr':precision_score(y_train,y_pred_c_tr,average='weighted'),
            'prec_ts':precision_score(y_test,y_pred_c,average='weighted'),
        })
    print('Done: {} samples, {} oscillations, {} qubits, {} ent, {} reps, bandwidth {}'.format(
        sampling, o, n_qubits, ent_type, reps, bandwidth), flush=True)

#Save results
pd.DataFrame(results).to_csv(os.path.join(out_dir,'classification_periodic_results.csv'),index=False)
