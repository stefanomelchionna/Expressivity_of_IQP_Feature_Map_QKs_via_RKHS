'''
Evaluation of the multimodal classification experiment (Section IV B, Figure 11(a)).

For every quantum kernel written by experiments/QK_computation/QK_computation_classification_multimodal.py, an SVM
with precomputed kernel is trained on the training kernel and evaluated on the test set. The same is done with the
classical RBF kernel with default gamma = 1/(D * Var(X_train)).

Usage (from the repository root):
    PYTHONPATH=. python experiments/evaluation/Classification_multimodal_eval.py \
        --config configuration_files/config_classification_multimodal_eval.json

Output: <Output_dir>/classification_multimodal_results.csv, one row per (kernel configuration, K_type).
'''

import glob
import pandas as pd
import scipy.sparse as sparse
import argparse
import json
import os

from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score

from src.utils import Compute_rbf_kernel, compute_gd, Kernel_concentration


ap=argparse.ArgumentParser()
ap.add_argument('-config','--config_file',
                default='configuration_files/config_classification_multimodal_eval.json',
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

# <Kernel_dir>/Sample_<n>/<C>_clusters/<D>_qubits/IQP_<ent>_reps_<r>/qk_tr_<b>.npz
for i in sorted(glob.glob(os.path.join(kernel_dir, '*', '*', '*', '*', 'qk_tr_*.npz'))):
    path = os.path.normpath(i).split(os.sep)
    sampling = int(path[-5].split('_')[1])
    n_clusters = int(path[-4].split('_')[0])
    n_qubits = int(path[-3].split('_')[0])
    ent_type = path[-2].split('_')[1]
    reps = int(path[-2].split('_')[3])
    bandwidth = float(path[-1].split('_')[2].split('.npz')[0])
    print('Sampling:',sampling, 'N_clusters:',n_clusters, 'N_qubits:',n_qubits,
          'Ent_type:',ent_type, 'Reps:',reps, 'Bandwidth:',bandwidth, flush=True)

    #load data train and test
    data_dir = os.path.dirname(os.path.dirname(i))
    data_tr=pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    data_te=pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    y_train=data_tr['label']
    y_test=data_te['label']

    #Compute Classical Kernel
    c_kernel_tr=Compute_rbf_kernel(data_tr.iloc[:,:-1],data_tr.iloc[:,:-1])
    c_kernel_ts=Compute_rbf_kernel(data_te.iloc[:,:-1],data_tr.iloc[:,:-1])
    c_conc=Kernel_concentration(c_kernel_tr)

    #Load quantum kernels: train (n_train x n_train) and train-test (n_train x n_test)
    q_kernel_tr=sparse.load_npz(i).toarray()
    q_kernel_ts=sparse.load_npz(os.path.join(os.path.dirname(i), path[-1].replace('qk_tr_', 'qk_ts_'))).toarray()
    qk_conc=Kernel_concentration(q_kernel_tr)

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
    results.append( {
        "K_type":"Quantum",
        "Sampling":sampling,
        "N_clusters":n_clusters,
        "N_qubits":n_qubits,
        "Ent_type":ent_type,
        "Reps":reps,
        "Bandwidth":bandwidth,
        "K_concentration":qk_conc,
        "GD":gd,
        'acc_tr':accuracy_score(y_train,y_pred_q_tr),
        'acc_ts':accuracy_score(y_test,y_pred_q),
        'f1_tr':f1_score(y_train,y_pred_q_tr,average='weighted'),
        'f1_ts':f1_score(y_test,y_pred_q,average='weighted'),
        'prec_tr':precision_score(y_train,y_pred_q_tr,average='weighted'),
        'prec_ts':precision_score(y_test,y_pred_q,average='weighted'),
        })

    results.append( {
        "K_type":"Classical",
        "Sampling":sampling,
        "N_clusters":n_clusters,
        "N_qubits":n_qubits,
        "Ent_type":None,
        "Reps":None,
        "Bandwidth":None,
        "K_concentration":c_conc,
        "GD":gd,
        'acc_tr':accuracy_score(y_train,y_pred_c_tr),
        'acc_ts':accuracy_score(y_test,y_pred_c),
        'f1_tr':f1_score(y_train,y_pred_c_tr,average='weighted'),
        'f1_ts':f1_score(y_test,y_pred_c,average='weighted'),
        'prec_tr':precision_score(y_train,y_pred_c_tr,average='weighted'),
        'prec_ts':precision_score(y_test,y_pred_c,average='weighted'),
        })

#Save results
pd.DataFrame(results).to_csv(os.path.join(out_dir,'classification_multimodal_results.csv'),index=False)
