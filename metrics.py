import numpy as np
import torch

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from evaluate import get_logits_and_trues_and_loss

def precision_at_k(y_true,class_probs,k,threshold=0.5,class_of_interest=1,isSorted=False):
    if (not isSorted):
        coi_probs = class_probs[:,class_of_interest]
        sorted_coi_probs = np.sort(coi_probs)[::-1]
        sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    else:
        sorted_coi_probs = class_probs
        sorted_y = y_true

    sorted_coi_probs = sorted_coi_probs[:k]
    sorted_y = sorted_y[:k]
    sorted_predicted_classes = np.where(sorted_coi_probs>threshold,
                                        float(class_of_interest),
                                        0.0)
    precisionK = np.sum(sorted_predicted_classes == sorted_y)/k  
    return precisionK

def map_at_N(y_true,class_probs,N,thrs=0.5,class_of_interest=1):
    pks = []
    coi_probs = class_probs[:,class_of_interest]
    sorted_coi_probs = np.sort(coi_probs)[::-1]
    sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    sorted_coi_probs = sorted_coi_probs[:N]
    sorted_y = sorted_y[:N]

    top_coi_indexes = np.argwhere(sorted_y>0)

    for value in top_coi_indexes:
        limite = value[0] + 1
        pks.append(
                    precision_at_k(sorted_y[:limite],
                    sorted_coi_probs[:limite],
                    limite,threshold=thrs,isSorted=True)
                    )
    pks = np.array(pks)
    return pks.mean()

def metrics_report(model, dataloader, device='cpu'):
    logits, y_eval, _ = get_logits_and_trues_and_loss(model, dataloader, loss_fn=None, device=device)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    fraud_prob = probs[:,1]

    fpr, tpr, thresholds = roc_curve(y_eval,fraud_prob)
    print('AUC: {:.3} --'.format(roc_auc_score(y_eval,fraud_prob)), end='')
    print(' MAP@100: {:.3} --'.format(map_at_N(y_eval,probs,100)), end='')
    print(' MAP@200: {:.3}'.format(map_at_N(y_eval,probs,200)) )
