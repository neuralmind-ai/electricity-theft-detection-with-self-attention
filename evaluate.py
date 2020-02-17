import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def get_logits_and_trues_and_loss(model, dataloader, loss_fn=None, device='cpu'):
    loss = 0.0
    model.eval()
    logits, trues = [], []
    losses = []
    for x,y in dataloader:
        with torch.no_grad():
            y_pred = model(x.to(device))

        if loss_fn:
            loss = loss_fn(y_pred, y.to(device)).item()
            losses.append(loss)

        logits.extend(y_pred.to('cpu').numpy().tolist())
        trues.extend(y.to('cpu').numpy().tolist())

    if len(losses):
        loss = np.array(losses).mean()

    return np.array(logits), np.array(trues), loss

def evaluate_fn(model,dataloader,loss_fn,device,verbose=False):
    tns,fps,fns,tps,valid_losses = [],[],[],[],[]
    conf_matrix_final = 0

    logits, trues, loss = get_logits_and_trues_and_loss(model, dataloader, loss_fn, device=device)
    conf_matrix = confusion_matrix(trues, logits.argmax(1))

    try:
        tp = conf_matrix[1][1]
    except:
        tp = 0
    try:
        tn = conf_matrix[0][0]
    except:
        t=0
    try:
        fp = conf_matrix[0][1]
    except:
        fp=0
    try:
        fn = conf_matrix[1][0]
    except:
        fn = 0
        
    if (tp != 0 or fp != 0): 
         precision = tp/(tp+fp)
    else:
        precision = 0.0
    
    if(tp != 0 or fn != 0):
        recall = tp/(tp+fn)
    else:
        recall = 0.0
    
    if (precision != 0.0 or recall != 0):
        F1Score = 2*precision*recall/(precision+recall)
    else:
        F1Score = 0.0

    if (verbose):
      #print(conf_matrix)
#       print('AUC: {:.3}'.format(roc_auc_score(trues,logits)), flush=True)
#       print('Precision: {:.3}'.format(precision), flush=True)
#       print('Recall: {:.3}'.format(recall), flush=True)
       pass 
    return loss,tn,fp,fn,tp,precision,recall,F1Score
