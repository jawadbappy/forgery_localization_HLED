import numpy as np
import math

def compute_mcc(y_actual, y_hat):
    TP = 0; FP = 0;TN = 0; FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1


    return TP,FP,TN,FN

def metrics(TP,FP,TN,FN):
    a=TP+FP
    b=TP+FN
    c=TN+FP
    d=TN+FN
    mcc=((TP*TN)-(FP*FN))/(math.sqrt(float(a*b*c*d)+0.0001))
    F1=(2*TP)/float(2*TP+FP+FN+.0000001)
    precision=TP/float(TP+FP+.0000001)
    recall=TP/float(TP+FN+.0000001)
    return mcc,F1,precision,recall

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    #freq = hist.sum(axis=1) / hist.sum()

    return acc, acc_cls, mean_iu


a,b,c,d=compute_mcc([1,1,0,1,0,1,0],[1,0,1,0,1,0,1])



