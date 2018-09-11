from sklearn import metrics
import numpy

def compute_ap_my(label,pred):

    if label.sum() == 0:
        print 'No positives found, retuning None'
        return None
    else:
        sids = numpy.argsort(pred)[::-1]
        pos = 0.0
        neg = 0.0
        prec = 0.0
        
        for ix in sids:
            if label[ix] == 1:
                pos = pos + 1
                prec = prec + (pos/(pos+neg))
            else:
                neg = neg + 1
    
        prec = prec/pos
        return prec

def compute_roc_auc(label,pred):
    
    return metrics.roc_auc_score(label,pred)

def compute_ap(label,pred):
    return metrics.average_precision_score(label,pred)


def compute_AP_all_class(all_labels,all_predictions):

    allaps = []
    for i in range(all_labels.shape[1]):
        cap = compute_ap(all_labels[:,i],all_predictions[:,i])
        #print cap
        allaps.append(cap)

    allaps.append(sum(allaps)/len(allaps))
    return allaps


def compute_AP_my_all_class(all_labels,all_predictions):

    allaps = []
    for i in range(all_labels.shape[1]):
        cap = compute_ap_my(all_labels[:,i],all_predictions[:,i])
	#print all_labels[:,i]
	#print all_predictions[:,i]
        #print cap
        allaps.append(cap)

    allaps.append(sum(allaps)/len(allaps))
    return allaps


def compute_AUC_all_class(all_labels,all_predictions):

    allroc = []
    for i in range(all_labels.shape[1]):
        cauc = compute_roc_auc(all_labels[:,i], all_predictions[:,i])
        allroc.append(cauc)

    allroc.append(sum(allroc)/len(allroc))
    return allroc
    
    
