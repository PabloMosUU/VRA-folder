# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:37:05 2021

@author: 20200016
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:37:05 2021

@author: 20200016
"""
import pandas as pd
import os
import numpy as np
import sys
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import auc

from statistics import stdev, mean

DIR = 'C:/Users/20200016/surfdrive/PhD/1. Onderzoek/3. Violence Risk Assessment/JAIMS/Data_predictions'

os.chdir(DIR)

#
# Loading data
#

bvc = pd.read_csv('1.bvc_predictions_truths.csv')
tfidf = pd.read_csv('2.tfidf_1grams_predictions.csv')
svm = pd.read_csv('3.Stemming+doc2vec+SVM.csv')
rf1 = pd.read_csv('4.Stemming+doc2vec+randomforest.csv')
rf2 = pd.read_csv('5.Stemming+doc2vec+struct+RF.csv')
rf3 = pd.read_csv('6.Stemming+LDA+struct+RF.csv')
rf4 = pd.read_csv('7.Stemming+doc2vec+LDA+struct+RF.csv')

#
# Initializing meta variables
#

def create_variables(probabilities, labels, integral = 'trapezoid'):

    model = Metrics(probabilities, labels, integral = integral)
    #prec_curve = model.precision_curve()
    #kappa_curve = model.kappa_curve()
    #tpr_curve = model.tpr_curve()
    #fpr_curve = model.fpr_curve()
    #pr_auc = auc(tpr_curve, prec_curve)
    pr_auc = model.calc_pr_auc()
    roc_auc = model.calc_roc_auc()
    auk = model.calc_auk()
    return [roc_auc,pr_auc, auk]

#
# Preprocess SVM
#

def replace_labels(label):
    if str(label) == 'True':
        return 1
    elif str(label) == 'False':
        return 0

svm['true_label'] = svm.true_label.apply(replace_labels)

#
# Prepare BERT metrics
#

index = list(range(0,20))
columns = ['fold', 'shortening_strategy', 'epoch', 'y_true', 'y_prob', 'name']
bert_raw = pd.DataFrame(columns = columns, index = index)

i = 0
for file in os.listdir(DIR):
    if file.startswith('test_prediction'):
        temp = pd.read_csv(file)
        bert_raw.iloc[i,0] = int(file.split('_')[2])
        bert_raw.iloc[i,1] = str(file.split('_')[3])
        bert_raw.iloc[i,2] = int(file.split('_')[4][0])
        bert_raw.iloc[i,3] = list(temp['y_true'])
        bert_raw.iloc[i,4] = list(temp['y_pred_proba'])
        bert_raw.iloc[i,5] = str(file)        
        i+=1        
    else:
        continue
    
#
# Build metrics matrix
#

rows = ['ROC-AUC','PR-AUC','AUK']

colnames = ['BVC_begin', 'BVC_end',
        'tfidf1','tfidf2','tfidf3','tfidf4','tfidf5',
         'svm1', 'svm2', 'svm3', 'svm4', 'svm5',
         'rf1_1', 'rf1_2', 'rf1_3', 'rf1_4', 'rf1_5', 
         'rf2_1', 'rf2_2', 'rf2_3', 'rf2_4', 'rf2_5', 
         'rf3_1', 'rf3_2', 'rf3_3', 'rf3_4', 'rf3_5', 
         'rf4_1', 'rf4_2', 'rf4_3', 'rf4_4', 'rf4_5',
         'bert_sum_1_1','bert_sum_1_2','bert_sum_1_3','bert_sum_1_4','bert_sum_1_5',
         'bert_trunc_1_1','bert_trunc_1_2','bert_trunc_1_3','bert_trunc_1_4','bert_trunc_1_5',
         'bert_sum_2_1','bert_sum_2_2','bert_sum_2_3','bert_sum_2_4','bert_sum_2_5',
         'bert_trunc_2_1','bert_trunc_2_2','bert_trunc_2_3','bert_trunc_2_4','bert_trunc_2_5'
         ]

metrics = pd.DataFrame(columns = colnames, index = rows)

#BVC
bvc_labels_end = bvc.iloc[:,1]
bvc_labels_begin = bvc.iloc[:,2]
bvc_probs = bvc.iloc[:,3]

bvc_begin = create_variables(bvc_probs, bvc_labels_begin, integral='min')
bvc_end = create_variables(bvc_probs, bvc_labels_end, integral = 'min')

rows = [0,1,2]

for r in rows:
    metrics.iloc[r,0] = bvc_begin[r]
    metrics.iloc[r,1] = bvc_end[r]

models = ['TFIDF','SVM','RF1','RF2','RF3','RF4']
folds = [0,1,2,3,4]

for i, model in enumerate(models):
    for f in folds:
        tfidf_prob = tfidf.probability[tfidf.fold_number == f+1]
        tfidf_label = list(tfidf.true_label[tfidf.fold_number == f+1])
        tf_idf_metrics = create_variables(tfidf_prob, tfidf_label)
    
        svm_prob = svm.probability[svm.fold_number == f+1]
        svm_label = list(svm.true_label[svm.fold_number == f+1])
        svm_metrics = create_variables(svm_prob, svm_label)
    
        rf1_prob = rf1.probability[rf1.fold_number == f+1]
        rf1_label = list(rf1.true_label[rf1.fold_number == f+1])
        rf1_metrics = create_variables(rf1_prob, rf1_label)
        
        rf2_prob = rf2.probability[rf2.fold_number == f+1]
        rf2_label = list(rf2.true_label[rf2.fold_number == f+1])
        rf2_metrics = create_variables(rf2_prob, rf2_label)
        
        rf3_prob = rf3.probability[rf3.fold_number == f+1]
        rf3_label = list(rf3.true_label[rf3.fold_number == f+1])
        rf3_metrics = create_variables(rf3_prob, rf3_label)
        
        rf4_prob = rf4.probability[rf4.fold_number == f+1]
        rf4_label = list(rf4.true_label[rf4.fold_number == f+1])
        rf4_metrics = create_variables(rf4_prob, rf4_label)
    
        print('Model: ',str(model),', Fold: ',str(f))
    
        for r in rows:
            metrics.iloc[r,f+2] = tf_idf_metrics[r]
            metrics.iloc[r,f+7] = svm_metrics[r]
            metrics.iloc[r,f+12] = rf1_metrics[r]
            metrics.iloc[r,f+17] = rf2_metrics[r]
            metrics.iloc[r,f+22] = rf3_metrics[r]
            metrics.iloc[r,f+27] = rf4_metrics[r]
    
epochs = [1,2]
short_strat = ['summarize','truncate']

i = 0

bert_index = metrics.columns.get_loc("bert_sum_1_1")

for e in epochs:
    for s in short_strat:
        for f in folds:
            #print(list(bert_raw.y_prob[bert_raw.fold == f][bert_raw.epoch == e][bert_raw.shortening_strategy == s])[0])
            
            prob = list(bert_raw.y_prob[bert_raw.fold == f][bert_raw.epoch == e][bert_raw.shortening_strategy == s])[0]
            label = list(bert_raw.y_true[bert_raw.fold == f][bert_raw.epoch == e][bert_raw.shortening_strategy == s])[0]

            eval_metrics = create_variables(prob, label)
            #print('bert_',s,'_',str(e),'_',str(f))
            for r in [0,1,2]:
                metrics.iloc[r,bert_index+i] = eval_metrics[r]
                
            i+=1 

models = ['TFIDF','SVM','RF1','RF2','RF3','RF4']
folds = [0,1,2,3,4]
averages = {}
stdevs = {}

for i,model in enumerate(models):
    roc_auc = []
    pr_auc = []
    auk = []
    
    for f in folds:
        col_index = (i*len(folds))+2+f
        #roc_auc.append(metrics.iloc[0,(i*5)+2+f])   
        roc_auc.append(metrics.iloc[0,col_index])
        pr_auc.append(metrics.iloc[1,col_index])        
        auk.append(metrics.iloc[2,col_index])
                 
    avg_roc = mean(roc_auc)
    avg_pr = mean(pr_auc)
    avg_uk = mean(auk)
        
    avg = [avg_roc, avg_pr, avg_uk]
        
    stdev_roc = stdev(roc_auc)
    stdev_pr = stdev(pr_auc)
    stdev_auk = stdev(auk)
       
    st = [stdev_roc, stdev_pr, stdev_auk]
        
    averages[model] = avg
    stdevs[model] = st
    
bert_models = ['bert_sum_1','bert_trunc_1','bert_sum_2','bert_trunc_2']

for i,model in enumerate(bert_models):
    roc_auc = []
    pr_auc = []
    auk = []
    print('Model',model)
    for f in folds:
        col_index = bert_index + (i*len(folds)) + f
        
        roc_auc.append(metrics.iloc[0,col_index])
        pr_auc.append(metrics.iloc[1,col_index])        
        auk.append(metrics.iloc[2,col_index])
    
    avg_roc = mean(roc_auc)
    avg_pr = mean(pr_auc)
    avg_uk = mean(auk)
        
    avg = [avg_roc, avg_pr, avg_uk]
        
    stdev_roc = stdev(roc_auc)
    stdev_pr = stdev(pr_auc)
    stdev_auk = stdev(auk)
       
    st = [stdev_roc, stdev_pr, stdev_auk]
        
    averages[model] = avg
    stdevs[model] = st
        
averages['SVM']    
stdevs['SVM']


#metrics.to_csv('All_metrics.csv', sep=',')

