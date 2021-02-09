# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:19:48 2021

@author: 20200016
"""

class Metrics:
    def __init__(self, probabilities, labels):
        self.probabilities = probabilities
        self.labels = labels
        self.probabilities_set = sorted(list(set(probabilities)))
    
    #make predictions based on the threshold value and self.probabilities
    def make_predictions(self, threshold):
        predictions = []
        for prob in self.probabilities:
            if prob >= threshold:
                predictions.append(1)
            else: 
                predictions.append(0)
        return predictions
    
    #make list with kappa scores for each threshold
    def kappa_curve(self):
        kappa_list = []
        
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            k = self.calc_kappa(tp, tn, fp, fn)
            kappa_list.append(k)
        return kappa_list
    
    #make list with fpr scores for each threshold
    def fpr_curve(self):
        fpr_list = []

        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            fpr = self.calc_fpr(fp, tn)
            fpr_list.append(fpr)
        return fpr_list
    
    def tpr_curve(self):
        tpr_list = []
                
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, _, _, fn = self.confusion_matrix(preds)
            tpr = self.calc_tpr(tp, fn)
            tpr_list.append(tpr)
        return tpr_list

    #make list with precision scores for each threshold
    def precision_curve(self):
        precision_list = []
        
        for thres in self.probabilities_set:
            preds = self.make_predictions(thres)
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision = self.calc_precision(tp, fp)
            precision_list.append(precision)
        return precision_list

    #calculate confusion matrix
    def confusion_matrix(self, predictions):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, pred in enumerate(predictions):
            if pred == self.labels[i]:
                if pred == 1:
                    tp += 1
                else: 
                    tn += 1
            elif pred == 1:
                fp += 1
            else: fn += 1
            tot = tp + tn + fp + fn
        return tp/tot, tn/tot, fp/tot, fn/tot
        
    
    
    #Calculate AUK
    def calc_auk(self):
        auk=0
        fpr_list = self.fpr_curve()
        
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp1 = self.calc_kappa(tp, tn, fp, fn)
            
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp2 = self.calc_kappa(tp, tn, fp, fn)
            
            y_dist = abs(kapp2-kapp1)
            top = (y_dist * x_dist)/2
            bottom = min(kapp1, kapp2)*x_dist
            auk += top + bottom
        return auk
       
       
    #Calculate roc-auc
    def calc_roc_auc(self):
        roc_auc = 0
        fpr_list = self.fpr_curve()
        
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            tpr1 = self.calc_tpr(tp, fn)
            
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            tpr2 = self.calc_tpr(tp, fn)

            y_dist = abs(tpr2-tpr1) 
            top = (y_dist * x_dist)/2
            bottom = x_dist * min(tpr1, tpr2)
            roc_auc += top + bottom
        return roc_auc
    
    def calc_pr_auc(self):
        pr_auc = 0
        tpr_list = self.tpr_curve()
        
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(tpr_list[i+1] - tpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision1 = self.calc_precision(tp, fp)
             
            preds = self.make_predictions(self.probabilities_set[i+1]) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision2 = self.calc_precision(tp, fp)

            y_dist = abs(precision2-precision1) 
            top = (y_dist * x_dist)/2
            bottom = x_dist * min(precision1, precision2)
            pr_auc += top + bottom
        
        
        #add begin area before smallest probability     
        preds = self.make_predictions(min(self.probabilities_set))
        tp, _, fp, _ = self.confusion_matrix(preds)
        precision = self.calc_precision(tp, fp)
        begin = (precision*min(tpr_list))/2
        pr_auc += begin
        
        #add end area after largest probability 
        preds=self.make_predictions(max(self.probabilities_set))
        tp, _, fp, _ = self.confusion_matrix(preds)
        precision = self.calc_precision(tp, fp)
        y_diff = 1-precision
        x_diff= 1-max(tpr_list)
        end_top = (y_diff)*(x_diff)/2
        end_bottom = precision * x_diff
        pr_auc += end_top + end_bottom
        return pr_auc
           
        
    def calc_fpr(self, fp, tn):
        return fp/(fp+tn)
    
    def calc_tpr(self, tp, fn): #same as recall
        return tp/(tp+fn)
    
    def calc_precision(self, tp, fp):
        return tp/(tp+fp)

    #Calculate kappa score
    def calc_kappa(self, tp, tn, fp, fn):
        acc = tp + tn
        p = tp + fn
        p_hat = tp + fp
        n = fp + tn
        n_hat = fn + tn
        p_c = p * p_hat + n * n_hat
        return (acc - p_c) / (1 - p_c)    
    