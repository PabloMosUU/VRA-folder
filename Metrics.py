# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:19:48 2021

@author: 20200016
"""

class Metrics:
    def __init__(self, probabilities, labels):
        self.probabilities = probabilities
        self.labels = labels
        self.thresholds = sorted(list(set(probabilities)))
    
    #make predictions based on the threshold value and self.probabilities
    def make_predictions(self, threshold):
        return [int(prob>=threshold) for prob in self.probabilities]
    
    #make list with kappa scores for each threshold
    def kappa_curve(self):
        kappas = []        
        for thres in self.thresholds:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            k = self.calc_kappa(tp, tn, fp, fn)
            kappas.append(k)
        return kappas
    
    #make list with fpr scores for each threshold
    def fpr_curve(self):
        fpr_list = []

        for thres in self.thresholds:
            preds = self.make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            fpr = self.calc_fpr(fp, tn)
            fpr_list.append(fpr)
        return fpr_list
    
    def tpr_curve(self):
        tpr_list = []
                
        for thres in self.thresholds:
            preds = self.make_predictions(thres)
            tp, _, _, fn = self.confusion_matrix(preds)
            tpr = self.calc_tpr(tp, fn)
            tpr_list.append(tpr)
        return tpr_list

    #make list with precision scores for each threshold
    def precision_curve(self):
        precision_list = []
        
        for thres in self.thresholds:
            preds = self.make_predictions(thres)
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision = self.calc_precision(tp, fp)
            precision_list.append(precision)
        return precision_list

    #calculate confusion matrix
    # Returns: TP, TN, FP, FN
    def confusion_matrix(self, predictions):
        # Can replace for:
        # (after import sklearn.metrics as m)
        # tn, fp, fn, tp = m.confusion_matrix(self.labels, predictions).ravel()
        # return tp, tn, fp, fn
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
            else:
                fn += 1
            tot = tp + tn + fp + fn
            # Dividing by the total is not necessary in the confusion matrix
        return tp/tot, tn/tot, fp/tot, fn/tot
        
    
    
    #Calculate AUK
    def calc_auk(self):
        # You can replace this method by
        # (After import sklearn.metrics as m)
        # cohen_kappas = [m.cohen_kappa_score(self.labels,
        #                                     self.make_predictions(threshold))
        #                 for threshold in thresholds]
        # return m.auc(self.fpr_curve(), cohen_kappas)
        auk=0
        fpr_list = self.fpr_curve()
        
        for i, prob in enumerate(self.thresholds[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp1 = self.calc_kappa(tp, tn, fp, fn)
            # All the calculations for +1 will be re-done in next iteration
            # This is inefficient
            preds = self.make_predictions(self.thresholds[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp2 = self.calc_kappa(tp, tn, fp, fn)
            # There's not thing wrong with this, but I feel like you are
            # reinventing the wheel if you write your own integrator
            # The problem with this is that if you make a tiny mistake here,
            # it's hard to catch because you have to go through a lot of code
            # But if you re-use libraries, you know they have been tested
            y_dist = abs(kapp2-kapp1)
            top = (y_dist * x_dist)/2
            bottom = min(kapp1, kapp2)*x_dist
            auk += top + bottom
        return auk
       
       
    #Calculate roc-auc
    # It's not reasonable to write an integrator twice just because the x
    # and y axis are different. You should have a calc_auc method that takes
    # the x and y axes as parameters, and integrate there. Like this:
    # def calc_auc(self, x, y):
    #     auc = 0
    #     for i in range(len(x)):
    #         x_dist = abs(x[i+1] - x[i])
    #         y_1 = y[i]
    #         y_2 = y[i+1]
    #         y_dist = abs(y_2 - y_1)
    #         top = (y_dist * x_dist) / 2
    #         bottom = x_dist * min(y_1, y_2)
    #         auc += (top + bottom)
    #     return auc
    # In that case, the current method would be like this:
    # def calc_roc_auc(self):
    #     fpr_list = self.fpr_curve()
    #     tpr_list = self.tpr_curve()
    #     return self.calc_auc(fpr_list, tpr_list)
    # This does mean iterating over the thresholds twice, but because the
    # calc_auc method is fast, and because the thresholds are not that many,
    # it will be faster, and easier to read
    def calc_roc_auc(self):
        roc_auc = 0
        fpr_list = self.fpr_curve()
        
        for i, prob in enumerate(self.thresholds[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            tpr1 = self.calc_tpr(tp, fn)
            
            preds = self.make_predictions(self.thresholds[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            tpr2 = self.calc_tpr(tp, fn)

            y_dist = abs(tpr2-tpr1) 
            top = (y_dist * x_dist)/2
            bottom = x_dist * min(tpr1, tpr2)
            roc_auc += top + bottom
        return roc_auc

    # Same comment as I wrote for calc_pr_auc
    def calc_pr_auc(self):
        pr_auc = 0
        tpr_list = self.tpr_curve()
        
        for i, prob in enumerate(self.thresholds[:-1]):
            x_dist = abs(tpr_list[i+1] - tpr_list[i])
            
            preds = self.make_predictions(prob) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision1 = self.calc_precision(tp, fp)
             
            preds = self.make_predictions(self.thresholds[i+1]) 
            tp, _, fp, _ = self.confusion_matrix(preds)
            precision2 = self.calc_precision(tp, fp)

            y_dist = abs(precision2-precision1) 
            top = (y_dist * x_dist)/2
            bottom = x_dist * min(precision1, precision2)
            pr_auc += top + bottom
        
        
        #add begin area before smallest probability     
        # Danger: tpr_list and self.thresholds may not be ordered in the same
        # way, so taking the "min" of one and assuming it corresponds with the
        # "min" of the other is risky
        # A better solution is to rely on the fact that the thresholds are
        # sorted (you sorted them in the constructor). Thus, you can write:
        preds = self.make_predictions(self.thresholds[0])
        tp, _, fp, _ = self.confusion_matrix(preds)
        precision = self.calc_precision(tp, fp)
        begin = (precision*tpr_list[0])/2
        pr_auc += begin
        # Note also that this relies on assuming that Precision(0,0)=0
        # Do you understand why this is?
        
        #add end area after largest probability
        # Danger: same as above; solution: same, but using "-1" instead of "0"
        preds=self.make_predictions(self.thresholds[-1])
        tp, _, fp, _ = self.confusion_matrix(preds)
        precision = self.calc_precision(tp, fp)
        y_diff = 1-precision
        x_diff= 1-tpr_list[-1]
        end_top = (y_diff)*(x_diff)/2
        end_bottom = precision * x_diff
        pr_auc += end_top + end_bottom
        return pr_auc
        # As before, note that this assumes Precision(max_threshold)=1
        # Do you understand why this is?
           
        
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
    
