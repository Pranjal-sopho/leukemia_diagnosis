#! python3
"""
Created on Tue Aug 15 12:57:39 2017

@author: Rex
"""

import numpy as np
from SVM_train import SMO_train

class SVM_RFE:
    
    def __init__(self, r=[], s=[], c=[]):
        self.feature_rank_list_r = r
        self.surviving_features_s = s
        self.ranking_criteria = c
    
    # end def
    
    def svm_feature_ranking(self, X0, Y):
        n = X0.shape[0]
        self.surviving_features_s = np.arange(0, n)
        
        # intializing model SMO_train  
        # creating an object {named SMO_model} of class SMO_train present in module named SVM_train.py
        SMO_model = SMO_train()
        
        while self.surviving_features_s.size > 0 :
            
            # taking only those genes that are still surviving in the algorihtm
            X = X0[:, self.surviving_features_s]
            
            # training SVM using SMO algorithm
            (alpha, w) = SMO_model.train(X, Y)
            
            # calculating ranking criteria
            c = w  ** 2
            
            # finding index of feature with smallest ranking criterion
            f = np.argmin(c)
            
            # updating feature_rank_list
            self.feature_rank_list_r = np.append(self.feature_rank_list_r, self.surviving_features_s[f])
            
            # removing feature with minimum ranking criteria
            self.surviving_features_s = np.delete(self.surviving_features_s, f)
            
        # end while
        
        return self.feature_rank_list_r
    
    # end def

# end class