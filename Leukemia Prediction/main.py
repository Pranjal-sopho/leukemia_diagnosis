#! python3

"""
Created on Tue Aug 15 19:32:57 2017

@author: Rex
"""

import os, csv, sys
import numpy as np

from SVM_RFE import SVM_RFE
#from SVM_train import SMO_train

filepath = os.path.dirname(os.path.abspath(__file__))
parentpath = os.path.dirname(filepath)

# defining function for loading data
def readData(filename):
    data = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    
    return np.array(data)
#end def

def main(filename=parentpath + 'Datasets\Training Set\Leukemia_train.csv', C=100.0, epsilon=0.0001):
    
    # loading data
    data = readData('%s%s' % (filepath, filename))
    
    data = data.astype(int)
    
    svm = SVM_RFE()
    
    # splitting data
    X, Y = data[:, 0:-1], data[:, -1].astype(int)
    
    rank_list = svm.svm_feature_ranking(X, Y)
    
    # rank_list contains features(genes) in decreasing order of significance
    
    return
# end def

if __name__ == '__main__' :
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("\nTrains a Support Vector Machine and also displays the accuracy and other relevant data.")
        print("\nUsage: %s FILENAME C epsilon" % (sys.argv[0]))
        print("\n\nwhere\n\nFILENAME: Relative path of data file.")
        print("\nC: Value of variable C.")
        print("\nepsilon: Value of variable epsilon (Convergence Value).\n")
    else :
        kwargs = {}
        
        if len(sys.argv) > 1 :
            kwargs['filename'] = sys.argv[1]
        
        if len(sys.argv) > 2 :
            kwargs['C'] = sys.argv[2]
            
        if len(sys.argv) > 3 :
            kwargs['epsilon'] = float(sys.argv[3])
        
        if len(sys.argv) > 4 :
            sys.exit("\nIncorrect usage of arguments. Use %s -h or %s --help for more information\n" % (sys.argv[0]))
            
        main(**kwargs)
    # end if-else
#end if