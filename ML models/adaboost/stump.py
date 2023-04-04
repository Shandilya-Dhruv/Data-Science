import numpy as np
from statistics import mode


class leaf:
    def __init__(self, classif = None):
        """
        desc : defines the structure of leaf.

        classif : (int) most occuring classification in our leaf.

        returns : (None)
        """
        self.classif = classif
        
    
  
class stump:
    
    def __init__(self, left=None, right=None, threshold=None, feature=None, stump_weight=None):
        """
        desc : defines the structure of our stump.

        left : (leaf) left connected leaf.
        right : (leaf) right connected leaf.
        threshold : (float) threshold on which we make the decision.
        feature : (int) index of the column on which we decide.
        stump_weight : (float) weight assigned to the stump.

        returns : (None)
        """
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.stump_weight = stump_weight
    
    def eval_stump_weight(self, X, Y, weights, criteria, threshold, feature, l_res, r_res):
        """
        desc : evaluates the weight assigned to a stump.

        X : (numpy) dataset we have to classify without the target variable.
        Y : (list) target variable.
        weights : (list) weight assigned to each datapoint.
        criteria : (string) from : ['TotalError']. specifies criteria for evaluating stump weight.
        threshold : (float) threshold on which we make the stump.
        feature : (int) index of the column on which we decide.
        l_res : (float/int) result of the left leaf
        r_res : (float/int) result of the right leaf

        returns : (Tuple(int,list)) tuple having the stump_weight and bool list having true at misclassified indices.
        """
        if criteria=='TotalError':
            # Evaluate the total error which is sum of misclassified weights
            totErr = 1e-10
            bool_mis = []
            for i in range(X.shape[0]):
                # pred is the classification prediction by our stump
                pred = 0
                if X[i][feature] <= threshold:
                    pred = l_res
                else:
                    pred = r_res
                    
                # Check if misclassified
                if Y[i]!=pred:
                    totErr += weights[i]
                    bool_mis.append(True)
                else:
                    bool_mis.append(False)
                    
            if totErr == 1 :
                totErr -= 1e-10
                
            stump_weight = 0.5 * (np.log((1-totErr)/totErr))
            
            return (stump_weight, bool_mis)
                
    
    def best_stump(self, X, Y, weights, types):
        """
        desc : finds the best stump.

        X : (numpy) dataset we have to classify without the target variable.
        Y : (list) target variable.
        weights : (list) weight assigned to each datapoint.
        type : (string) from : ['classif','regression']. specifies the type of problem.
        
        return : (list) the list of updated weights after finding the best stump.
        """
        if types == 'classif':
            
            # Evaluate the best stump on the basis of criteria
            max_weight = float("-inf")
            for i in range(X.shape[0]): 
                for j in range(X.shape[1]):
                    
                    l = []
                    r = []
                    
                    for k in range(X.shape[0]):
                        if X[k][j] <= X[i][j]:
                            l.append(Y[i])
                        else:
                            r.append(Y[i])
                    
                    if len(r)==0:
                        continue
                    
                    l_classif = mode(l)
                    r_classif = mode(r)
                    
                    res = self.eval_stump_weight(X,Y,weights,'TotalError',X[i][j],j,l_classif,r_classif)
                    stump_weight = res[0]
                    if stump_weight > max_weight:
                        max_weight = stump_weight
                        threshold = X[i][j]
                        feature = j
                        l_res = l_classif
                        r_res = r_classif
                        bool_mis = res[1]
            
            self.threshold = threshold
            self.feature = feature
            self.left = leaf(l_res)
            self.right = leaf(r_res)
            self.stump_weight = max_weight
            
            # Re-evaluating the weight of each datapoint
            for i in range(len(weights)):
                if bool_mis[i] == True:
                    weights[i] = weights[i] * np.exp(max_weight)
                else:
                    weights[i] = weights[i] * np.exp(-1*max_weight)
            
            # normalisinng the updated weights
            s = sum(weights)
            weights[:] = [x/s for x in weights]
            
            return weights