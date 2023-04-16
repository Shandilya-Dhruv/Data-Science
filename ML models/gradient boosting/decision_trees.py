import numpy as np
import pandas as pd

class node:

    def __init__(self, left=None, right=None, threshold=None, feature=None, score=None, classif = None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.score = score
        self.classif = classif


class decision_tree:

    """
    Limitation : only designed to handle classification with 0 and 1 as classes.
    """

    def __init__(self,max_depth,min_sample_size):
        self.root = None

        self.max_depth = max_depth
        self.min_sample_size = min_sample_size

    def make_split(self, X, feature, threshold, Y):
        """ 
        desc : makes two lists from the data

        X : (numpy) numpy data to be split
        feature : (int) the feature number on which we will split the data
        threshold : (int) the threshold for splitting
        Y : (numpy) the series storing classification of each datapoint

        returns : (tuple of lists) returns two lists having the classifications of split data in list format
        """
        l = []
        g = []
        for i in range(X.shape[0]):
            if X[i][feature]<=threshold:
                l.append(Y[i])
            else:
                g.append(Y[i])
        
        return (l,g)

    def gini(self,l):
        """
        desc: calculates gini score

        l : (list) stores the classification of each datapoint

        returns : (tuple) return gini score(float) and the most occuring class in l
        """
        gin = 0
        p = 0
        r = (l.count(0)/len(l),l.count(1)/len(l))

        classes = np.unique(l)

        for i in classes:
            t = l.count(i)
            if len(l)==0:
                t = 0
            else:
                t = t/len(l)
            gin += t*(1-t)
            if t>p : 
                p = t
        
        return (-gin,r)
    
    def sum_square_error(self, l):
        """
        desc: calculates sum of squared error

        l : (list) stores the target variable of each datapoint

        returns : (tuple) return sum of squared error(float) and the average of that list
        """
        y_mean = sum(l)/len(l)
        diff_sq = [(i-y_mean)**2 for i in l]
        return sum(diff_sq,y_mean)

    def split_data(self,X,feature,threshold):
        """
        desc : splits the dataframe into two dataframes

        X : (numpy) numpy data to be split
        feature : (int) the feature number on which we will split the data
        threshold : (int) the threshold for splitting
        
        returns : (tuple) two split dataframes
        """

        l = np.zeros((1,X.shape[1]))
        g = np.zeros((1,X.shape[1]))

        for i in range(X.shape[0]):

            if X[i][feature] <= threshold:
                l = np.vstack([l, X[i]])
            else:
                g = np.vstack([g, X[i]])

        return (l[1:],g[1:])

    def best_split(self,X,Y,type):
        """
        desc : finds the best split for the data

        X : (numpy) numpy data for which we need the best split
        Y: (numpy) the series storing classification of each datapoint
        type : (string) from : ['Classification','Regression']. specifies the type of problem.

        returns : (tuple) returns the (feature(best split feature),threshold(best split feature),gini_score(data),classification(most occuring class of data)) of the best data
            return gin as -inf if no such split exists
        """
        if type == 'Classification':
            gin = float('-inf')
            f = ""
            thresh = float('-inf')

            for i in range(X.shape[1]):
                for j in range(X.shape[0]):
                    l,g = self.make_split(X,i,X[j][i],Y)
                    if len(l)<self.min_sample_size or len(g)<self.min_sample_size:
                        continue
                    num = (len(l)*self.gini(l)[0] + len(g)*self.gini(g)[0])/(len(l)+len(g))
                    if num>gin:
                        gin = num
                        f = i
                        thresh = X[j][i]

            t = self.gini(list(Y))
            if thresh == float('-inf') and f == "":
                return (f,thresh,t[0],-1)
            l,g = self.make_split(X,f,thresh,Y)
            t_l = self.gini(l)
            t_g = self.gini(g)
            return (f,thresh,t[0],(t_l[1],t_g[1]))
        
        else:
            sse = float('inf')
            f = ""
            thresh = float('-inf')

            for i in range(X.shape[1]):
                for j in range(X.shape[0]):
                    l,g = self.make_split(X,i,X[j][i],Y)
                    if len(l)<self.min_sample_size or len(g)<self.min_sample_size:
                        continue
                    num = self.sum_square_error(l)[0] + self.sum_square_error(g)[0]
                    if num<sse:
                        sse = num
                        f = i
                        thresh = X[j][i]

            t = self.sum_square_error(list(Y))
            if thresh == float('-inf') and f == "":
                return (f,thresh,t[0],-1)
            l,g = self.make_split(X,f,thresh,Y)
            t_l = self.sum_square_error(l)
            t_g = self.sum_square_error(g)
            return (f,thresh,t[0],(t_l[1],t_g[1]))

    def build_tree(self,X,depth,Y,type):
        """
        desc : builds our decision tree

        X : (numpy) current numpy data of node
        depth : (int) the current depth of the tree
        Y: (numpy) the series storing classification of each datapoint
        type : (string) from : ['Classification','Regression']. specifies the type of problem.
        
        return : (no return type)
        """

        tup = self.best_split(X,Y,type)
        a = node(threshold=tup[1],feature=tup[0],score=tup[2],classif=tup[3])

        #if size constraint is violated
        if tup[1] == float('-inf') and tup[0] == "":
            return

        if depth == 0:
            self.root = a
            l,g = self.split_data(X,tup[0],tup[1])
            Y_l,Y_g = self.make_split(X,tup[0],tup[1],Y)
            self.root.left = self.build_tree(l,depth+1,Y_l)
            self.root.right = self.build_tree(g,depth+1,Y_g)
            return self.root
            
        elif depth<=self.max_depth and depth>0:
            n = a
            l,g = self.split_data(X,tup[0],tup[1])
            Y_l,Y_g = self.make_split(X,tup[0],tup[1],Y)
            n.left = self.build_tree(l,depth+1,Y_l)
            n.right = self.build_tree(g,depth+1,Y_g)
            return n

    def print_tree(self,n,depth):
        if n!=None :
            s = '   '*depth
            print(s,n.threshold,n.feature,n.classif)
            self.print_tree(n.left,depth+1)
            self.print_tree(n.right,depth+1)

    def predict_row(self,X):
        """
        desc : predicts the output for a single row of features from the dataset

        X : (numpy row) row of features

        return : (int) the appropriate classification
        """
        r = self.root
        l = True

        while l:
            if X[r.feature]<=r.threshold:
                if r.left == None:
                    l = True
                    break
                r = r.left
            else:
                if r.right == None:
                    l = False
                    break
                r = r.right
        
        c = (-1,-1)

        if l:
            c = r.classif[0]
        else:
            c = r.classif[1]

        if c[0]>=c[1]:
            return 0
        else:
            return 1


    def predict(self,X_test):
        """
        desc : predicts the output for a the test dataset

        X : (numpy) the test dataset

        return : (list) the appropriate classifications of each row
        """
        l = []
        for i in range(X_test.shape[0]):
            l.append(self.predict_row(X_test[i]))
        return l
