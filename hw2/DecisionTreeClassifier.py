"""
Author: TING-HUNG LIN
Git: https://github.com/nilpilihp/cs760.git

This base structure of Node is adapted from deepandas11's implmentation on git
"""
import numpy as np
import pandas as pd
import math
from pprint import pprint
from collections import OrderedDict


def load_data(filename='D1.txt'):
    df = pd.read_csv(filename, sep=" ", header=None)
    df.columns = ['x1', 'x2', 'Outcome']
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


class Node():
    def __init__(self, attribute, threshold, left_nums=0, right_nums=0, leaf_nums=0, igr=0):
        # feature dimen j
        self.attr = attribute
        # threshold c, on training point
        self.thres = threshold

        self.left = None
        self.right = None

        # Demographic data at that node
        self.left_nums = 0
        self.right_nums = 0
        self.leaf_nums = 0

        # Is it a leaf
        self.leaf = False
        # Label assigned to a node if leaf
        self.predict = None
        self.igr = igr


class DecisionTree():

    def __init__(
        self,
        data,
        cols,
        predict_attr='Outcome',
    ):
        self.df = data
        self.test_df = None
        self.attributes = cols
        self.predict_attr = predict_attr
        # Abstract object built
        self.final_tree = None
        # Object printed
        self.repr_tree = []


    def _build_tree(self, df):
        # Get the number of positive and negative examples in the training data
        p, n = self._num_class(df)
        # Train data has one kind of value remaining
        if p == 0 or n == 0:
            # Create a leaf node indicating it's prediction
            leaf = Node(None,None)
            leaf.leaf = True
            #tie break with class 1
            if p >= n:
                leaf.predict = 1
            else:
                leaf.predict = 0
            leaf.leaf_nums = p+n
            return leaf
        else:
            # Otherwise, find the attribute for deciding with threshold
            best_attr, threshold, igr = self._select_attr(df)
            # Create internal tree node based on attribute and it's threshold
            tree = Node(best_attr, threshold, igr=igr)

            sub_1 = df[df[best_attr] >= threshold]
            sub_2 = df[df[best_attr] < threshold]
            
            # Recursively build left and right subtree
            tree.left_nums = sub_1.shape[0]
            tree.right_nums = sub_2.shape[0]

            tree.left = self._build_tree(sub_1)
            tree.right = self._build_tree(sub_2)
            return tree

    def _num_class(self, df):
        p_df = df[df[self.predict_attr] == 1]
        n_df = df[df[self.predict_attr] == 0]

        return p_df.shape[0], n_df.shape[0]

    def _select_attr(self, df):
        """
        Find the attribute that maximize information gain ratio

        :param df: Data
        """
        max_igr = float("-inf")
        best_attr = None
        threshold = 0

        # Check across attributes
        for attribute in self.attributes:
            thres = self._select_threshold(df, attribute)
            # one extra call to get the best igr from _select_threshold 
            igr = self._info_gain_ratio(df, attribute, thres)
            if igr > max_igr:
                max_igr = igr
                best_attr = attribute
                threshold = thres
        
        return best_attr, threshold, max_igr

    def _select_threshold(self, df, attribute):
        """
        Find value in a column or attribute that maximizes Gain
        no threshold in between value

        :param df: data
        :param attribute: Column Name for the attribute of interest
        """

        values = df[attribute].tolist()
        values = [float(x) for x in values]

        max_igr = float("-inf")
        thres_val = 0

        # Check across all values of an attribute and find IGainRatio
        for i in range(0, len(values)):
            thres = values[i]
            igr = self._info_gain_ratio(df, attribute, thres)
            if igr >= max_igr:
                max_igr = igr
                thres_val = thres

        return thres_val

    def _info_gain(self, df, attribute, threshold):
        """
        [ARCHIVE]
        Return Information Gain caused by splitting based on threshold
        

        :param df: data
        :param attribute: column chosen to be used for decision making
        :param threshold: value in the attribute that is used as threshold
        """
        sub1 = df[df[attribute] < threshold]
        sub2 = df[df[attribute] >= threshold]

        ig = self._info_entropy(df) - self._remainder(df, [sub1, sub2])
        return ig
    
    def _info_gain_ratio(self, df, attribute, threshold):
        """
        Return Information Gain Ratio caused by splitting based on threshold
        
        IGR = (H(Y)-H(Y|S))/H{S}

        :param df: data
        :param attribute: column chosen to be used for decision making
        :param threshold: value in the attribute that is used as threshold
        """
        sub1 = df[df[attribute] < threshold]
        sub2 = df[df[attribute] >= threshold]
        
        proba = [float(sub1.shape[0]), float(sub2.shape[0])]
        proba[0] /= sum(proba)
        proba[1] /= sum(proba)

        # leaf node
        if proba[0] == 0 or proba[1] == 0:
            igr =  0
        else:
            igr = (self._info_entropy(df) - self._remainder(df, [sub1, sub2]))/ (-1 * np.dot(proba, np.log2(proba)))
       
        # for Q3 print out root candidates
        # print(f'attr: {attribute}, c: {threshold}, igr:{igr:.5f}')
        return igr
    

    def _remainder(self, df, df_list):
        """
        Return weighted sum of entropies of the splits

        :param df: unsplitted dataset
        :param df_list: list of splits
        """
        num_data = df.shape[0]
        remainder = float(0)
        for df_sub in df_list:
            if df_sub.shape[0] > 1:
                to_add = float(df_sub.shape[0]/num_data) * self._info_entropy(df_sub)
                remainder += to_add

        return remainder

    def _info_entropy(self, df):
        """
        Return Entropy inside a dataframe
        
        :param df: Dataframe
        """
        pos_df = df[df[self.predict_attr] == 1]
        neg_df = df[df[self.predict_attr] == 0]

        proba = [float(pos_df.shape[0]), float(neg_df.shape[0])]
        proba[0] /= sum(proba)
        proba[1] /= sum(proba)

        if proba[0] == 0 or proba[1] == 0:
            I = 0
        else:
            I = -1 * np.dot(proba, np.log2(proba))
        
        return I

    def _create_view(self, root, level):
        if root.leaf:
            elem = OrderedDict({
                "Label": root.predict,
                "Instances": root.leaf_nums,
                "Level": level,
            })
            self.repr_tree.append(elem)
        
        else:
            elem = OrderedDict({
                "Level": level,
                "Predecessor": root.left_nums + root.right_nums,
                "Node": str(root.attr) + " >= " + str(root.thres) + ";  Gain = "+str(root.igr),
                "Split": [root.left_nums, root.right_nums],
            })
            self.repr_tree.append(elem)
        if root.left:
            self._create_view(root.left, level+1)
        if root.right:
            self._create_view(root.right, level+1)


    def _represent_tree(self, tree):
        max_level = tree[-1]['Level']
        level = 0
        i = 0
        node_i = 0
        print("\n","*"*30,"\n")
        print("Level ", level,"\n")
        while level <= max_level and i < len(tree):
            if tree[i]['Level'] > level:
                if 'Node' in tree[i]:
                    node_i += 1
                print("*"*30,"\n")
                level += 1
                print("Level ", level, "\n")

                pprint(dict(tree[i]))

            else:
                if 'Node' in tree[i]:
                    node_i += 1
                pprint(dict(tree[i]))
        
            i += 1

        print("Total Nodes: ", node_i)

    
    def __call__(self):
        self.final_tree = self._build_tree(df=self.df)
        self._create_view(self.final_tree, 0)

        repr_tree = sorted(self.repr_tree, key=lambda k:k['Level'])
        self._represent_tree(repr_tree)

    
    def _single_predict(self, node, row_df):
        """
        Predict for a single row from a dataframe.
        Uses the trained tree 
        """
        # If we are at a leaf node, return the prediction of the leaf node
        if node.leaf:
            return node.predict
        # Traverse left or right subtree based on instance's data
        if row_df[node.attr] >= node.thres:
            return self._single_predict(node.left, row_df)
        elif row_df[node.attr] < node.thres:
            return self._single_predict(node.right, row_df)

    def predict(self, test_data):
        """
        predict label, return yhat in np.array

        :param test_data: DataFrame
        """
        yhat = np.zeros(len(test_data))
        root = self.final_tree
        for i,row in test_data.iterrows():
            prediction = self._single_predict(root, row)
            yhat[i] = prediction
        return yhat

    def train_error(self, train_data):
        """
        training metric, return accuracy and error

        :param train_data: DataFrame
        """
        root = self.final_tree
        num_data = train_data.shape[0]
        num_correct = 0
        for i,row in train_data.iterrows():
            prediction = self._single_predict(root, row)
            if prediction == row['Outcome']:
                num_correct += 1
        metrics = {
            "Accuracy": round(num_correct/num_data, 2),
            "Error": 1 - round(num_correct/num_data, 2),
        }
        return metrics