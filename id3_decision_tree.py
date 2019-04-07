# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:28:41 2018
"prinitn as XML"
@author: Karthik
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:38:48 2018

@author: Karthik
"""
import copy
import numpy as np
import pandas as pd
import math
import argparse as ag
import os


flag = 1
treeEntropy = 0


def check_location(loc_file):
    if not os.path.exists(loc_file):
        raise ag.ArgumentTypeError("{0} does not exist".format(loc_file))
    return loc_file


def calculateEntropy(target_col,class_log): 
    Entropy = 0
    value, sum1 = np.unique(target_col, return_counts=True)
    for i in range(len(value)):
        Entropy = Entropy-(float(sum1[i])/sum(sum1)) * math.log(float(sum1[i])/sum(sum1),class_log)
    return Entropy


def InformationGain(data, feature, class_log):
    global flag
    total_Entropy = calculateEntropy(data["target_class"], class_log)
    
    if flag == 1:
        treeEntropy = total_Entropy 
    flag = 0
    #print total_Entropy
    Entropy = 0
    value, sum1 = np.unique(data[feature], return_counts=True)
  #  print total_Entropy
   # print value,feature,sum1
    sum11 = sum(sum1)
    weightedEntropy = 0
    #print value,sum1
    Entropy = [0] * len(value)
    for i in range(len(value)): 
         Entropy[i] = calculateEntropy(data.where(data[feature] == value[i]).dropna()["target_class"], class_log)
         weightedEntropy = weightedEntropy + ((float(sum1[i])/sum11) * Entropy[i])
  #  print "weighted entropy",weightedEntropy
    InfoGain = total_Entropy - weightedEntropy     
    #print feature,attribute_entropy
    return InfoGain


def ID3(data, original_data, remaining_attributes, class_log):
    if len(data) == 0:
        return
    elif len(np.unique(data["target_class"])) == 1:
      #  print np.unique(data["target_class"])
        #print data
        #print remaining_attributes
        print(np.unique(data["target_class"])[0])
        return np.unique(data["target_class"])[0]
    elif len(remaining_attributes) == 0:
        value, count = np.unique(data["target_class"],return_counts = True)
        print(value[np.argmax(count)])
        return value[np.argmax(count)]
    
    else:
        attribute_values = [InformationGain(data,feature,class_log) for feature in remaining_attributes]
        best_feature = remaining_attributes[np.argmax(attribute_values)]
        tree = {best_feature:{}}
        #remaining_attributes.remove(best_feature)
        
        remaining_attributes = [i for i in remaining_attributes if i != best_feature]
        for value in data[best_feature].unique():
            split_data = data.where(data[best_feature] == value).dropna()
            entropy = calculateEntropy(split_data["target_class"], class_log)
            print("<node " + " entropy= \"" + str(entropy) + "\" feature=\"" + str(best_feature) + "\" value=\"" + str(value) + "\">")

            subtree = ID3(split_data, original_data, remaining_attributes, class_log)
            print("</node>")
            tree[best_feature][value] = subtree
    return tree


def main():
    parser = ag.ArgumentParser()
    parser.add_argument("--data", help="data filename", type=check_location)
    # parser.add_argument("--output", help="output file for final decision tree")
    args = parser.parse_args()
    file_name = args.data
    data = pd.read_csv(file_name, header=None)
    list = []
    for i in range(len(data.columns)-1):
      list.append("attr"+str(i))
    required_attribute = copy.deepcopy(list)
    list.append("target_class")
    data.columns = list
    class_log = float(len(data["target_class"].unique()))
    tree_entropy = calculateEntropy(data["target_class"], class_log)
    print("<tree " + "entropy = \"" + str(tree_entropy) + "\">")
    tree = ID3(data[:][:-1], data, required_attribute, class_log)
    print("</tree>")


if __name__ == "__main__":
    main()