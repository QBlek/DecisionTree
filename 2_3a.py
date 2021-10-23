import pandas as pd
import numpy as np
import DecisionTree as dt
from pprint import pprint

if __name__ == '__main__':

    # Dataset setting
    dataset = pd.read_csv('./data/breast-cancer-wisconsin.data', header=None)
    indexName = dataset[dataset[6] == '?'].index
    dataset.drop(indexName, inplace=True)
    attributes = dataset[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    target_attribute = dataset[10]  # last attribute which has 2 or 4

    # dataset.columns = ['Sample Code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    train = dataset[:490]
    validation = dataset[490:560]
    test = dataset[560:]



    # decision tree
    dtm = dt.DecisionTree()
    print('Class [num(2), num(4)] = ', np.unique(train[10], return_counts=True)[1])

    tree = dtm.ID3(train, train, [1, 2, 3, 4, 5, 6, 7, 8, 9], 10)

    pprint(tree)









