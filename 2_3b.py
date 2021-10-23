import pandas as pd
import numpy as np
from sklearn import tree

if __name__ == '__main__':

    # Dataset setting
    dataset = pd.read_csv('./data/breast-cancer-wisconsin.data', header=None)
    indexName = dataset[dataset[6] == '?'].index
    dataset.drop(indexName, inplace=True)

    x = dataset[[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    y = dataset[10]  # last attribute which has 2 or 4

    # dataset.columns = ['Sample Code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    x_train = x[:490]
    y_train = y[:490]
    x_validation = x[490:560]
    y_validation = y[490:560]
    x_test = x[560:]
    y_test = y[560:]

    print(x_train)
    print(x_validation)

    dtm = tree.DecisionTreeClassifier(criterion='entropy')  # fail to run DecisionTree class, so use sklearn instead
    dtm = dtm.fit(x_train, y_train)

    print("Validation set accuracy: ", np.sum(y_validation == dtm.predict(x_validation))/len(y_validation))

    print("Test set accuracy: ", np.sum(y_test == dtm.predict(x_test)) / len(y_test))
    