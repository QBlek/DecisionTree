import numpy as np

class Node:
    def __init__(self, nodeId, label, isRoot=False, parentNode=None, leftNode=None,rightNode=None,isTerminal=False):
        self.nodeId = nodeId
        self.label = label
        self.isRoot = isRoot
        self.parentNode = parentNode
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.isTerminal = isTerminal
        self.level = 0

class DecisionTree():

    def __init__(self):
        self

    # calculate H(target) which has target attribute(Class: 2 for benign, 4 for malignant)
    def entropy(self, dataset):
        vals, id_counts1 = np.unique(dataset, return_counts=True)
        entropy_val = -np.sum(
            [(id_counts1[i] / np.sum(id_counts1)) * np.log(id_counts1[i] / np.sum(id_counts1)) for i in range(len(vals))])
        return entropy_val

    # calculate condition_entropy such as H(target|other_attribute)
    def infogain(self, dataset, attribute_name, target_name):
        target_entropy = self.entropy(dataset[target_name])

        vals, id_counts2 = np.unique(dataset[attribute_name], return_counts=True)
        condition_entropy = np.sum([(id_counts2[i] / np.sum(id_counts2)) * self.entropy(dataset.where(dataset[attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
        print('H(', attribute_name, ') = ', round(condition_entropy, 5))

        info_gain = target_entropy - condition_entropy
        return info_gain

    def ID3(self, dataset, dataset_original, attributes, target_name, parent_node_class=None):

        # Stop criteria
        # If target attribute's value is only one value(only True or only False) -> return that value
        if len(np.unique(dataset[target_name])) <= 1:
            return np.unique(dataset[target_name])[0]
        # If there is no dataset -> return max value from original dataset
        elif len(dataset) == 0:
            return np.unique(dataset_original[target_name])[np.argmax(np.unique(dataset_original[target_name], return_counts=True)[1])]
        # There is no attributes for decision
        elif len(attributes) == 0:
            return parent_node_class

        # build tree
        else:
            parent_node_class = np.unique(dataset[target_name])[np.argmax(np.unique(dataset[target_name], return_counts=True)[1])]

            # Decide attribute to split
            item_values = [self.infogain(dataset, attribute, target_name) for attribute in attributes]
            best_attribute_index = np.argmax(item_values)
            best_attribute = attributes[best_attribute_index]
            print("Split decision: ", best_attribute)

            # tree structure
            # tree = Node()
            tree = {best_attribute: {}}

            attributes = [i for i in attributes if i != best_attribute]

            print('Class [num(2), num(4)] = ', np.unique(dataset[10], return_counts=True)[1], "\n")

            # splitting as left and right
            left_set = dataset.where((dataset[best_attribute]) <= 5.5).dropna()
            right_set = dataset.where((dataset[best_attribute]) > 5.5).dropna()
            leftnode = self.ID3(left_set, dataset, attributes, target_name, parent_node_class)
            rightnode = self.ID3(right_set, dataset, attributes, target_name, parent_node_class)
            tree[best_attribute][0] = leftnode
            tree[best_attribute][1] = rightnode
            # problem was happened on making tree node, so it is not run on after some of iteration for treeing
            """
            # splitting for each class(1~10)
            for value in np.unique(dataset[best_attribute]):
                # drop.na() - we have 16 missing value, so use it to except missing value
                sub_dataset = dataset.where(dataset[best_attribute] == value).dropna()

                # iter ID3
                subtree = self.ID3(sub_dataset, dataset, attributes, target_name, parent_node_class)
                tree[best_attribute][value] = subtree
            """

            return tree
