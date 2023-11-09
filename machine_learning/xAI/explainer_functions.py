import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from typing import List, Dict, Optional, Tuple

def extract_tree_decision_rules(
    decision_tree: DecisionTreeClassifier,
    feature_names: Optional[List[str]] = None, N: Optional[int] = None
    ) -> Dict[str, List[List[Tuple[str, str, float]]]]:
    """
    Extract the top decision rules from a fitted decision tree for each class.

    Parameters
    ----------
    decision_tree : DecisionTreeClassifier
        A fitted sklearn DecisionTreeClassifier.
    feature_names : List[str], optional
        Names of the features used in the decision tree. If None, the feature names
        are taken from the decision tree model, by default None.
    N : int, optional
        The number of top rules to extract for each class. If None, it defaults to
        the maximum depth of the tree, by default None.

    Returns
    -------
    Dict[str, List[List[Tuple[str, str, float]]]]
        A dictionary where keys are class names and values are lists of the top N rules.
        Each rule is represented as a list of tuples (feature, operator, threshold).

    Raises
    ------
    AssertionError
        If the decision tree model is not fitted or was fitted without feature names.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> iris = load_iris()
    >>> clf = DecisionTreeClassifier().fit(iris.data, iris.target)
    >>> rules = extract_tree_decision_rules(clf, feature_names=iris.feature_names)
    """
    
    # Validate the input model and feature names
    if feature_names is None:
        assert hasattr(decision_tree, 'tree_'), 'Decision tree model is not fitted.'
        assert hasattr(decision_tree, 'feature_names_in_'), 'Decision tree fitted without feature names.'
        feature_names = decision_tree.feature_names_in_
    
    # Map class names to their indices
    class_index_map = {cls: idx for idx, cls in enumerate(decision_tree.classes_)}

    # Initialize a dictionary to hold rules for each class
    class_rules = {cls: [] for cls in decision_tree.classes_}

    # Recursive function to extract the rules
    def recurse(node, rule):
        if decision_tree.tree_.children_left[node] != _tree.TREE_LEAF:
            # Current node is an internal node
            name = feature_names[decision_tree.tree_.feature[node]]
            threshold = decision_tree.tree_.threshold[node]
            rule.append((name, '<=', threshold))
            recurse(decision_tree.tree_.children_left[node], rule)
            rule.pop()
            rule.append((name, '>', threshold))
            recurse(decision_tree.tree_.children_right[node], rule)
            rule.pop()
        else:
            # Leaf node: determine the predicted class
            distribution = decision_tree.tree_.value[node][0]
            predicted_class_index = np.argmax(distribution)
            predicted_class = decision_tree.classes_[predicted_class_index]

            # Append rule to the predicted class
            class_rules[predicted_class].append(list(rule))

    recurse(0, [])

    # If N is None, set it to the maximum depth of the tree
    if N is None:
        N = decision_tree.tree_.max_depth

    # Sort rules for each class and select top N
    for class_name in class_rules:
        class_rules[class_name].sort(key=lambda x: len(x))  # Sort by rule length
        class_rules[class_name] = class_rules[class_name][:N]  # Keep top N rules

    return class_rules