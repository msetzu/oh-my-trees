from typing import Optional, Dict, Tuple

import numpy
from sklearn.metrics import classification_report

from ohmt.trees.structure.trees import ObliqueTree, InternalNode


class MemoizedEvaluator:
    """Evaluate Trees."""
    def __init__(self, tree: ObliqueTree, data: numpy.ndarray, labels: Optional[numpy.ndarray] = None):
        """Create a MemEvaluator that evaluates on `data`, optionally with the given `labels`."""
        self.data = data
        self.nr_instances = data.shape[0]
        self.dimensionality = data.shape[1]
        self.labels = labels
        self.tree = tree
        self.predictions = tree.predict(data)

    def accuracy(self, data: numpy.ndarray, labels: numpy.ndarray) -> float:
        """
        Evaluate the accuracy of the given `tree` on the given `data` and `labels`

        Args:
            data: The data.
            labels: The labels.
        """
        return sum(self.tree.predict(data) == labels) / labels.size

    def confusion_matrix(self, data: numpy.array, labels: numpy.ndarray) -> Tuple[float, float, float, float]:
        """Compute the confusion matrix of the given `tree` on the given `labels`, if any.
        Data is provided in construction, see __init__

        Args:
            data: The data.
            labels: The labels, if any. If None, use the labels given in construction.

        Returns:
            A tuple (True positive rate, True negative rate, False positive rate, False negative rate)
        """
        predicted_labels = self.tree.predict(data)
        stacked_labels = numpy.vstack((labels, predicted_labels)).transpose()

        true_positive_rate = (stacked_labels.sum(axis=1) == 2).sum() / (labels == 1).sum()
        true_negative_rate = (stacked_labels.sum(axis=1) == 0).sum() / (labels == 0).sum()

        false_positive_rate = len([i for i in range(stacked_labels.shape[0]) if all(stacked_labels[i] == [0, 1])])
        false_negative_rate = len([i for i in range(stacked_labels.shape[0]) if all(stacked_labels[i] == [1, 0])])

        return true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate

    def complexity(self) -> dict:
        """
        Returns:
            A dictionary holding several complexity measures:
                "size": Number of nodes
                "length_mean": Mean number of non-zero coefficients of a node
                "length_std": Std of non-zero coefficients of a node
        """
        coefficients = [node.hyperplane.coefficients for node in self.tree.nodes.values()
                        if isinstance(node, InternalNode)]
        lengths = numpy.array([(node_coefficients != 0).sum() for node_coefficients in coefficients])
        size = len(lengths)

        return {
            "size": size,
            "length_mean": lengths.mean(),
            "length_std": lengths.std()
        }

    def report(self) -> Dict:
        """Full report on the tree.

        Returns:
            An analysis of the tree
        """
        accuracy_res = {"accuracy": self.accuracy(self.data, self.labels)}
        complexity_res = self.complexity()
        tp, tn, fp, fn = self.confusion_matrix(self.data, self.labels)
        report = classification_report(self.labels, self.predictions, output_dict=True)
        confusion_res = {
            "true_positive_rate": tp,
            "true_negative_rate": tn,
            "false_positive_rate": fp,
            "false_negative_rate": fn
        }
        res = dict()
        res.update(accuracy_res)
        res.update(complexity_res)
        res.update(confusion_res)
        res["full_report"] = report

        return res
