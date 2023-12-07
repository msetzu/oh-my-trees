from __future__ import annotations

from typing import Optional, Dict, Callable

import numpy
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lasso, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ohmt.trees.splits.evaluation import gini
from ohmt.trees.structure.trees import Node, InternalNode, ObliqueTree
from ohmt.planes import OHyperplane
from ohmt.systems.utils import from_cart


class OmnivariateDT(ObliqueTree):
    """Parametric Oblique Tree built using the provided model family for internal node split."""

    def __init__(self, root: Optional[InternalNode] = None):
        super(OmnivariateDT, self).__init__(root)

    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Callable = gini,
             node_hyperparameters: Optional[Dict] = None,
             **step_hyperparameters) -> Optional[Node]:
        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels, validation_data=data)
        should_stop_for_one_class = self.should_stop_for_one_class(labels, parent_node, direction, validation_data=data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node,
                                                                       min_samples)

        if not should_stop_for_depth and not should_stop_for_min_samples and not should_stop_for_one_class:
            dual = data.shape[0] < data.shape[1]
            models = [Ridge(),
                      ElasticNet(),
                      SGDClassifier(loss="hinge", penalty="l1" if not dual else "l2", max_iter=100000),
                      LinearRegression(),
                      LinearSVC(penalty="l1" if not dual else "l2", dual=dual, max_iter=100000,
                                **node_hyperparameters if node_hyperparameters is not None else {}),
                      Lasso(),
                      DecisionTreeClassifier(max_depth=1)]
            model_names = ["ridge", "elastic", "sgd-svm", "linear", "svm", "lasso", "tree"]
            filtered_model_names = model_names if step_hyperparameters.get("models", None) is None else step_hyperparameters["models"]
            models = [m for m, name in zip(models, model_names) if name in filtered_model_names]
            losses = list()
            hyperplanes = list()
            for model_name, model in zip(model_names, models):
                model.fit(data, labels)
                if model_name == "tree":
                    if model.tree_.node_count > 1:
                        extracted_hyperplane, _ = from_cart(model)[0]
                        extracted_hyperplane = extracted_hyperplane[0]
                        hyperplane = OHyperplane.from_aphyperplane(extracted_hyperplane, dimensionality=data.shape[1])
                    else:
                        hyperplanes.append(numpy.zeros(data.shape[1],))
                else:
                    hyperplane = OHyperplane(model.coef_.transpose().squeeze(), model.intercept_)
                fit_node_loss = node_fitness_function(hyperplane, data=data, labels=labels, classes=classes)
                losses.append(fit_node_loss)
                hyperplanes.append(hyperplane)
            losses = numpy.array(losses)
            best_idx = numpy.argmin(losses)
            hyperplane = hyperplanes[best_idx]

            if parent_node is not None:
                if self.should_stop_for_min_eps(parent_node, InternalNode(hyperplane),
                                                node_fitness_function=node_fitness_function,
                                                validation_data=data, validation_labels=labels, classes=classes,
                                                min_eps=min_eps):
                    fit_node = self.build_leaf(labels, data=data)
                else:
                    fit_node = InternalNode(hyperplane)
            else:
                fit_node = InternalNode(hyperplane)

            fit_node.model_name = model_names[best_idx]
            fit_node.losses = losses

        else:
            fit_node = self.build_leaf(labels, data=data)

        self.store(fit_node, data, labels)

        return fit_node
