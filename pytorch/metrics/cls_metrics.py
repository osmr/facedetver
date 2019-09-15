"""
Evaluation Metrics for Image Classification.
"""

import math
import numpy as np
import torch
from .metric import EvalMetric

__all__ = ['Accuracy', 'Top1Error', 'F1', 'MCC', 'StoreMisses']


class Accuracy(EvalMetric):
    """
    Computes accuracy classification score.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="accuracy",
                 output_names=None,
                 label_names=None):
        super(Accuracy, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : torch.Tensor
            The labels of the data with class indices as values, one per sample.
        preds : torch.Tensor
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)


class Top1Error(Accuracy):
    """
    Computes top-1 error (inverted accuracy classification score).

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'top_1_error'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="top_1_error",
                 output_names=None,
                 label_names=None):
        super(Top1Error, self).__init__(
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names)

    def get(self):
        """
        Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst


class _BinaryClassificationMetrics(object):
    """
    Private container class for classification metric statistics. True/false positive and
     true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.
    """
    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.global_true_positives = 0
        self.global_false_negatives = 0
        self.global_false_positives = 0
        self.global_true_negatives = 0

    def update_binary_stats(self, label, pred):
        """
        Update various binary classification counts for a single (label, pred)
        pair.

        Parameters
        ----------
        label : np.array
            The labels of the data.
        pred : np.array
            Predicted values.
        """
        pred_label = pred

        pred_true = (pred_label == 1)
        pred_false = 1 - pred_true
        label_true = (label == 1)
        label_false = 1 - label_true

        true_pos = (pred_true * label_true).sum()
        false_pos = (pred_true * label_false).sum()
        false_neg = (pred_false * label_true).sum()
        true_neg = (pred_false * label_false).sum()
        self.true_positives += true_pos
        self.global_true_positives += true_pos
        self.false_positives += false_pos
        self.global_false_positives += false_pos
        self.false_negatives += false_neg
        self.global_false_negatives += false_neg
        self.true_negatives += true_neg
        self.global_true_negatives += true_neg

    @property
    def precision(self):
        if self.true_positives + self.false_positives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_positives)
        else:
            return 0.0

    @property
    def global_precision(self):
        if self.global_true_positives + self.global_false_positives > 0:
            return float(self.global_true_positives) / (self.global_true_positives + self.global_false_positives)
        else:
            return 0.0

    @property
    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_negatives)
        else:
            return 0.0

    @property
    def global_recall(self):
        if self.global_true_positives + self.global_false_negatives > 0:
            return float(self.global_true_positives) / (self.global_true_positives + self.global_false_negatives)
        else:
            return 0.0

    @property
    def fscore(self):
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.0

    @property
    def global_fscore(self):
        if self.global_precision + self.global_recall > 0:
            return 2 * self.global_precision * self.global_recall / (self.global_precision + self.global_recall)
        else:
            return 0.0

    def matthewscc(self, use_global=False):
        """
        Calculate the Matthew's Correlation Coefficent.

        Parameters
        ----------
        use_global : bool, default False
            Whether to use global statistic values.
        """
        if use_global:
            if not self.global_total_examples:
                return 0.0

            true_pos = float(self.global_true_positives)
            false_pos = float(self.global_false_positives)
            false_neg = float(self.global_false_negatives)
            true_neg = float(self.global_true_negatives)
        else:
            if not self.total_examples:
                return 0.0

            true_pos = float(self.true_positives)
            false_pos = float(self.false_positives)
            false_neg = float(self.false_negatives)
            true_neg = float(self.true_negatives)

        terms = [(true_pos + false_pos),
                 (true_pos + false_neg),
                 (true_neg + false_pos),
                 (true_neg + false_neg)]
        denom = 1.
        for t in filter(lambda t: t != 0.0, terms):
            denom *= t
        return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(denom)

    @property
    def total_examples(self):
        return self.false_negatives + self.false_positives + \
               self.true_negatives + self.true_positives

    @property
    def global_total_examples(self):
        return self.global_false_negatives + self.global_false_positives + \
               self.global_true_negatives + self.global_true_positives

    def local_reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

    def reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.global_false_positives = 0
        self.global_false_negatives = 0
        self.global_true_positives = 0
        self.global_true_negatives = 0


class F1(EvalMetric):
    """
    Computes the F1 score of a binary classification problem.

    Parameters
    ----------
    name : str, default 'f1'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    average : str, default 'macro'
        Strategy to be used for aggregating across mini-batches.
            "macro": average the F1 scores for each batch.
            "micro": compute a single F1 score across all batches.
    """
    def __init__(self,
                 name="f1",
                 output_names=None,
                 label_names=None,
                 average="macro"):
        self.average = average
        self.metrics = _BinaryClassificationMetrics()
        EvalMetric.__init__(
            self,
            name=name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = 1

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred = torch.argmax(preds, dim=self.axis)
            else:
                pred = preds
            pred = pred.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            assert (len(label) == len(pred))
            self.metrics.update_binary_stats(label, pred)

            if self.average == "macro":
                self.sum_metric += self.metrics.fscore
                self.global_sum_metric += self.metrics.global_fscore
                self.num_inst += 1
                self.global_num_inst += 1
                self.metrics.reset_stats()
            else:
                self.sum_metric = self.metrics.fscore * self.metrics.total_examples
                self.global_sum_metric = self.metrics.global_fscore * self.metrics.global_total_examples
                self.num_inst = self.metrics.total_examples
                self.global_num_inst = self.metrics.global_total_examples

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.sum_metric = 0.0
        self.num_inst = 0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0
        self.metrics.reset_stats()

    def reset_local(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.sum_metric = 0.0
        self.num_inst = 0
        self.metrics.local_reset_stats()


class MCC(EvalMetric):
    """
    Computes the Matthews Correlation Coefficient of a binary classification problem.

    Parameters
    ----------
    name : str, default 'mcc'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    average : str, default 'macro'
        Strategy to be used for aggregating across mini-batches.
            "macro": average the MCC for each batch.
            "micro": compute a single MCC across all batches.
    """
    def __init__(self,
                 name="mcc",
                 output_names=None,
                 label_names=None,
                 average="macro"):
        self._average = average
        self._metrics = _BinaryClassificationMetrics()
        EvalMetric.__init__(
            self,
            name=name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = 1

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred = torch.argmax(preds, dim=self.axis)
            else:
                pred = preds
            pred = pred.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            assert (len(label) == len(pred))
            self._metrics.update_binary_stats(label, pred)

            if self._average == "macro":
                self.sum_metric += self._metrics.matthewscc()
                self.global_sum_metric += self._metrics.matthewscc(use_global=True)
                self.num_inst += 1
                self.global_num_inst += 1
                self._metrics.reset_stats()
            else:
                self.sum_metric = self._metrics.matthewscc() * self._metrics.total_examples
                self.global_sum_metric = self._metrics.matthewscc(use_global=True) *\
                                         self._metrics.global_total_examples
                self.num_inst = self._metrics.total_examples
                self.global_num_inst = self._metrics.global_total_examples

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.sum_metric = 0.0
        self.num_inst = 0.0
        self.global_sum_metric = 0.0
        self.global_num_inst = 0.0
        self._metrics.reset_stats()

    def reset_local(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.sum_metric = 0.0
        self.num_inst = 0.0
        self._metrics.local_reset_stats()


class StoreMisses(EvalMetric):
    """
    Fake metric, that computes indices of misses.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'store_errs'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="store_errs",
                 output_names=None,
                 label_names=None):
        super(StoreMisses, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.ind_list = []
        self.last_ind = 0

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            assert (len(label) == len(pred_label))
            sample_count = len(label)
            inds = np.arange(self.last_ind, self.last_ind + sample_count)
            new_ind_list = inds[label != pred_label]
            self.ind_list += list(new_ind_list)
            self.last_ind += sample_count

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0
        self.ind_list = []
        self.last_ind = 0

    def get(self):
        """
        Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of int
           Value of the evaluations.
        """
        return self.name, self.ind_list
