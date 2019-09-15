"""
Evaluation Metrics for Image Classification.
"""

import numpy as np
import mxnet as mx

__all__ = ['Top1Error', 'StoreMisses']


class Top1Error(mx.metric.Accuracy):
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


class StoreMisses(mx.metric.EvalMetric):
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
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        assert (len(labels) == len(preds))
        for label, pred in zip(labels, preds):
            label_imask = label.asnumpy().astype(np.int32)
            pred_imask = mx.nd.argmax(pred, axis=self.axis).asnumpy().astype(np.int32)
            assert (len(label_imask) == len(pred_imask))
            sample_count = len(label_imask)
            inds = np.arange(self.last_ind, self.last_ind + sample_count)
            new_ind_list = inds[label_imask != pred_imask]
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
