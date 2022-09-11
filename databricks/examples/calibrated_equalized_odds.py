# Databricks notebook source
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


# COMMAND ----------

# from aif360.sklearn.utils import check_inputs, check_groups

# code copied from this location
# https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/utils.py

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d


def check_inputs(X, y, sample_weight=None, ensure_2d=True):
    """Input validation for debiasing algorithms.
    Checks all inputs for consistent length, validates shapes (optional for X),
    and returns an array of all ones if sample_weight is ``None``.
    Args:
        X (array-like): Input data.
        y (array-like, shape = (n_samples,)): Target values.
        sample_weight (array-like, optional): Sample weights.
        ensure_2d (bool, optional): Whether to raise a ValueError if X is not
            2D.
    Returns:
        tuple:
            * **X** (`array-like`) -- Validated X. Unchanged.
            * **y** (`array-like`) -- Validated y. Possibly converted to 1D if
              not a :class:`pandas.Series`.
            * **sample_weight** (`array-like`) -- Validated sample_weight. If no
              sample_weight is provided, returns a consistent-length array of
              ones.
    """
    if ensure_2d and X.ndim != 2:
        raise ValueError("Expected X to be 2D, got ndim == {} instead.".format(
                X.ndim))
    if not isinstance(y, pd.Series):  # don't cast Series -> ndarray
        y = column_or_1d(y)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])
    check_consistent_length(X, y, sample_weight)
    return X, y, sample_weight

def check_groups(arr, prot_attr, ensure_binary=False):
    """Get groups from the index of arr.
    If there are multiple protected attributes provided, the index is flattened
    to be a 1-D Index of tuples. If ensure_binary is ``True``, raises a
    ValueError if there are not exactly two unique groups. Also checks that all
    provided protected attributes are in the index.
    Args:
        arr (array-like): Either a Pandas object containing protected attribute
            information in the index or array-like with explicit protected
            attribute array(s) for `prot_attr`.
        prot_attr (label or array-like or list of labels/arrays): Protected
            attribute(s). If contains labels, arr must include these in its
            index. If ``None``, all protected attributes in ``arr.index`` are
            used. Can also be 1D array-like of the same length as arr or a
            list of a combination of such arrays and labels in which case, arr
            may not necessarily be a Pandas type.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.
    Returns:
        tuple:
            * **groups** (:class:`pandas.Index`) -- Label (or tuple of labels)
              of protected attribute for each sample in arr.
            * **prot_attr** (`FrozenList`) -- Modified input. If input is a
              single label, returns single-item list. If input is ``None``
              returns list of all protected attributes.
    """
    arr_is_pandas = isinstance(arr, (pd.DataFrame, pd.Series))
    if prot_attr is None:  # use all protected attributes provided in arr
        if not arr_is_pandas:
            raise TypeError("Expected `Series` or `DataFrame` for arr, got "
                           f"{type(arr).__name__} instead. Otherwise, pass "
                            "explicit prot_attr array(s).")
        groups = arr.index
    elif arr_is_pandas:
        df = arr.index.to_frame()
        groups = df.set_index(prot_attr).index  # let pandas handle errors
    else:  # arr isn't pandas. might be okay if prot_attr is array-like
        df = pd.DataFrame(index=[None]*len(arr))  # dummy to check lengths match
        try:
            groups = df.set_index(prot_attr).index
        except KeyError as e:
            raise TypeError("arr does not include protected attributes in the "
                            "index. Check if this got dropped or prot_attr is "
                            "formatted incorrectly.") from e
    prot_attr = groups.names
    groups = groups.to_flat_index()

    n_unique = groups.nunique()
    if ensure_binary and n_unique != 2:
        raise ValueError("Expected 2 protected attribute groups, got "
                        f"{groups.unique() if n_unique > 5 else n_unique}")

    return groups, prot_attr


# COMMAND ----------

# from aif360.sklearn.metrics import difference, base_rate
# from aif360.sklearn.metrics import generalized_fnr, generalized_fpr
from sklearn.metrics._classification import _prf_divide, _check_zero_division

# code exceperted from this github repo
# https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py

def generalized_fpr(y_true, probas_pred, *, pos_label=1, sample_weight=None,
                    zero_division='warn'):
    r"""Return the ratio of generalized false positives to negative examples in
    the dataset, :math:`GFPR = \tfrac{GFP}{N}`.
    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.
    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
    Returns:
        float: Generalized false positive rate.
    """
    _check_zero_division(zero_division)
    y_true, probas_pred, sample_weight = check_inputs(y_true, probas_pred,
                                                      sample_weight, False)

    idx = (y_true != pos_label)
    gfp = np.array([np.dot(probas_pred[idx], sample_weight[idx])])
    neg = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfp, neg, 'generalized FPR', 'negative', None,
                       ('generalized FPR',), zero_division).item()

def generalized_fnr(y_true, probas_pred, *, pos_label=1, sample_weight=None,
                    zero_division='warn'):
    r"""Return the ratio of generalized false negatives to positive examples in
    the dataset, :math:`GFNR = \tfrac{GFN}{P}`.
    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.
    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
    Returns:
        float: Generalized false negative rate.
    """
    _check_zero_division(zero_division)
    y_true, probas_pred, sample_weight = check_inputs(y_true, probas_pred,
                                                      sample_weight, False)

    idx = (y_true == pos_label)
    gfn = np.array([np.dot(1 - probas_pred[idx], sample_weight[idx])])
    pos = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfn, pos, 'generalized FNR', 'positive', None,
                       ('generalized FNR',), zero_division).item()

def base_rate(y_true, y_pred=None, *, pos_label=1, sample_weight=None):
    r"""Compute the base rate, :math:`Pr(Y = \text{pos_label}) = \frac{P}{P+N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Base rate.
    """
    idx = (y_true == pos_label)
    return np.average(idx, weights=sample_weight)

def difference(func, y_true, y_pred=None, prot_attr=None, priv_group=1,
               sample_weight=None, **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.
    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.
    Unprivileged group is taken to be the inverse of the privileged group.
    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.
    Returns:
        scalar: Difference in metric value for unprivileged and privileged
        groups.
    Examples:
        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> difference(precision_score, y, y_pred, prot_attr='sex',
        ... priv_group='male')
        -0.06955430006277463
    """
    groups, _ = check_groups(y_true, prot_attr)
    idx = (groups == priv_group)
    unpriv = [y[~idx] for y in (y_true, y_pred) if y is not None]
    priv = [y[idx] for y in (y_true, y_pred) if y is not None]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)

# COMMAND ----------

# code copied from this location...
# https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/postprocessing/calibrated_equalized_odds.py

class CalibratedEqualizedOdds(BaseEstimator, ClassifierMixin):
    """Calibrated equalized odds post-processor.

    Calibrated equalized odds is a post-processing technique that optimizes over
    calibrated classifier score outputs to find probabilities with which to
    change output labels with an equalized odds objective [#pleiss17]_.

    Note:
        A :class:`~sklearn.pipeline.Pipeline` expects a single estimation step
        but this class requires an estimator's predictions as input. See
        :class:`PostProcessingMeta` for a workaround.

    See also:
        :class:`PostProcessingMeta`

    References:
        .. [#pleiss17] `G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and
           K. Q. Weinberger, "On Fairness and Calibration," Conference on Neural
           Information Processing Systems, 2017.
           <http://papers.nips.cc/paper/7151-on-fairness-and-calibration.pdf>`_

    Adapted from:
    https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/calib_eq_odds.py

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for post-
            processing.
        groups_ (array, shape (2,)): A list of group labels known to the
            classifier. Note: this algorithm require a binary division of the
            data.
        classes_ (array, shape (num_classes,)): A list of class labels known to
            the classifier. Note: this algorithm treats all non-positive
            outcomes as negative (binary classification only).
        pos_label_ (scalar): The label of the positive class.
        mix_rates_ (array, shape (2,)): The interpolation parameters -- the
            probability of randomly returning the group's base rate. The group
            for which the cost function is higher is set to 0.
    """
    def __init__(self, prot_attr=None, cost_constraint='weighted',
                 random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the post-processing. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used. Note: This algorithm requires there
                be exactly 2 groups (privileged and unprivileged).
            cost_constraint ('fpr', 'fnr', or 'weighted'): Which equal-cost
                constraint to satisfy: generalized false positive rate ('fpr'),
                generalized false negative rate ('fnr'), or a weighted
                combination of both ('weighted').
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for sampling from the mix rates.
        """
        self.prot_attr = prot_attr
        self.cost_constraint = cost_constraint
        self.random_state = random_state

    def _more_tags(self):
        return {'requires_proba': True}

    def _weighted_cost(self, y_true, probas_pred, pos_label=1,
                       sample_weight=None):
        """Evaluates the cost function specified by ``self.cost_constraint``."""
        if self.cost_constraint == 'fpr':
            return generalized_fpr(y_true, probas_pred, pos_label=pos_label,
                                   sample_weight=sample_weight)
        elif self.cost_constraint == 'fnr':
            return generalized_fnr(y_true, probas_pred, pos_label=pos_label,
                                   sample_weight=sample_weight)
        elif self.cost_constraint == 'weighted':
            fpr = generalized_fpr(y_true, probas_pred, pos_label=pos_label,
                                  sample_weight=sample_weight)
            fnr = generalized_fnr(y_true, probas_pred, pos_label=pos_label,
                                  sample_weight=sample_weight)
            br = base_rate(y_true, probas_pred, pos_label=pos_label,
                           sample_weight=sample_weight)
            return fpr * (1 - br) + fnr * br
        else:
            raise ValueError("`cost_constraint` must be one of: 'fpr', 'fnr', "
                             "or 'weighted'")

    def fit(self, X, y, labels=None, pos_label=1, sample_weight=None):
        """Compute the mixing rates required to satisfy the cost constraint.

        Args:
            X (array-like): Probability estimates of the targets as returned by
                a ``predict_proba()`` call or equivalent.
            y (pandas.Series): Ground-truth (correct) target values.
            labels (list, optional): The ordered set of labels values. Must
                match the order of columns in X if provided. By default,
                all labels in y are used in sorted order.
            pos_label (scalar, optional): The label of the positive class.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        groups, self.prot_attr_ = check_groups(y, self.prot_attr,
                                               ensure_binary=True)
        self.classes_ = np.array(labels) if labels is not None else np.unique(y)
        self.groups_ = np.unique(groups)
        self.pos_label_ = pos_label

        if len(self.classes_) != 2:
            raise ValueError('Only binary classification is supported.')
        if len(self.classes_) != X.shape[1]:
            raise ValueError('Only binary classification is supported. X should'
                    ' contain one column per class. Got: {} columns.'.format(
                            X.shape[1]))

        if pos_label not in self.classes_:
            raise ValueError('pos_label={} is not in the set of labels. The '
                    'valid values are:\n{}'.format(pos_label, self.classes_))

        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        try:
            X = X.iloc[:, pos_idx]
        except AttributeError:
            X = X[:, pos_idx]

        # local function to evaluate corresponding metric
        def _eval(func, grp_idx, trivial=False):
            idx = (groups == self.groups_[grp_idx])
            pred = np.full_like(X, self.base_rates_[grp_idx]) if trivial else X
            return func(y[idx], pred[idx], pos_label=pos_label,
                        sample_weight=sample_weight[idx])

        self.base_rates_ = [_eval(base_rate, i) for i in range(2)]

        costs = np.array([[_eval(self._weighted_cost, i, t) for i in range(2)]
                          for t in (False, True)])
        self.mix_rates_ = [
                (costs[0, 1] - costs[0, 0]) / (costs[1, 0] - costs[0, 0]),
                (costs[0, 0] - costs[0, 1]) / (costs[1, 1] - costs[0, 1])]
        self.mix_rates_[np.argmax(costs[0])] = 0

        return self

    def predict_proba(self, X):
        """The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        check_is_fitted(self, 'mix_rates_')
        rng = check_random_state(self.random_state)

        groups, _ = check_groups(X, self.prot_attr_)
        if not set(np.unique(groups)) <= set(self.groups_):
            raise ValueError('The protected groups from X:\n{}\ndo not '
                             'match those from the training set:\n{}'.format(
                                     np.unique(groups), self.groups_))

        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        X = X.iloc[:, pos_idx]

        yt = np.empty_like(X)
        for grp_idx in range(2):
            i = (groups == self.groups_[grp_idx])
            to_replace = (rng.rand(sum(i)) < self.mix_rates_[grp_idx])
            new_preds = X[i].copy()
            new_preds[to_replace] = self.base_rates_[grp_idx]
            yt[i] = new_preds

        return np.c_[1 - yt, yt] if pos_idx == 1 else np.c_[yt, 1 - yt]

    def predict(self, X):
        """Predict class labels for the given scores.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.predict_proba(X)
        return self.classes_[scores.argmax(axis=1)]

    def score(self, X, y, sample_weight=None):
        """Score the predictions according to the cost constraint specified.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.
            y (array-like): Ground-truth (correct) target values.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: Absolute value of the difference in cost function for the two
            groups (e.g. :func:`~aif360.sklearn.metrics.generalized_fpr` if
            ``self.cost_constraint`` is 'fpr')
        """
        check_is_fitted(self, ['classes_', 'pos_label_'])
        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        probas_pred = self.predict_proba(X)[:, pos_idx]

        return abs(difference(self._weighted_cost, y, probas_pred,
                prot_attr=self.prot_attr_, priv_group=self.groups_[1],
                pos_label=self.pos_label_, sample_weight=sample_weight))


# COMMAND ----------

# https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py


def base_rate(y_true, y_pred=None, *, pos_label=1, sample_weight=None):
    r"""Compute the base rate, :math:`Pr(Y = \text{pos_label}) = \frac{P}{P+N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Base rate.
    """
    idx = (y_true == pos_label)
    return np.average(idx, weights=sample_weight)

def selection_rate(y_true, y_pred, *, pos_label=1, sample_weight=None):
    r"""Compute the selection rate, :math:`Pr(\hat{Y} = \text{pos_label}) =
    \frac{TP + FP}{P + N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values. Ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Selection rate.
    """
    return base_rate(y_pred, pos_label=pos_label, sample_weight=sample_weight)

# ============================= META-METRICS ===================================
def difference(func, y_true, y_pred=None, prot_attr=None, priv_group=1,
               sample_weight=None, **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.
    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.
    Unprivileged group is taken to be the inverse of the privileged group.
    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.
    Returns:
        scalar: Difference in metric value for unprivileged and privileged
        groups.
    Examples:
        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> difference(precision_score, y, y_pred, prot_attr='sex',
        ... priv_group='male')
        -0.06955430006277463
    """
    groups, _ = check_groups(y_true, prot_attr)
    idx = (groups == priv_group)
    unpriv = [y[~idx] for y in (y_true, y_pred) if y is not None]
    priv = [y[idx] for y in (y_true, y_pred) if y is not None]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)


# ============================ GROUP FAIRNESS ==================================
def statistical_parity_difference(y_true, y_pred=None, *, prot_attr=None,
                                  priv_group=1, pos_label=1, sample_weight=None):
    r"""Difference in selection rates.
    .. math::
        Pr(\hat{Y} = \text{pos_label} | D = \text{unprivileged})
        - Pr(\hat{Y} = \text{pos_label} | D = \text{privileged})
    Note:
        If only y_true is provided, this will return the difference in base
        rates (statistical parity difference of the original dataset). If both
        y_true and y_pred are provided, only y_pred is used.
    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Statistical parity difference.
    See also:
        :func:`selection_rate`, :func:`base_rate`
    """
    rate = base_rate if y_pred is None else selection_rate
    return difference(rate, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)
    
def discrimination(y_true, y_pred, prot_attr):
    return abs(statistical_parity_difference(y_true, y_pred, prot_attr=prot_attr))
