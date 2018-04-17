from .imports import *
from .torch_imports import *

def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds==targs).float().mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh).float()==targs).float().mean()

def accuracy_multi_np(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def precision(**kwargs):
    return _wrapped_partial(precision, sklearn_precision, **kwargs)

def sklearn_precision(preds, targs, **kwargs):
    preds = torch.max(preds, dim=1)[1]
    return sklearn.metrics.precision_score(targs, preds, **kwargs)

def recall(**kwargs):
    return _wrapped_partial(recall, sklearn_recall, **kwargs)

def sklearn_recall(preds, targs, **kwargs):
    preds = torch.max(preds, dim=1)[1]
    return sklearn.metrics.recall_score(targs, preds, **kwargs)

def f1(**kwargs):
    return _wrapped_partial(f1, sklearn_f1, **kwargs)

def sklearn_f1(preds, targs, **kwargs):
    preds = torch.max(preds, dim=1)[1]
    return sklearn.metrics.f1_score(targs, preds, **kwargs)

def fbeta(**kwargs):
    return _wrapped_partial(fbeta, sklearn_fbeta, **kwargs)

def sklearn_fbeta(preds, targs, **kwargs):
    preds = torch.max(preds, dim=1)[1]
    return sklearn.metrics.fbeta_score(targs, preds, **kwargs)

def _wrapped_partial(wrapping_func, func_to_wrap, *args, **kwargs):
    """
    A helper func to add the __name__ and __doc__ attributes to a partial func.
    If we don't use it, it shows '<lambda>' or raises an exception ('functools.partial' object has no attribute '__name__')
    when printing the metric after each epoch.
    reference: http://tiao.io/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/

    Args:
        wrapping_func (function): the wrapping function, it's __name__ and __doc__ attributes will be used.

        func_to_wrap(function): function to wrap in a partial

        args: partial arguments

        kwargs: partial keyword arguments

    Returns:
        A partial function with __name__ and __doc__ attributes of the called_func param.
    """
    partial_func = partial(func_to_wrap, *args, **kwargs)
    update_wrapper(partial_func, wrapping_func)
    return partial_func


