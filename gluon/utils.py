import os
import re
import logging
import numpy as np
import mxnet as mx
from gluoncv2.model_provider import get_model
from .metrics.cls_metrics import Top1Error


def prepare_mx_context(num_gpus,
                       batch_size):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size *= max(1, num_gpus)
    return ctx, batch_size


def get_initializer(initializer_name):
    if initializer_name == "MSRAPrelu":
        return mx.init.MSRAPrelu()
    elif initializer_name == "Xavier":
        return mx.init.Xavier()
    elif initializer_name == "Xavier-gaussian-out-2":
        return mx.init.Xavier(
            rnd_type="gaussian",
            factor_type="out",
            magnitude=2)
    else:
        return None


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  dtype,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  tune_layers=None,
                  classes=None,
                  in_channels=None,
                  do_hybridize=True,
                  initializer=mx.init.MSRAPrelu(),
                  ctx=mx.cpu()):
    kwargs = {"ctx": ctx,
              "pretrained": use_pretrained}
    if classes is not None:
        kwargs["classes"] = classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx,
            ignore_extra=load_ignore_extra)

    net.cast(dtype)

    if do_hybridize:
        net.hybridize(
            static_alloc=True,
            static_shape=True)

    if pretrained_model_file_path or use_pretrained:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(initializer, ctx=ctx)
    else:
        net.initialize(initializer, ctx=ctx)

    if (tune_layers is not None) and tune_layers:
        tune_layers_pattern = re.compile(tune_layers)
        for k, v in net._collect_params_with_prefix().items():
            if tune_layers_pattern.match(k):
                logging.info("Fine-tune parameter: {}".format(k))
            else:
                v.grad_req = "null"
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(initializer, ctx=ctx)

    return net


def calc_net_weight_count(net):
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def validate(metric,
             net,
             val_data,
             batch_fn,
             data_source_needs_reset,
             dtype,
             ctx):
    if data_source_needs_reset:
        val_data.reset()
    metric.reset()
    for batch in val_data:
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        metric.update(labels_list, outputs_list)
    return metric


def report_accuracy(metric,
                    extended_log=False):
    metric_info = metric.get()
    if extended_log:
        msg_pattern = "{name}={value:.4f} ({value})"
    else:
        msg_pattern = "{name}={value:.4f}"
    if isinstance(metric, mx.metric.CompositeEvalMetric):
        msg = ""
        for m in zip(*metric_info):
            if msg != "":
                msg += ", "
            msg += msg_pattern.format(name=m[0], value=m[1])
    elif isinstance(metric, mx.metric.EvalMetric):
        msg = msg_pattern.format(name=metric_info[0], value=metric_info[1])
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
    return msg


def get_metric(metric_name, metric_extra_kwargs):
    if metric_name == "Top1Error":
        return Top1Error(**metric_extra_kwargs)
    elif metric_name == "F1":
        return mx.metric.F1(**metric_extra_kwargs)
    elif metric_name == "MCC":
        return mx.metric.MCC(**metric_extra_kwargs)
    else:
        raise Exception("Wrong metric name: {}".format(metric_name))


def get_composite_metric(metric_names, metric_extra_kwargs):
    if len(metric_names) == 1:
        metric = get_metric(metric_names[0], metric_extra_kwargs[0])
    else:
        metric = mx.metric.CompositeEvalMetric()
        for name, extra_kwargs in zip(metric_names, metric_extra_kwargs):
            metric.add(get_metric(name, extra_kwargs))
    return metric


def get_metric_name(metric, index):
    if isinstance(metric, mx.metric.CompositeEvalMetric):
        return metric.metrics[index].name
    elif isinstance(metric, mx.metric.EvalMetric):
        assert (index == 0)
        return metric.name
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
