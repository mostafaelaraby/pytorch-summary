import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(
    model,
    input_size,
    batch_size=-1,
    device=torch.device("cuda:0"),
    dtypes=None,
    with_hyper_params=False,
):
    if not (torch.cuda.is_available()):
        device = torch.device("cpu")
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes, with_hyper_params
    )
    print(result)

    return params_info


def summary_string(
    model,
    input_size,
    batch_size=-1,
    device=torch.device("cuda:0"),
    dtypes=None,
    with_hyper_params=False,
):
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ""

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output if o is not None
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            summary[m_key]["hyper_params"] = {}
            if with_hyper_params:
                primitive = (int, str, bool, float, tuple)
                for attr, value in module.__dict__.items():
                    if value is None or attr == "training":
                        continue
                    if isinstance(value, primitive):
                        summary[m_key]["hyper_params"][attr] = str(value)

        if not isinstance(module, nn.Sequential) and not isinstance(
            module, nn.ModuleList
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [
        torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    if with_hyper_params:
        table_width = 100
    else:
        table_width = 64
    print("-" * table_width)
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #"
    )
    if with_hyper_params:
        line_new = "{:>64} {:>35}".format(line_new, "Layer params")

    print(line_new)
    print("=" * table_width)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if len(summary[layer]["hyper_params"]) > 0:
            for hyper_param, hyper_param_value in summary[layer][
                "hyper_params"
            ].items():
                line_new = "{:>64} {:>15}{:>20}".format(
                    line_new, hyper_param, hyper_param_value
                )
                print(line_new)
                summary_str += line_new + "\n"
                line_new = ""
        else:
            print(line_new)
            summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(
        np.prod(sum(input_size, ())) * batch_size * 4.0 / (1024 ** 2.0)
    )
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024 ** 2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * table_width + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += (
        "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    )
    summary_str += "-" * table_width + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * table_width + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)
