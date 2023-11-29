from prettytable import PrettyTable
import torch

def simplify_number(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f} K"
    else:
        return str(num)

def get_model_param(model):
    param_num = {}
    total_param = 0
    for name, param in model.named_parameters():
        n = name.split(".")[:2]
        n = ".".join(n)
        if n not in param_num:
            param_num[n] = param.numel()
        else:
            param_num[n] += param.numel()
        total_param += param.numel()
    return  param_num, total_param

def get_model_param_table(param_num_bef, param_num_aft):
    table = PrettyTable()
    table.field_names = ["Layer", "Before", "After", "Ratio"]
    total_bef = 0
    total_aft = 0
    for name in param_num_bef:
        bef = param_num_bef[name]
        aft = param_num_aft[name]
        ratio = aft / bef
        total_bef += bef
        total_aft += aft
        table.add_row([name, simplify_number(bef), simplify_number(aft), f"{ratio:.2f}"])
    table.add_row(["Total", simplify_number(total_bef), simplify_number(total_aft), f"{total_aft / total_bef:.2f}"])
    return table.get_formatted_string()

def get_model_spec(model):
    layer_spec = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layer_spec[name] = f"Conv {module.kernel_size[0]}x{module.kernel_size[1]}, {module.out_channels}"
        elif isinstance(module, torch.nn.Linear):
            layer_spec[name] = f"Linear, {module.out_features}"
    return layer_spec
            
def get_model_spec_table(spec_bef, spec_aft):
    table = PrettyTable()
    table.field_names = ["Layer", "Before", "After", "Pruned"]
    for name in spec_bef:
        bef = spec_bef[name]
        aft = spec_aft[name]
        pruned = bef == aft
        table.add_row([name, bef, aft, pruned])
    return table.get_formatted_string()