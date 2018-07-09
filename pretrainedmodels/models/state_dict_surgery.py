from collections import OrderedDict


def update_state_dict_pytorch(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k] = v if v.ndimension() > 2 else v.squeeze()

    return new_state_dict