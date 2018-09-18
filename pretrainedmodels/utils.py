import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from munch import munchify
from collections import OrderedDict


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False, random_hflip=False, random_vflip=False):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)
            
    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self, model, scale=0.875):
        self.load = LoadImage()
        self.tf = TransformImage(model, scale=scale)

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



# ---- Public functions
# https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/utils/flops_benchmark.py
#
def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
    the flops count should be activated and the model should be run on an input
    image.
    
    Example:
    
    fcn = add_flops_counting_methods(fcn)
    fcn = fcn.cuda().train()
    fcn.start_flops_count()
    

    _ = fcn(batch)
    
    fcn.compute_average_flops_cost() / 1e9 / 2 # Result in GFLOPs per image in batch
    
    Important: dividing by 2 only works for resnet models -- see below for the details
    of flops computation.
    
    Attention: we are counting multiply-add as two flops in this work, because in
    most resnet models convolutions are bias-free (BN layers act as bias there)
    and it makes sense to count muliply and add as separate flops therefore.
    This is why in the above example we divide by 2 in order to be consistent with
    most modern benchmarks. For example in "Spatially Adaptive Computatin Time for Residual
    Networks" by Figurnov et al multiply-add was counted as two flops.
    
    This module computes the average flops which is necessary for dynamic networks which
    have different number of executed layers. For static networks it is enough to run the network
    once and get statistics (above example).
    
    Implementation:
    The module works by adding batch_count to the main module which tracks the sum
    of all batch sizes that were run through the network.
    
    Also each convolutional layer of the network tracks the overall number of flops
    performed.
    
    The parameters are updated with the help of registered hook-functions which
    are being called each time the respective layer is executed.
    
    Parameters
    ----------
    net_main_module : torch.nn.Module
        Main module containing network
        
    Returns
    -------
    net_main_module : torch.nn.Module
        Updated main module with new methods/attributes that are used
        to compute flops.
    """
    
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)
    
    net_main_module.reset_flops_count()
    
    # Adding varialbles necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)
    
    return net_main_module



def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    
    Returns current mean flops consumption per image.
    
    """
    
    batches_count = self.__batch_counter__
    
    flops_sum = 0
    
    for module in self.modules():

        if isinstance(module, torch.nn.Conv2d):

            flops_sum += module.__flops__
    
    
    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    
    """
    
    add_batch_counter_hook_function(self)
    
    self.apply(add_flops_counter_hook_function)

    
def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    
    """
    
    remove_batch_counter_hook_function(self)
    
    self.apply(remove_flops_counter_hook_function)

    
def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    
    Resets statistics computed so far.
    
    """
    
    add_batch_counter_variables_or_reset(self)
    
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    
    def add_flops_mask_func(module):
        
        if isinstance(module, torch.nn.Conv2d):
            
            module.__mask__ = mask
    
    module.apply(add_flops_mask_func)

    
def remove_flops_mask(module):
    
    module.apply(add_flops_mask_variable_or_reset)

    
# ---- Internal functions


def conv_flops_counter_hook(conv_module, input, output):
        
    # Can have multiple inputs, getting the first one
    input = input[0]
    
    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]
    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    
    # We count multiply-add as 2 flops
    conv_per_position_flops = 2 * kernel_height * kernel_width * in_channels * out_channels
    
    active_elements_count = batch_size * output_height * output_width
    if conv_module.__mask__ is not None:
        
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()
        
    
    overall_conv_flops = conv_per_position_flops * active_elements_count
      
    bias_flops = 0
    
    if conv_module.bias is not None:
        
        bias_flops = out_channels * active_elements_count
    
    overall_flops = overall_conv_flops + bias_flops
    overall_flops /= groups
    #print('{} {} {} {} {} {}: {}'.format(in_channels, kernel_height, kernel_width, out_channels, output_height, output_width, overall_flops))
    
    conv_module.__flops__ += overall_flops

    
def batch_counter_hook(module, input, output):
    
    # Can have multiple inputs, getting the first one
    input = input[0]
    
    batch_size = input.shape[0]
    
    module.__batch_counter__ += batch_size


    
def add_batch_counter_variables_or_reset(module):
    
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    
    if hasattr(module, '__batch_counter_handle__'):
        
        return
    
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle

    
def remove_batch_counter_hook_function(module):
    
    if hasattr(module, '__batch_counter_handle__'):
        
        module.__batch_counter_handle__.remove()
        
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    
    if isinstance(module, torch.nn.Conv2d):
        
        module.__flops__ = 0

def add_flops_counter_hook_function(module):
        
    if isinstance(module, torch.nn.Conv2d):
        
        if hasattr(module, '__flops_handle__'):
            
            return

        handle = module.register_forward_hook(conv_flops_counter_hook)
        module.__flops_handle__ = handle

def remove_flops_counter_hook_function(module):
    
    if isinstance(module, torch.nn.Conv2d):
        
        if hasattr(module, '__flops_handle__'):
            
            module.__flops_handle__.remove()
            
            del module.__flops_handle__

# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    
    if isinstance(module, torch.nn.Conv2d):
        
        module.__mask__ = None


def summary(input_size, model):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                if isinstance(input[0], tuple):
                    summary[m_key]['input_shape'] = list(input[0][1].size())
                else:
                    summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, tuple):
                    summary[m_key]['output_shape'] = list(output[1].size())
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias') and module.bias is not None:
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
                
        dtype = torch.cuda.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [torch.rand(1,*in_size).type(dtype) for in_size in input_size]
        else:
            x = torch.rand(1,*input_size).type(dtype)
              
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        #print('----------------------------------------------------------------')
        #line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        #print(line_new)
        #print('================================================================')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            ## input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, summary[layer]['output_shape'], summary[layer]['nb_params'])
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
        #    print(line_new)
        print('================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('----------------------------------------------------------------')
        return summary, total_params