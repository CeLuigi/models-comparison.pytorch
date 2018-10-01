import argparse
import gc
import numpy as np
import sys
import json
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F


sys.path.append('..')
import pretrainedmodels
import pretrainedmodels.utils as utils

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Estimate GFlops')
parser.add_argument('--save', default='../rf_values.json', help='result filename')


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def compute_RF_numerical(net,img_):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        #if classname.find('Conv') != -1:
        if type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d or type(m) == nn.Linear:
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.data.fill_(1)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0)
    net.apply(weights_init)
    img_ = Variable(img_,requires_grad=True)
    out_cnn=net(img_)
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    grad=torch.zeros(out_cnn.size()).cuda()
    l_tmp=[]
    for i in range(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
    #print(tuple(l_tmp))
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.cpu().numpy()
    idx_nonzeros=np.where(grad_np!=0)
    import ipdb;ipdb.set_trace()
    RF=[int(np.max(idx)-np.min(idx)+1) for idx in idx_nonzeros]

    return RF

def main():
    args = parser.parse_args()
    
    try:
        with open(args.save) as fp:
            model_info = json.load(fp)
    except:
        model_info = {}

    torch.manual_seed(1234)
    
    for m in model_names:
        if not m in model_info.keys():

            # create model
            print("=> creating model '{}'".format(m))
            model = pretrainedmodels.__dict__[m](num_classes=1000, pretrained='imagenet')
            model.cuda()
            model.eval()

            # for module in model.modules():
            #     if type(module) == nn.MaxPool2d:
            #         module = nn.AvgPool2d(
            #             module.kernel_size,
            #             module.stride,
            #             module.padding)
            #     elif type(module) == nn.BatchNorm2d:
            #         module = Identity()
            #     elif type(module) == nn.BatchNorm1d:
            #         module = Identity()

            print(model)
            cudnn.benchmark = True

            scale = 0.875

            print('Images transformed from size {} to {}'.format(
                int(round(max(model.input_size) / scale)),
                model.input_size))

            x = torch.ones(1, *model.input_size).cuda()

            rf_values = compute_RF_numerical(model, x)

            model_info[m] = rf_values
            print(m, rf_values)

            with open(args.save, 'w') as fp:
                json.dump(model_info, fp)

    # force garbage collection
    gc.collect()


if __name__ == '__main__':
    main()