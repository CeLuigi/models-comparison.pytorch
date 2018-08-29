import GPUtil
import argparse
import sys
import json

import torch
import torch.backends.cudnn as cudnn

sys.path.append('..')
import pretrainedmodels
import pretrainedmodels.utils as utils

model_names = sorted(name for name in pretrainedmodels.__dict__
        if not name.startswith("__")
        and name.islower()
        and callable(pretrainedmodels.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Estimate GFlops')
parser.add_argument('-b', '--batch-size', default=1, type=int,
        metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--save', default='../memory.json', help='models info')


def main():
        args = parser.parse_args()

        try:
                with open(args.save) as fp:
                        model_info = json.load(fp)
        except:
                model_info = {}

        gpu = GPUtil.getGPUs()[0]
        empty_gpu = gpu.memoryUsed
        
        for m in model_names:
                if not m in model_info.keys() and not m.startswith('dpn'):
                        
                        # create model
                        print("=> creating model '{}'".format(m))
                        if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
                                print("=> using pre-trained parameters '{}'".format(args.pretrained))
                                model = pretrainedmodels.__dict__[m](num_classes=1000,
                                        pretrained=args.pretrained)
                        else:
                                model = pretrainedmodels.__dict__[m]()

                        cudnn.benchmark = True

                        scale = 0.875

                        print('Images transformed from size {} to {}'.format(
                                int(round(max(model.input_size) / scale)),
                                model.input_size))

                        model = model.cuda().eval()
			
			with torch.no_grad():
                                _ = model(torch.randn(args.batch_size, *model.input_size).cuda(non_blocking=True))
                        
                        gpu = GPUtil.getGPUs()[0]
                        busy_gpu = gpu.memoryUsed - empty_gpu
#                        import ipdb ; ipdb.set_trace()
                        model_info[m] = gpu.memoryUsed - empty_gpu

                        with open(args.save, 'w') as fp:
                                json.dump(model_info, fp)

                        del model
                        torch.cuda.empty_cache()

if __name__ == '__main__':
        main()

#print('GPU {}\n\tload: {}\n\tused memory: {}\n\tavail memory: {}\n\tused memory: {}\n\tfree memory: {}'.format(gpu.name,gpu.load*100,gpu.memoryUtil*100,gpu.memoryTotal,gpu.memoryUsed,gpu.memoryFree))
