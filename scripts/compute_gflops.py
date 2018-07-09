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
parser.add_argument('-b', '--batch-size', default=64, type=int,
	metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--save', default='../tflops_params.json', help='models info')


def main():
	args = parser.parse_args()
	
	model_info = {}
	for m in model_names:
		if not m.startswith('dpn'):
		
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
			model = utils.add_flops_counting_methods(model)
			model.start_flops_count()
			
			with torch.no_grad():
				_ = model(torch.randn(args.batch_size, *model.input_size).cuda(non_blocking=True))
			 
			div_coeff = 2 if m.startswith('resnet') else 1
			
			summary, n_params = utils.summary(model.input_size, model)
			model_info[m] = (model.compute_average_flops_cost() / 1e9 / div_coeff, n_params.item())


	with open(args.save, 'w') as fp:
		json.dump(model_info, fp)
		

if __name__ == '__main__':
	main()