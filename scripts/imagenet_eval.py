import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import sys
sys.path.append('..')


import pretrainedmodels
import pretrainedmodels.utils

model_names = sorted(name for name in pretrainedmodels.__dict__
					 if not name.startswith("__")
					 and name.islower()
					 and callable(pretrainedmodels.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
					help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
					action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--save', default='../accuracy.json', help='models info')


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

			# Data loading code
			valdir = os.path.join(args.data, 'val')

			# if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
			#	 scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
			# else:
			#	 scale = 0.875
			scale = 0.875

			print('Images transformed from size {} to {}'.format(
				int(round(max(model.input_size) / scale)),
				model.input_size))

			val_tf = pretrainedmodels.utils.TransformImage(model, scale=scale)

			val_loader = torch.utils.data.DataLoader(
				datasets.ImageFolder(valdir, val_tf),
				batch_size=args.batch_size, shuffle=False,
				num_workers=args.workers, pin_memory=True)

			model = model.cuda()

			top1, top5 = validate(val_loader, model)
			model_info[m] = (top1, top5)
	
	with open(args.save, 'w') as fp:
		json.dump(model_info, fp)


def validate(val_loader, model):
	batch_time = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(non_blocking=True)
		input = input.cuda(non_blocking=True)
		
		# compute output
		with torch.no_grad():
			output = model(input)
		
		# measure accuracy and record loss
		prec1, prec5 = accuracy(output, target, topk=(1, 5))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

	print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
		  .format(top1=top1, top5=top5))

	return top1.avg, top5.avg


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


if __name__ == '__main__':
	main()