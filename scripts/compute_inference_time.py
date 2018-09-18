import argparse
import gc
import numpy as np
import sys
import json
import time
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
parser.add_argument('--save', default='../fp_ms.json', help='result filename')


def measure(model, x):
	# synchronize gpu time and measure fp
	torch.cuda.synchronize()
	t0 = time.time()
	with torch.no_grad():
		y_pred = model(x)
	torch.cuda.synchronize()
	elapsed_fp = time.time() - t0
	
	return elapsed_fp


def benchmark(model, x):
	# transfer the model on GPU
	model = model.cuda().eval()
	
	# DRY RUNS
	for i in range(10):
		_ = measure(model, x)

	print('DONE WITH DRY RUNS, NOW BENCHMARKING')

	# START BENCHMARKING
	t_forward = []
	t_backward = []
	for i in range(10):
		t_fp = measure(model, x)
		t_forward.append(t_fp)
	
	# free memory
	del model
	
	return t_forward


#def measure(model, x, y):
#	# synchronize gpu time and measure fp
#	torch.cuda.synchronize()
#	t0 = time.time()
#	y_pred = model(x)
#	torch.cuda.synchronize()
#	elapsed_fp = time.time() - t0
#	
#	# zero gradients, synchronize time and measure
#	model.zero_grad()
#	t0 = time.time()
#	y_pred.backward(y)
#	torch.cuda.synchronize()
#	elapsed_bp = time.time()-t0
#	return elapsed_fp, elapsed_bp
#
#
#def benchmark(model, x, y):
#	# transfer the model on GPU
#	model.cuda()
#	
#	# DRY RUNS
#	for i in range(10):
#		_, _ = measure(model, x, y)
#
#	print('DONE WITH DRY RUNS, NOW BENCHMARKING')
#
#	# START BENCHMARKING
#	t_forward = []
#	t_backward = []
#	for i in range(10):
#		t_fp, t_bp = measure(model, x, y)
#		t_forward.append(t_fp)
#		t_backward.append(t_bp)
#	
#	# free memory
#	del model
#	
#	return t_forward, t_backward


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
			model = pretrainedmodels.__dict__[m](num_classes=1000)

			cudnn.benchmark = True

			scale = 0.875

			print('Images transformed from size {} to {}'.format(
				int(round(max(model.input_size) / scale)),
				model.input_size))

			batch_sizes = [1, 2, 4, 8, 16, 32, 64]
			mean_tfp = []
			std_tfp = []
			for i, bs in enumerate(batch_sizes):
				x = torch.randn(bs, *model.input_size).cuda()
				tmp = benchmark(model, x)
				# NOTE: we are estimating inference time per image
				mean_tfp.append(np.asarray(tmp).mean() / bs*1e3)
				std_tfp.append(np.asarray(tmp).std() /bs*1e3)
			
			model_info[m] = (mean_tfp, std_tfp)

			with open(args.save, 'w') as fp:
				json.dump(model_info, fp)

	# force garbage collection
	gc.collect()

	
#		# print results
#		print('FORWARD PASS: ', np.mean(np.asarray(t_fp)*1e3), '+/-', np.std(np.asarray(t_fp)*1e3))
#		print('BACKWARD PASS: ', np.mean(np.asarray(t_bp)*1e3), '+/-', np.std(np.asarray(t_bp)*1e3))
#		print('RATIO BP/FP:', np.mean(np.asarray(t_bp))/np.mean(np.asarray(t_fp)))
#		
#		# write the list of measures in files
#		fname = deep_net+'-benchmark.txt'
#		with open(fname, 'w') as f:
#			for (fp_time, bp_time) in zip(t_fp, t_bp):
#				f.write(str(fp_time)+" "+str(bp_time)+" \n")
#		

if __name__ == '__main__':
	main()