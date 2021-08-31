from tqdm import tqdm
import os
import warnings
import pickle as pk
import numpy as np
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import zipfile
from importlib import import_module

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp

from dataloader import *
from optimizer import get_optimizer
from loss import *
from parser import get_args
from trainer import *
from utils import *
from metric import metric_manager
from Summary import summary_string
from ddp_utils import *


def main():
	#parse arguments
	args = get_args()
	
	#make experiment reproducible if specified
	if args.reproducible:
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed) 
		np.random.seed(args.seed)
		random.seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False		
	
	#DDP setting
	args.ngpus_per_node = torch.cuda.device_count()
	args.world_size = args.ngpus_per_node
	args.rank = 0

	loader_args = {}

	#get utt_lists & define labels
	loader_args['dev_lines'] = sorted(get_utt_list(args.DB_vox2))
	loader_args['vox1_all_lines'] = sorted(get_utt_list(args.DB_vox1_all))
	loader_args['vox1_eval_lines'] = sorted(get_utt_list(args.DB_vox1_eval))  
	
	args.model['nb_dev_utt'] = len(loader_args['dev_lines'])
	print('#dev_lines: {}'.format(len(loader_args['dev_lines'])))
	print('#vox_all_lines: {}'.format(len(loader_args['vox1_all_lines'])))
	print('#vox_eval_lines: {}'.format(len(loader_args['vox1_eval_lines'])))
	
	#get label dictionary
	loader_args['d_label'], loader_args['l_label'], loader_args['d_spk2utt'] = make_d_label_spk2uttr(loader_args['dev_lines'])
	args.model['nb_spk'] = len(loader_args['l_label'])
	mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args, loader_args))

def main_worker(gpu, ngpus_per_node, args, loader_args):
	if gpu == 0:
	
		#set save directory
		save_dir = args.save_dir + args.name + '/'
		
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		if not os.path.exists(save_dir+'results/'):
			os.makedirs(save_dir+'results/')
		if not os.path.exists(save_dir+'weights/'):
			os.makedirs(save_dir+'weights/')
		with zipfile.ZipFile(
			save_dir + "codes.zip", "w", zipfile.ZIP_DEFLATED
		) as f_zip:
			zipdir("../", f_zip)
			
		#log experiment parameters to local
		f_params = open(save_dir + 'f_params.txt', 'w')
		for k, v in sorted(vars(args).items()):
			f_params.write('{}:\t{}\n'.format(k, v))
		f_params.close()

		dic_eval_trial = get_trials(args)
	

	#device setting
	cuda = torch.cuda.is_available()
	if not cuda: raise NotImplementedError("This script is written for single-node multi-GPUs env only.")
	print('Using GPU:{} for training'.format(gpu))
	torch.cuda.set_device(gpu)
	
	os.environ['MASTER_ADDR']='localhost'
	os.environ['MASTER_PORT']='8888'
	args.rank = args.rank * ngpus_per_node + gpu
	dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

	trnset_gen, trnset_sampler = get_train_loader(args, loader_args)
	
	loader_args['list_eval_nb_sample'] =[-1, 1, 2, 5]
	
	l_evalset_gen = get_vox1_eval_loader_list(args, loader_args)
	evalset_gen_vox1_all = get_vox1_all_loader(args, loader_args)
	
	args.nb_iter = len(trnset_gen)
	
	#define model	
	module = import_module("models.{}".format(args.module_name))
	_model = getattr(module, args.model_name)
	args.model["device"] = gpu
	model = _model(**args.model).cuda(gpu)

	if gpu == 0:

		result, nb_params = summary_string(model, (1, args.nb_samp))
		print('nunber of parameters: ',nb_params[0])

		f_model = open(save_dir + 'f_model.txt', 'w')
		f_model.write(result)
		f_model.close()

		metric_man = metric_manager(
				save_dir=save_dir,
				model=model,
				dic_eval_trial = dic_eval_trial,
				save_best_only=args.save_best_only
			)

	if not args.load_model: 
		model.apply(init_weights)
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(gpu)
	else:
		check_point = torch.load(args.load_model_path)
		check_point_dict = check_point['model']
		model.load_state_dict(check_point_dict)

	########################
	#Test trained model#
	########################
	if args.eval:

		###########################################################
		#Evaluation for various-duration utterance on Vox1-O trial#
		###########################################################
		l_embd = []
		l_all_ID = []

		for i, evalset_gen in enumerate(l_evalset_gen):
			#extract speaker embedding
			l_embedding, l_ID = extract_speaker_embedding(
				model = model,
				db_gen = evalset_gen, 
				gpu = gpu)
				
			dist.barrier()
			
			#gather utterance ID and speaker embedding to gpu 0
			l_all_ID.append(gather(l_ID))
			l_embd.append(gather(l_embedding))
			
		if gpu == 0:
			#convert list embd to dic embd
			d_embeddings_all = dic_embd(l_all_ID[0], l_embd[0])
			d_embeddings_1 = dic_embd(l_all_ID[1], l_embd[1])
			d_embeddings_2 = dic_embd(l_all_ID[1], l_embd[2])
			d_embeddings_5 = dic_embd(l_all_ID[1], l_embd[3])

			l_eer, l_min_dcf = sv_l([d_embeddings_all, d_embeddings_1, d_embeddings_2, d_embeddings_5], dic_eval_trial['voxceleb1_eval'], args)
			metric_man.update_metric_l(epoch = 0, l_eer = l_eer, l_min_dcf = l_min_dcf, trial_type = 'voxceleb1_eval')

		torch.cuda.empty_cache()
		dist.barrier()
		
		###############################
		#Evaluation on Vox1-E, H trial#
		###############################
		l_embd = []
		l_all_ID = []

		#extract speaker embedding
		l_embedding, l_ID = extract_speaker_embedding(
			model = model,
			db_gen = evalset_gen_vox1_all, 
			gpu = gpu)
			
		dist.barrier()

		#gather utterance ID and speaker embedding to gpu 0
		l_all_ID.append(gather(l_ID))
		l_embd.append(gather(l_embedding))
		
		if gpu == 0:
			#convert list embd to dic embd
			d_embeddings_all = dic_embd(l_all_ID[0], l_embd[0])
			for eval_trial in dic_eval_trial.keys():
				if eval_trial != 'voxceleb1_eval':
					print(eval_trial)
					eer, min_dcf = sv(d_embeddings_all, dic_eval_trial[eval_trial], args)
					metric_man.update_metric(epoch = 0, eer = eer, min_dcf = min_dcf, trial_type = eval_trial)
			metric_man.f_result.close()

	#################
	#Train the model#
	#################
	else:		
		criterion = {}
		#set ojbective funtions, optimizer and lr scheduler
		if args.use_metric_l:
			if args.metric_l == 'ge2e': criterion['metric_l'] = GE2E(gpu = gpu).cuda(gpu)
			elif args.metric_l == 'proto': criterion['metric_l'] = Prototypical(gpu = gpu).cuda(gpu)
			elif args.metric_l == 'apro': criterion['metric_l'] = AngleProto(gpu = gpu).cuda(gpu)
			else: criterion['metric_l'] = None
		if args.use_clf_l:
			if args.clf_l == 'softmax': criterion['clf_l'] = Softmax(nOut = args.model['code_dim'], nClasses = args.model['nb_spk']).cuda(gpu)
			elif args.clf_l == 'focal': criterion['clf_l'] = FocalLoss(nOut = args.model['code_dim'], nClasses = args.model['nb_spk'], alpha = args.focal_alpha, gamma = args.focal_gamma, trainable= False).cuda(gpu) 
			elif args.clf_l == 'am': criterion['clf_l'] = AMSoftmax(nOut = args.model['code_dim'], nClasses = args.model['nb_spk'], margin=args.loss_margin, scale=args.loss_scale, gpu = gpu).cuda(gpu)
			elif args.clf_l == 'aam': criterion['clf_l'] = AAMSoftmax(nOut = args.model['code_dim'], nClasses = args.model['nb_spk'], margin=args.loss_margin, scale=args.loss_scale, gpu = gpu).cuda(gpu)
			else: criterion['clf_l'] = None
		
		optimizer, lr_scheduler = get_optimizer(args, model, criterion)
		
		if args.use_metric_l and not(args.use_clf_l): 
			[model, criterion['metric_l']], optimizer = amp.initialize([model, criterion['metric_l']], optimizer, opt_level=args.opt_level)
			criterion['metric_l'] = nn.parallel.DistributedDataParallel(criterion['metric_l'], device_ids=[gpu], output_device = gpu, find_unused_parameters=True)
		elif args.use_clf_l and not(args.use_metric_l): 
			[model, criterion['clf_l']], optimizer = amp.initialize([model, criterion['clf_l']], optimizer, opt_level=args.opt_level)
			criterion['clf_l'] = nn.parallel.DistributedDataParallel(criterion['clf_l'], device_ids=[gpu], output_device = gpu, find_unused_parameters=True)
		elif args.use_clf_l and args.use_metric_l: 
			[model, criterion['clf_l'], criterion['metric_l']], optimizer = amp.initialize([model, criterion['clf_l'], criterion['metric_l']], optimizer, opt_level=args.opt_level)
			criterion['clf_l'] = nn.parallel.DistributedDataParallel(criterion['clf_l'], device_ids=[gpu], output_device = gpu, find_unused_parameters=True)
			criterion['metric_l'] = nn.parallel.DistributedDataParallel(criterion['metric_l'], device_ids=[gpu], output_device = gpu, find_unused_parameters=True)
		
		model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device = gpu, find_unused_parameters=True)


		for epoch in tqdm(range(args.epoch)):
			####################
			#Training the model#
			####################	
			trnset_sampler.set_epoch(epoch)

			train_model(model = model,
				db_gen = trnset_gen,
				args = args,
				optimizer = optimizer,
				lr_scheduler = lr_scheduler,
				criterion = criterion,
				gpu = gpu,
				epoch = epoch
				)
			torch.cuda.empty_cache()
			dist.barrier()
			
			###########################################################
			#Evaluation for various-duration utterance on Vox1-O trial#
			###########################################################
			l_embd = []
			l_all_ID = []

			for i, evalset_gen in enumerate(l_evalset_gen):
				#extract speaker embedding
				l_embedding, l_ID = extract_speaker_embedding(
					model = model,
					db_gen = evalset_gen, 
					gpu = gpu)
					
				dist.barrier()
				
				#gather utterance ID and speaker embedding to gpu 0
				l_all_ID.append(gather(l_ID))
				l_embd.append(gather(l_embedding))
				
			if gpu == 0:
				#convert list embd to dic embd
				d_embeddings_all = dic_embd(l_all_ID[0], l_embd[0])
				d_embeddings_1 = dic_embd(l_all_ID[1], l_embd[1])
				d_embeddings_2 = dic_embd(l_all_ID[1], l_embd[2])
				d_embeddings_5 = dic_embd(l_all_ID[1], l_embd[3])

				l_eer, l_min_dcf = sv_l([d_embeddings_all, d_embeddings_1, d_embeddings_2, d_embeddings_5], dic_eval_trial['voxceleb1_eval'], args)
				metric_man.update_metric_l(epoch = epoch, l_eer = l_eer, l_min_dcf = l_min_dcf, trial_type = 'voxceleb1_eval')

			torch.cuda.empty_cache()
			dist.barrier()
			
			###############################
			#Evaluation on Vox1-E, H trial#
			###############################
			l_embd = []
			l_all_ID = []

			#extract speaker embedding
			l_embedding, l_ID = extract_speaker_embedding(
				model = model,
				db_gen = evalset_gen_vox1_all, 
				gpu = gpu)
				
			dist.barrier()

			#gather utterance ID and speaker embedding to gpu 0
			l_all_ID.append(gather(l_ID))
			l_embd.append(gather(l_embedding))
			
			if gpu == 0:
				#convert list embd to dic embd
				d_embeddings_all = dic_embd(l_all_ID[0], l_embd[0])
				for eval_trial in dic_eval_trial.keys():
					if eval_trial != 'voxceleb1_eval':
						print(eval_trial)
						eer, min_dcf = sv(d_embeddings_all, dic_eval_trial[eval_trial], args)
						metric_man.update_metric(epoch = 0, eer = eer, min_dcf = min_dcf, trial_type = eval_trial)
				metric_man.f_result.close()

if __name__ == '__main__':
	main()
