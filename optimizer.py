import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR

from utils import *

class GradualWarmupScheduler(_LRScheduler):
	""" Gradually warm-up(increasing) learning rate in optimizer.
	Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
	Args:
		optimizer (Optimizer): Wrapped optimizer.
		multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
		total_epoch: target learning rate is reached at total_epoch, gradually
		after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
	"""

	def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
		self.multiplier = multiplier
		if self.multiplier < 1.:
			raise ValueError('multiplier should be greater thant or equal to 1.')
		self.total_epoch = total_epoch
		self.after_scheduler = after_scheduler
		self.finished = False
		super(GradualWarmupScheduler, self).__init__(optimizer)

	def get_lr(self):
		if self.last_epoch > self.total_epoch:
			if self.after_scheduler:
				if not self.finished:
					self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
					self.finished = True
				return self.after_scheduler.get_last_lr()
			return [base_lr * self.multiplier for base_lr in self.base_lrs]

		if self.multiplier == 1.0:
			return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
		else:
			return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

	def step_ReduceLROnPlateau(self, metrics, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
		self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
		if self.last_epoch <= self.total_epoch:
			warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
			for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
				param_group['lr'] = lr
		else:
			if epoch is None:
				self.after_scheduler.step(metrics, None)
			else:
				self.after_scheduler.step(metrics, epoch - self.total_epoch)

	def step(self, epoch=None, metrics=None):
		if type(self.after_scheduler) != ReduceLROnPlateau:
			if self.finished and self.after_scheduler:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.total_epoch)
				self._last_lr = self.after_scheduler.get_last_lr()
			else:
				return super(GradualWarmupScheduler, self).step(epoch)
		else:
			self.step_ReduceLROnPlateau(metrics, epoch)

def keras_lr_decay(step, decay = 0.0001):
	return 1./(1. + decay * step)


class PolynomialLRDecay(_LRScheduler):
	"""Polynomial learning rate decay until step reach to max_decay_step
	
	Args:
		optimizer (Optimizer): Wrapped optimizer.
		max_decay_steps: after this step, we stop decreasing learning rate
		end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
		power: The power of the polynomial.
	"""
	
	def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
		if max_decay_steps <= 1.:
			raise ValueError('max_decay_steps should be greater than 1.')
		self.max_decay_steps = max_decay_steps
		self.end_learning_rate = end_learning_rate
		self.power = power
		self.last_step = 0
		super().__init__(optimizer)
		
	def get_lr(self):
		if self.last_step > self.max_decay_steps:
			return [self.end_learning_rate for _ in self.base_lrs]

		return [(base_lr - self.end_learning_rate) * 
				((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
				self.end_learning_rate for base_lr in self.base_lrs]
	
	def step(self, step=None):
		if step is None:
			step = self.last_step + 1
		self.last_step = step if step != 0 else 1
		if self.last_step <= self.max_decay_steps:
			decay_lrs = [(base_lr - self.end_learning_rate) * 
						 ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
						 self.end_learning_rate for base_lr in self.base_lrs]
			for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
				param_group['lr'] = lr
		self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def get_optimizer(args, model, criterion):

	params = [
		{
			"params": [
				param for name, param in model.named_parameters() if "bn" not in name
			]
		},
		{
			"params": [
				param for name, param in model.named_parameters() if "bn" in name
			],
			"weight_decay": 0,
		},
	]

	for cri in criterion.keys():
		params += [
			{
				"params": [
					param for name, param in criterion[cri].named_parameters()
				]
			},
		]

	if args.optimizer.lower() == 'sgd':
		optimizer = torch.optim.SGD(
			params,
			lr = args.lr,
			momentum = args.opt_mom,
			weight_decay = args.wd,
			nesterov = args.nesterov
			)
	elif args.optimizer.lower() == 'adam':
		optimizer = torch.optim.Adam(
			params,
			lr = args.lr,
			weight_decay = args.wd,
			amsgrad = args.amsgrad
			)
	else:
		raise NotImplementedError('Add other optimizers if needed')

	#set learning rate decay
	lr_scheduler = None
	if bool(args.do_lr_decay):
		if args.lr_decay == 'keras':
			lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
		elif args.lr_decay == 'cosine':
			lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = args.nb_iter * args.lrdec_t0, eta_min = 0.000001)
		elif args.lr_decay == 'warmup':
			scheduler_steplr = CosineAnnealingWarmRestarts(optimizer, T_0 = args.nb_iter * args.lrdec_t0, eta_min = args.cos_eta)
			lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch= args.nb_iter * 3, after_scheduler=scheduler_steplr)
			optimizer.zero_grad()
			optimizer.step()
		elif args.lr_decay == 'poly':
			lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(args.iter * args.ngpus_per_node * args.lrdec_t0), end_learning_rate=0.0001, power=2.0)
		elif args.lr_decay == 'cyclic':
			lr_scheduler = CyclicLR(optimizer, base_lr = args.cos_eta, max_lr = args.lr, step_size_up=args.nb_iter * int(args.lrdec_t0 *0.3), step_size_down=args.nb_iter * int(args.lrdec_t0 *0.7), mode='triangular2', cycle_momentum = False)
		elif args.lr_decay == 'onecycle':
			lr_scheduler = {}
			for i in range(args.epoch // args.lrdec_t0):
				lr_scheduler[i] = OneCycleLR(optimizer, args.lr, epochs=args.lrdec_t0, steps_per_epoch=args.nb_iter, cycle_momentum = False)
		else:
			raise NotImplementedError('Not implemented yet')

		 
	return optimizer, lr_scheduler