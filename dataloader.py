#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/Jungjee/RawNet/blob/master/python/RawNet2_modified/dataloader.py
# Adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py

import glob
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta

ta.set_audio_backend("sox_io")
import warnings

from scipy import signal

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils import data

from utils import *
ta.set_audio_backend("sox_io")

def get_train_loader(args, loader_args):
	
	l_trn = loader_args['dev_lines']
	d_label = loader_args['d_label']

	# define dataset generators
	trnset = Trainset(
		l_utt=l_trn,
		labels=d_label,
		nb_samp=args.nb_samp,
		base_dir=args.DB_vox2,# + args.dev_wav,
		augment=args.augment,
		musan_dir=args.musan_path,
		rir_dir=args.rir_path,
	)
	trnset_sampler = Voxceleb_sampler(
		dataset=trnset, nb_utt_per_spk=args.nb_utt_per_spk, max_seg_per_spk=args.max_seg_per_spk, batch_size=args.bs
	)
	trnset_gen = data.DataLoader(
		trnset,
		batch_size=args.bs,
		shuffle=(trnset_sampler is None),
		sampler=trnset_sampler,
		pin_memory=True,
		worker_init_fn=worker_init_fn,
		drop_last=True,
		num_workers=args.nb_worker,
	)
	return trnset_gen, trnset_sampler

def get_vox1_eval_loader_list(args, loader_args):
	
	l_vox1_eval = loader_args['vox1_eval_lines']
	l_nb_eval_samp = loader_args['list_eval_nb_sample']
	l_evlset_gen = []

	# define vox1 eval generators per nb_sample
	for ns in l_nb_eval_samp:
		evlset = EvaluationSet(
			l_utt=l_vox1_eval,
			nb_seg=args.nb_eval_utt,
			nb_samp=args.nb_samp,
			base_dir=args.DB_vox1_eval,
			nb_split = ns
		)
		evlset_sampler = torch.utils.data.DistributedSampler(evlset, shuffle=False)

		evlset_gen = data.DataLoader(
			evlset,
			batch_size=args.bs // 4,
			shuffle=False,
			pin_memory=True,
			drop_last=False,
			num_workers=args.nb_worker,
			sampler=evlset_sampler,
		)
		l_evlset_gen.append(evlset_gen)
	return l_evlset_gen

def get_vox1_all_loader(args, loader_args):
	
	l_vox1_all = loader_args['vox1_all_lines']

	# define vox1 all generators
	evlset = EvaluationSet(
		l_utt=l_vox1_all,
		nb_seg=args.nb_eval_utt,
		nb_samp=args.nb_samp,
		base_dir=args.DB_vox1_all
	)
	evlset_sampler = torch.utils.data.DistributedSampler(evlset, shuffle=False)

	evlset_gen = data.DataLoader(
		evlset,
		batch_size=args.bs // 4,
		shuffle=False,
		pin_memory=True,
		drop_last=False,
		num_workers=args.nb_worker,
		sampler=evlset_sampler,
	)

	return evlset_gen

	
class Trainset(data.Dataset):
	def __init__(
		self,
		l_utt,
		labels,
		nb_samp=59049,
		base_dir="",
		augment=False,
		musan_dir="",
		rir_dir="",
	):
		"""
		arguments:
		l_utt    :list of strings (each string: utt key)
		labels   :dictionary where key: utt key and value: label integer
		nb_samp  :integer, the number of samples in each utterance for each mini-batch
		base_dir :directory of dataset
		"""
		self.l_utt = l_utt
		self.labels = labels
		self.nb_samp = nb_samp + 1
		self.base_dir = base_dir
		self.augment = augment
		self.musan_dir = musan_dir
		self.rir_dir = rir_dir

		if augment:
			self.augment_wav = AugmentWAV(
				musan_dir=musan_dir, rir_dir=rir_dir, nb_samp=self.nb_samp
			)

		#####
		# for sampler
		self.utt_per_spk = {}
		self.revised_utts = []
		self.revised_labels = []
		for idx, line in enumerate(l_utt):
			label = self.labels[line.split("/")[4]]
			if label not in self.utt_per_spk:
				self.utt_per_spk[label] = []
			self.utt_per_spk[label].append(idx)


	def __len__(self):
		return len(self.l_utt)

	def __getitem__(self, indices):
		feats = []
		
		for idx, index in enumerate(indices):
			# get utterance id
			key = self.l_utt[index]

			# load utt
			try:
				x = ta.load(key)[0]
			except:
				raise ValueError("%s" % key)

			# adjust duration to "nb.samp" for mini-batch construction
			

			if idx == 1:	# second utterance cut by random sec(1 ~ 3.59)
				target_nb_samp = np.random.randint(low=16000 * 1, high=59049)

				if x.size(1) < target_nb_samp:
					nb_dup = int(target_nb_samp / x.size(1)) + 1
					x = torch.repeat(x, (1, nb_dup))[:, : target_nb_samp]
				elif x.size(1) > target_nb_samp:
					start_idx = np.random.randint(low=0, high=x.size(1) - target_nb_samp)
					x = x[:, start_idx : start_idx + target_nb_samp]

			nb_actual_samp = x.size(1)
			
				
			if nb_actual_samp > self.nb_samp:
				start_idx = np.random.randint(low=0, high=nb_actual_samp - self.nb_samp)
				x = x[:, start_idx : start_idx + self.nb_samp]
			elif nb_actual_samp < self.nb_samp:
				nb_dup = int(self.nb_samp / nb_actual_samp) + 1
				x = x.repeat(1, nb_dup)[:, : self.nb_samp]
			else:
				x = x

			# apply data augmentation
			if self.augment:
				augtype = random.randint(0, 5)
				if augtype == 1:
					x = self.augment_wav.reverberate(x)
				if augtype == 2:
					x = self.augment_wav.reverberate(x)
					augtype = random.randint(2, 5)
				if augtype == 3:
					x = self.augment_wav.additive_noise("music", x)
				elif augtype == 4:
					x = self.augment_wav.additive_noise("speech", x)					
				elif augtype == 5:
					x = self.augment_wav.additive_noise("noise", x)

			# apply pre-emphasis
			x = pre_emphasis(x)  # 59050 to 59049
		
			feats.append(x)
		# get label
		y = self.labels[key.split("/")[4]]

		return torch.stack(feats), y


class Voxceleb_sampler(torch.utils.data.DistributedSampler):
	"""
    Acknowledgement: Github project 'clovaai/voxceleb_trainer'.
    link: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
    Adjusted for RawNeXt
    """
	def __init__(self, dataset, nb_utt_per_spk, max_seg_per_spk, batch_size):
		# distributed settings
		if not dist.is_available():
			raise RuntimeError("Requires distributed package.")
		self.nb_replicas = dist.get_world_size()
		self.rank = dist.get_rank()
		self.epoch = 0

		# sampler config
		self.dataset = dataset
		self.utt_per_spk = dataset.utt_per_spk
		self.nb_utt_per_spk = nb_utt_per_spk
		self.max_seg_per_spk = max_seg_per_spk
		self.batch_size = batch_size
		self.nb_samples = int(
			math.ceil(len(dataset) / self.nb_replicas)
		)  
		self.total_size = (
			self.nb_samples * self.nb_replicas
		) 
		self.__iter__() 

	def __iter__(self):
		
		np.random.seed(self.epoch)

		# speaker ids
		spk_indices = np.random.permutation(list(self.utt_per_spk.keys()))

		# pair utterances by 2
		# list of list
		lol = lambda lst: [lst[i : i + self.nb_utt_per_spk] for i in range(0, len(lst), self.nb_utt_per_spk)]

		flattened_list = []
		flattened_label = []

		# Data for each class
		for findex, key in enumerate(spk_indices):
			# list, utt keys for one speaker
			utt_indices = self.utt_per_spk[key]
			# number of pairs of one speaker's utterances
			nb_seg = round_down(min(len(utt_indices), self.max_seg_per_spk), self.nb_utt_per_spk)
			# shuffle -> make to pairs
			rp = lol(np.random.permutation(len(utt_indices))[:nb_seg])
			flattened_label.extend([findex] * (len(rp)))
			for indices in rp:
				flattened_list.append([utt_indices[i] for i in indices])
		# print("a", np.array(flattened_list).shape) # a (562675, 2)
		# print("b", np.array(flattened_label).shape) # b (562675,)
		# data in random order
		mixid = np.random.permutation(len(flattened_label))
		mixlabel = []
		mixmap = []

		# prevent two pairs of the same speaker in the same batch
		for ii in mixid:
			startbatch = len(mixlabel) - (
				len(mixlabel) % (self.batch_size * self.nb_replicas)
			)
			if flattened_label[ii] not in mixlabel[startbatch:]:
				mixlabel.append(flattened_label[ii])
				mixmap.append(ii)
		it = [flattened_list[i] for i in mixmap]

		# adjust mini-batch-wise for DDP
		nb_batch, leftover = divmod(len(it), self.nb_replicas * self.batch_size)
		if leftover != 0:
			warnings.warn(
				"leftover:{} in sampler, epoch:{}, gpu:{}, cropping..".format(
					leftover, self.epoch, self.rank
				)
			)
			it = it[: self.nb_replicas * self.batch_size * nb_batch]
		_it = []
		for idx in range(
			self.rank * self.batch_size, len(it), self.nb_replicas * self.batch_size
		):
			_it.extend(it[idx : idx + self.batch_size])
		it = _it
		self._len = len(it)  # print("nb utt per GPU", self._len) # 138700 for 4GPU

		return iter(it)

	def __len__(self):
		return self._len


class EvaluationSet(data.Dataset):
	def __init__(self, l_utt, nb_seg=10, nb_samp=59049, base_dir="", nb_split=-1):
		"""
		l_utt       :list of strings (each string: utt key)
		nb_seg      :integer, the number of segments to extract from an utterance
		nb_samp     :integer, the number of samples in each utterance for each mini-batch
		base_dir    :directory of dataset
		"""
		self.l_utt = l_utt
		self.nb_seg = nb_seg
		self.nb_samp = nb_samp
		self.base_dir = base_dir
		self.nb_split = nb_split

	def __len__(self):
		return len(self.l_utt)

	def __getitem__(self, index):
		key = self.l_utt[index]
		try:
			x = ta.load(key)[0]
		except:
			raise ValueError("%s" % key)

		# apply pre-emphasis
		x = pre_emphasis(x)

		if self.nb_split>0:
			self.nb_seg = 3

			win_size = self.nb_split * 16000

			nb_actual_samp = x.size(1)

			if nb_actual_samp <= win_size:
				nb_dup = int(win_size / nb_actual_samp) + 1
				x = x.repeat(1, nb_dup)
				nb_actual_samp = x.size(1)
				
			x = x[:, nb_actual_samp//2 - win_size //2: nb_actual_samp//2 + win_size //2]


		# match minimum required duration if too short
		nb_actual_samp = x.size(1)
		if nb_actual_samp < self.nb_samp:
			nb_dup = int(self.nb_samp / nb_actual_samp) + 1
			x = x.repeat(1, nb_dup)[:, : self.nb_samp]
			nb_actual_samp = x.size(1)

		# start indices of each segment
		stt_idx = np.linspace(0, nb_actual_samp - self.nb_samp, self.nb_seg)
		
		# list of segments
		l_x = []
		for idx in stt_idx:
			l_x.append(x[:, int(idx) : int(idx) + self.nb_samp])
		
		x = torch.stack(l_x, dim=0)  # (10, self.nb_samp)
		
		return x, "/".join(key.split('/')[4:])



#####
# Pre-emphasize an utterance (single & multi-channel)
# x : (numpy array or torch tensor) shape (#channel, #sample)
def pre_emphasis(x):
	return x[:, 1:] - 0.97 * x[:, :-1]


class AugmentWAV(object):
	"""
	Acknowledgement: Github project 'clovaai/voxceleb_trainer'.
	link: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py

	Adjusted for RawNeXt
	"""

	def __init__(self, musan_dir, rir_dir, nb_samp):
		self.nb_samp = nb_samp
		self.noisetypes = ["noise", "speech", "music"]
		self.noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
		self.numnoise = {"noise": [1, 1], "speech": [3, 7], "music": [1, 1]}
		self.noiselist = {}

		augment_files = glob.glob(os.path.join(musan_dir, "*/*/*/*.wav"))
		for file in augment_files:
			if not file.split("/")[-4] in self.noiselist:
				self.noiselist[file.split("/")[-4]] = []
			self.noiselist[file.split("/")[-4]].append(file)

		self.rir_files = glob.glob(os.path.join(rir_dir, "*/*/*.wav"))

	def additive_noise(self, noisecat, audio):
		clean_db = 10 * torch.log10(torch.mean(audio ** 2) + 1e-4)

		numnoise = self.numnoise[noisecat]
		noiselist = random.sample(
			self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
		)

		noises = []
		for noise in noiselist:
			noiseaudio = ta.load(noise)[0][:, : self.nb_samp]
			noise_snr = random.uniform(
				self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
			)
			noise_db = 10 * torch.log10(torch.mean(noiseaudio[0] ** 2) + 1e-4)
			noises.append(
				torch.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
			)

		return torch.sum(torch.cat(noises, dim=0), dim=0, keepdims=True) + audio

	def reverberate(self, audio):
		rir_file = random.choice(self.rir_files)
		rir = ta.load(rir_file)[0][:, : self.nb_samp]
		rir = rir / torch.sqrt(torch.sum(rir ** 2))

		res = torch.Tensor(signal.convolve(audio, rir, mode="full")[:, : self.nb_samp])
		return res


def round_down(num, divisor):
	return num - (num % divisor)


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

