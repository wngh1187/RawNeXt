#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/clovaai/voxceleb_trainer/tree/master/loss

import numpy as np, math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(Softmax, self).__init__()

		self.test_normalize = True
		
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut, nClasses, bias=True)

		print('Initialised Softmax Loss')

	def forward(self, x, label=None):


		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		return nloss

class FocalLoss(nn.Module):
	def __init__(self, nOut, nClasses, alpha=1, gamma=2, trainable = False):
		super(FocalLoss, self).__init__()
		self.trainable = trainable
		if self.trainable: 
			self.alpha = nn.Parameter(torch.tensor(alpha))
			self.gamma = nn.Parameter(torch.tensor(gamma))
			self.alpha.requires_grad = True
			self.gamma.requires_grad = True
		else:
			self.alpha = alpha
			self.gamma = gamma

		self.fc	= nn.Linear(nOut,nClasses)
		print('Initialised Focal Loss')

	def forward(self, x, label):
		
		x = self.fc(x)
		ce_loss = nn.functional.cross_entropy(x, label, reduction='none') 
		pt = torch.exp(-ce_loss)
		nloss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
		return nloss

class AngleProto(nn.Module):

	def __init__(self, gpu, init_w=10.0, init_b=-5.0, **kwargs):
		super(AngleProto, self).__init__()

		self.gpu = gpu
		self.w = nn.Parameter(torch.tensor(init_w))
		self.b = nn.Parameter(torch.tensor(init_b))
		self.w.requires_grad = True
		self.b.requires_grad = True 
		self.cce = torch.nn.CrossEntropyLoss().cuda(gpu)

		print('Initialised AngleProto')

	def forward(self, x):
		
		assert x.size()[1] >= 2

		out_anchor      = torch.mean(x[:,1:,:],1)
		out_positive    = x[:,0,:]
		stepsize        = out_anchor.size()[0]

		cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
		torch.clamp(self.w, 1e-6)
		cos_sim_matrix = cos_sim_matrix * self.w + self.b
		
		label       = torch.from_numpy(np.asarray(range(0,stepsize))).cuda(self.gpu)
		criterion = self.cce
		nloss       = criterion(cos_sim_matrix, label)

		return nloss

class Prototypical(nn.Module):

	def __init__(self, gpu, **kwargs):
		super(Prototypical, self).__init__()

		self.gpu = gpu
		self.criterion  = torch.nn.CrossEntropyLoss().cuda(gpu)

		print('Initialised Prototypical Loss')

	def forward(self, x, label=None):

		assert np.shape(x)[1] >= 2 #(num_speakers, num_utts_per_speaker, dvec_feats)
		
		out_anchor      = torch.mean(x[:,1:,:],1)   # after 0th utt per speaker
		out_positive    = x[:,0,:]  #0th utt per speaker
		stepsize        = out_anchor.size()[0]

		output      = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
		label       = torch.from_numpy(np.asarray(range(0,stepsize))).cuda(self.gpu)
		nloss       = self.criterion(output, label)

		return nloss

class GE2E(nn.Module):

	def __init__(self, gpu, init_w=10.0, init_b=-5.0, **kwargs):
		super(GE2E, self).__init__()

		self.gpu = gpu
		self.w = nn.Parameter(torch.tensor(init_w))
		self.b = nn.Parameter(torch.tensor(init_b))
		self.w.requires_grad = True
		self.b.requires_grad = True
		self.criterion  = torch.nn.CrossEntropyLoss().cuda(gpu)
		print('Initialised GE2E')

	def forward(self, x, label=None):

		assert x.size()[1] >= 2 #(num_speakers, num_utts_per_speaker, dvec_feats)

		gsize = x.size()[1]
		centroids = torch.mean(x, 1)
		stepsize = x.size()[0]

		cos_sim_matrix = []
 
		for ii in range(0,gsize): 
			idx = [*range(0,gsize)]
			idx.remove(ii)
			exc_centroids = torch.mean(x[:,idx,:], 1)   #except ii th utterance centeroid
			cos_sim_diag    = F.cosine_similarity(x[:,ii,:],exc_centroids) # k = j, similarity
			cos_sim         = F.cosine_similarity(x[:,ii,:].unsqueeze(-1),centroids.unsqueeze(-1).transpose(0,2))
			cos_sim[range(0,stepsize),range(0,stepsize)] = cos_sim_diag
			cos_sim_matrix.append(torch.clamp(cos_sim,1e-6))

		cos_sim_matrix = torch.stack(cos_sim_matrix,dim=1)

		torch.clamp(self.w, 1e-6)
		cos_sim_matrix = cos_sim_matrix * self.w + self.b
		
		label = torch.from_numpy(np.asarray(range(0,stepsize))).cuda(self.gpu)
		nloss = self.criterion(cos_sim_matrix.view(-1,stepsize), torch.repeat_interleave(label,repeats=gsize,dim=0).cuda(self.gpu))
  
		return nloss

class AMSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(AMSoftmax, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss    = self.ce(costh_m_s, label)
        
        return loss

class AAMSoftmax(nn.Module):
	def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
		super(AAMSoftmax, self).__init__()

		self.test_normalize = True
		
		self.m = margin
		self.s = scale
		self.in_feats = nOut
		self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
		self.ce = nn.CrossEntropyLoss()
		nn.init.xavier_normal_(self.weight, gain=1)

		self.easy_margin = easy_margin
		self.cos_m = math.cos(self.m)
		self.sin_m = math.sin(self.m)

		# make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
		self.th = math.cos(math.pi - self.m)
		self.mm = math.sin(math.pi - self.m) * self.m

		print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

	def forward(self, x, label=None):
		code_dim = x.size()[-1]
		x = x.reshape(-1,code_dim)
	
		assert x.size()[0] == label.size()[0]
		assert x.size()[1] == self.in_feats
		
		# cos(theta)
		cosine = F.linear(F.normalize(x), F.normalize(self.weight))
		# cos(theta + m)
		sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m

		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

		#one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, label.view(-1, 1), 1)
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output = output * self.s

		loss    = self.ce(output, label)

		return loss
