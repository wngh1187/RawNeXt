import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from utils import *
from multiprocessing import Pool
from apex import amp
from ddp_utils import *

def train_model(model, db_gen, optimizer, epoch, args, lr_scheduler, criterion, gpu):
	model.train()
	
	if args.use_clf_l: criterion['clf_l'].train()
	if args.use_metric_l: criterion['metric_l'].train()
  
	_loss = 0.
	if args.use_clf_l: _loss_clf = 0.
	if args.use_metric_l: _loss_metric = 0.

	with tqdm(total = len(db_gen), ncols = 150) as pbar:
		for idx, (m_batch, m_label) in enumerate(db_gen):
			loss = 0
			optimizer.zero_grad()
			
			m_label = m_label.view(-1, 1).repeat(1, args.nb_utt_per_spk).view(-1).cuda(gpu)
			m_batch = m_batch.cuda(gpu, non_blocking=True)
			m_batch = m_batch.view(-1, 1, m_batch.size()[-1])
			code = model(m_batch)	
		
			code_reshape = code.view(m_batch.size()[0], args.nb_utt_per_spk, -1)

			description = '%s epoch: %d '%(args.name, epoch)
			
			if args.use_clf_l:
				loss_clf = criterion['clf_l'](code, m_label)
				loss += args.weight_clf * loss_clf
				_loss_clf += loss_clf.cpu().detach() 
				description += 'loss_clf:%.3f '%(loss_clf)

			if args.use_metric_l:
				all_embeddings = torch.cat(GatherLayer.apply(code_reshape), dim=0)  
				loss_metric = criterion['metric_l'](all_embeddings)
				loss += args.weight_metric * loss_metric
				_loss_metric += loss_metric.cpu().detach() 
				description += 'loss_metric: %.3f '%(loss_metric)

			with amp.scale_loss(loss, optimizer) as loss_scaled:
				loss_scaled.backward()

			_loss += loss.cpu().detach()
			optimizer.step()

			description += 'TOT: %.4f'%(loss)
			pbar.set_description(description)
			pbar.update(1)
			
			if idx % args.nb_iter_per_log == 0:
				if idx != 0:
					_loss /= args.nb_iter_per_log
					if args.use_metric_l:_loss_metric /= args.nb_iter_per_log
					if args.use_clf_l: _loss_clf /= args.nb_iter_per_log
				if gpu == 0:	
					_loss = 0.
					if args.use_clf_l:
						_loss_clf = 0.
					if args.use_metric_l:
						_loss_metric = 0.
			if args.do_lr_decay:
				lr_scheduler.step()

def extract_speaker_embedding(model, db_gen, gpu):
	model.eval()
	with torch.set_grad_enabled(False):
		l_embeddings = []
		l_ID = []
  
		with tqdm(total = len(db_gen), ncols = 70) as pbar:
			for (m_batch, ID) in db_gen:
				
				nb_eval_utt = m_batch.shape[1]
				m_batch = m_batch.cuda(gpu, non_blocking=True)
				m_batch = m_batch.reshape(-1,1,m_batch.size(-1))
		
				code = model(x = m_batch, is_test = True)
				if m_batch.size(-1) == 59049:
					for i in range(int(code.size(0)/nb_eval_utt)):        
						l_code = []
						for j in range(nb_eval_utt):  
							l_code.append(code[j+i*nb_eval_utt].cpu().numpy())
						l_embeddings.append(np.mean(l_code, axis=0))
						
				else: l_embeddings.extend(code.cpu().numpy())
				l_ID.extend(ID)
				pbar.update(1)

		return l_embeddings, l_ID

		
def sv(d_embeddings, l_eval_trial, args):

	y_score = [] # score for each sample
	y = [] # label for each sample 
	
	l_trial_split = split_list(
	l_in = l_eval_trial,
	nb_split = args.nb_proc_eval,
	d_embeddings = d_embeddings
	)
	p = Pool(args.nb_proc_eval)
	res = p.map(_sp_process_trial, l_trial_split)
	for _y, _y_s in res:
		y.extend(_y)
		y_score.extend(_y_s)
	p.close()
	p.join()
	fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
	
	fnr = 1 - tpr
	eer = get_eer(fnr, fpr)
	min_dcf = get_min_dcf(fpr, fnr)
	
	return eer, min_dcf

def sv_l(d_embeddings, l_eval_trial, args):
	d_embeddings_all = d_embeddings[0]
	d_embeddings_1 = d_embeddings[1]
	d_embeddings_2 = d_embeddings[2]
	d_embeddings_5 = d_embeddings[3]
	
	#2nd, calculate EER
	y, y_score_org, y_score_1, y_score_2, y_score_5 = [], [], [], [], []

	l_trial_split = split_list(
	l_in = l_eval_trial,
	nb_split = args.nb_proc_eval,
	d_embeddings = [d_embeddings_all, d_embeddings_1, d_embeddings_2, d_embeddings_5]
	)
	p = Pool(args.nb_proc_eval)
	res = p.map(_sp_process_trial_l, l_trial_split)
	for _y, _y_s_org, _y_s_1, _y_s_2, _y_s_5 in res:
		y.extend(_y)
		y_score_org.extend(_y_s_org)
		y_score_1.extend(_y_s_1)
		y_score_2.extend(_y_s_2)
		y_score_5.extend(_y_s_5)

	ys = [y_score_org, y_score_1, y_score_2, y_score_5]

	l_eer, l_min_dcf = [], []
	for y_s in ys:
		fpr, tpr, thresholds = roc_curve(y, y_s, pos_label=1)
		fnr = 1 - tpr
		p.close()
		p.join()
		l_eer.append(get_eer(fnr, fpr))
		l_min_dcf.append(get_min_dcf(fpr, fnr))

	return l_eer, l_min_dcf

def split_list(l_in, nb_split, d_embeddings, drop_leftover = False):
	nb_per_split = int(len(l_in) / nb_split)
	l_return = []
	for i in range(nb_split):
		l_return.append([l_in[i*nb_per_split:(i+1)*nb_per_split], d_embeddings])
	if not drop_leftover:
		l_return[-1][0].extend(l_in[nb_split*nb_per_split:])

	return l_return

def _sp_process_trial(args):
	l_trial, d_embeddings = args
	y, y_score = [], []
	for line in l_trial:
		trg, utt_a, utt_b = line.strip().split(' ')
		y.append(int(trg))
		y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
	return y, y_score

def _sp_process_trial_l(args):
	l_trial, ld = args
	d_embeddings_all = ld[0]
	d_embeddings_1 = ld[1]
	d_embeddings_2 = ld[2]
	d_embeddings_5 = ld[3]
	y, y_score_org, y_score_1, y_score_2, y_score_5 = [], [], [], [], []
	for line in l_trial:
		trg, utt_a, utt_b = line.strip().split(' ')
		y.append(int(trg))
		y_score_org.append(cos_sim(d_embeddings_all[utt_a], d_embeddings_all[utt_b]))
		y_score_1.append(cos_sim(d_embeddings_all[utt_a], d_embeddings_1[utt_b]))
		y_score_2.append(cos_sim(d_embeddings_all[utt_a], d_embeddings_2[utt_b]))
		y_score_5.append(cos_sim(d_embeddings_all[utt_a], d_embeddings_5[utt_b]))
	return y, y_score_org, y_score_1, y_score_2, y_score_5
