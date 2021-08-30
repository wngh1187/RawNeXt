import torch

class metric_manager(object):
    def __init__(self, save_dir, model, dic_eval_trial, save_best_only=True):
     
        self.save_dir = save_dir
        self.model = model
        self.save_best_only = save_best_only

        self.best_eer = {}
        self.best_min_dcf = {}
        for key in dic_eval_trial.keys():
            self.best_eer[key] = 99.
            self.best_min_dcf[key] = 99.

        self.f_result = open(save_dir + 'results.txt', 'a', buffering = 1)
        
    def update_metric_l(self, epoch, l_eer, l_min_dcf, trial_type):
        print('\nepoch:%d, %s, eval_eer_org:%.4f, eval_min_dcf_org:%.4f, eval_eer_1:%.4f, eval_min_dcf_1:%.4f, eval_eer_2:%.4f, eval_min_dcf_2:%.4f, eval_eer_5:%.4f, eval_min_dcf_5:%.4f\n'\
        %(epoch, trial_type, l_eer[0], l_min_dcf[0], l_eer[1], l_min_dcf[1], l_eer[2], l_min_dcf[2], l_eer[3], l_min_dcf[3]))
        self.f_result.write('epoch:%d, %s, eval_eer_org:%.4f, eval_min_dcf_org:%.4f, eval_eer_1:%.4f, eval_min_dcf_1:%.4f, eval_eer_2:%.4f, eval_min_dcf_2:%.4f, eval_eer_5:%.4f, eval_min_dcf_5:%.4f\n'\
        %(epoch, trial_type, l_eer[0], l_min_dcf[0], l_eer[1], l_min_dcf[1], l_eer[2], l_min_dcf[2], l_eer[3], l_min_dcf[3]))
        
        #record best validation model
        if self.best_eer[trial_type] > l_eer[0]:
            print('New best eer %s: %f'%(trial_type, float(l_eer[0])))
            self.best_eer[trial_type] = l_eer[0]
            if self.save_best_only:
                checkpoint = {'model': self.model.state_dict()}            
                torch.save(checkpoint, self.save_dir +  'weights/checkpoint_best.pt')

        if self.best_min_dcf[trial_type] > l_min_dcf[0]:
            print('New best mindcf %s: %f'%(trial_type, float(l_min_dcf[0])))
            self.best_min_dcf[trial_type] = l_min_dcf[0]

        if not self.save_best_only:
            checkpoint = {'model': self.model.state_dict()} 
            torch.save(checkpoint, self.save_dir +  'weights/checkpoint_%.2f_%.4f.pt'%(epoch, l_eer[0]))

    def update_metric(self, epoch, eer, min_dcf, trial_type):
        print('\nepoch:%d, %s, eval_eer:%.4f, eval_min_dcf:%.4f\n'%(epoch, trial_type, eer, min_dcf))
        self.f_result.write('epoch:%d, %s, eval_eer:%.4f, eval_min_dcf:%.4f\n'%(epoch, trial_type, eer, min_dcf))

        #record best validation model
        if self.best_eer[trial_type] > eer:
            print('New best eer %s: %f'%(trial_type, float(eer)))
            self.best_eer[trial_type] = eer
        if self.best_min_dcf[trial_type] > min_dcf:
            print('New best mindcf %s: %f'%(trial_type, float(min_dcf)))
            self.best_min_dcf[trial_type] = min_dcf

