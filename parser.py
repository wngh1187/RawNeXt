import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    #essential
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument("-module_name", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True)

    #dir 
    parser.add_argument('-save_dir', type = str, default = '/source/ECAPA_RawNet2/')
    parser.add_argument('-DB_vox2', type = str, default = '/DB2/VoxCeleb2/')
    parser.add_argument('-DB_vox1_all', type = str, default = '/DB2/VoxCeleb1/vox1_wav')
    parser.add_argument('-DB_vox1_eval', type = str, default = '/DB2/VoxCeleb1/vox1_test_wav')
    parser.add_argument('-trial_path', type = str, default = '/DB2/VoxCeleb1/vox1_trials')
    parser.add_argument('-musan_path', type=str,   default="/DB2/musan_split/")
    parser.add_argument('-rir_path',  type=str,   default="/DB2/rir_noises/simulated_rirs/")
    parser.add_argument('-load_model_path', type=str, default="/source/ECAPA_RawNet2/Rawnet2_DLA_013_correct_aam2/weights/checkpoint_69.00_0.0128.pt")

    #hyper-params
    parser.add_argument('-bs', type = int, default = 60)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-wd', type = float, default = 0.0001)
    parser.add_argument("-nb_samp", type=int, default=59049)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-opt_mom', type = float, default = .9)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_eval_utt', type = int, default = 10) 
    parser.add_argument('-nb_proc_eval', type = int, default = 16) 
    parser.add_argument('-cos_eta', type = float, default = 1e-7) 
    parser.add_argument('-lrdec_t0', type = int, default = 80)  
    parser.add_argument('-nb_iter_per_log', type = int, default = 40) 
    parser.add_argument('-lr_decay', type = str, default = 'cosine')
    parser.add_argument('-opt_level', type = str, default = 'O1')
    parser.add_argument("-nb_utt_per_spk", default=2, type=int)
    parser.add_argument("-max_seg_per_spk", default=500, type=int)

    #loss
    parser.add_argument('-clf_l', type = str, default = 'none')
    parser.add_argument('-metric_l', type = str, default = 'none')
    parser.add_argument('-loss_margin', type = float, default = .2)
    parser.add_argument('-loss_scale', type = float, default = 30)
    parser.add_argument("-focal_alpha", default=2., type=float)
    parser.add_argument("-focal_gamma", default=0.25, type=float)
    parser.add_argument("-weight_clf", default=1, type=float)
    parser.add_argument("-weight_metric", default=1, type=float)

    
    #DNN args
    parser.add_argument(
        "-m_channels", type=int, nargs="+", default=[128, 128, 256, 256, 512, 512]
    )
    parser.add_argument("-m_levels", type=int, nargs="+", default=[1, 1, 1, 2, 2, 1])
    parser.add_argument("-m_code_dim", type=int, default=512)
    parser.add_argument("-m_nb_samp", type=int, default=59049)
    parser.add_argument('-m_dsp', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-m_up_path', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-m_gate', type = str2bool, nargs='?', const=True, default = True)

    
    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-nesterov', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-use_clf_l', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-use_metric_l', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-augment', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-eval_extend', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-log_metrics', type = str2bool, nargs='?', const=True, default = False)
    
    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            args.model[k[2:]] = v
    return args