#######################################
# Train the RawNeXt model with softmax#
#######################################

args=(
    -name RawNeXt -bs 40 -nb_utt_per_spk 2
    -use_clf_l T -clf_l softmax
    -module_name model_RawNeXt -model_name get_RawNeXt
    )
python ../main.py "${args[@]}"

##################################################################################
# Train the RawNeXt model with aam-softmax, AngleProto loss and data augmentation#
##################################################################################

# args=(
#     -name RawNeXt_aam_ap -bs 40 -nb_utt_per_spk 2 -augment t
#     -use_clf_l T -clf_l aam -use_metric_l T -metric_l apro -loss_scale 32 -loss_margin .25
#     -module_name model_RawNeXt -model_name get_RawNeXt
#     )
# python ../main.py "${args[@]}"


#######################################
# Train the ResNeXt model with softmax#
#######################################

# args=(
#     -name ResNeXt -bs 80 -nb_utt_per_spk 1 
#     -use_clf_l T -clf_l softmax 
#     -module_name model_ResNeXt -model_name get_ResNeXt 
#     -m_dsp F -m_up_path F -m_gate F
#     )
# python ../main.py "${args[@]}"
