######################################
# evaluate the trained RawNeXt model #
######################################

args=(
    -name RawNeXt -bs 80 -eval T -load_model T -load_model_path '../weights/RawNeXt_aam.pt'
    -module_name model_RawNeXt -model_name get_RawNeXt
    )
python ../main.py "${args[@]}"


######################################
# evaluate the trained ResNeXt model #
######################################

# args=(
#     -name ResNeXt -bs 80 -eval T -load_model T -load_model_path '../weights/ResNeXt.pt'
#     -module_name model_ResNeXt -model_name get_ResNeXt 
#     -m_dsp F -m_up_path F -m_gate F -do_lr_decay F 
#     )
# python ../main.py "${args[@]}"
