args=(
    -name RawNeXt -bs 10 -epoch 1 -augment t
    -use_clf_l T -clf_l aam -loss_scale 32 -loss_margin .25 -use_metric_l t -metric_l apro
    -module_name model_RawNeXt -model_name get_RawNeXt -load_model F
    )
python ../main.py "${args[@]}"


args=(
    -name ResNeXt -bs 4 -epoch 1 -augment t
    -use_clf_l T -clf_l aam -loss_scale 32 -loss_margin .25 -use_metric_l f 
    -module_name model_ResNeXt -model_name get_ResNeXt 
    -m_dsp F -m_up_path F -m_gate F
    )
python ../main.py "${args[@]}"
