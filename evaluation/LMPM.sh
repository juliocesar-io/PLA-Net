WEIGHTS_PATHS='LMPM'
DEVICE=0
python ensamble.py --device $DEVICE --batch_size 60 --save $WEIGHTS_PATHS --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary --use_prot

