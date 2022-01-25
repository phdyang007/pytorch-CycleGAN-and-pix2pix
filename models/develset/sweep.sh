python=/home/guojin/miniconda3/envs/hdevel/bin/python


$python train.py -m \
model.loss_weights.ilt_weight=0.5,1.0,1.5 \
model.loss_weights.pvb_weight=0.1,0.5,1,3,5,7 \
model.ilt_exp=2,4 \
model.add_curv=True,False

