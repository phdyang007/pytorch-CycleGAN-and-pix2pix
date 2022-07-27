#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 2000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 2000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 2000 --aug_iter 1

#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 4000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 4000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 4000 --aug_iter 1

#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 8000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 8000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 8000 --aug_iter 1

# Pretrain Model
#python train.py --dataroot ./../mask_all/ --name LithoParaAug --netF oinnopc_parallel --batch_size 16

# Train stylegan
#python train_stylegan.py --dataroot ./../mask_all --name LithoParaAug --load_iter 11 --continue_train --augmode random --rank_buffer_size 200 --aug_iter 10 --netF oinnopc_parallel --batch_size 16

#python train_generate.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode random --rank_buffer_size 2000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16 
#python train_model.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode random --rank_buffer_size 2000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16 

#python train_generate.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 2000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16  
#python train_model.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 2000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16  

python train_generate.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 8000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16 --style_loss_type predict
python train_model.py --dataroot ./../mask_all/ --name LithoParaAug --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 8000 --aug_iter 4 --netF oinnopc_parallel --batch_size 16 

cp ./results.txt /mingjiel_mnt/results_07_26.txt
