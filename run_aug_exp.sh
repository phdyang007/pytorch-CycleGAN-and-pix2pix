#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 2000 --aug_iter 1
#python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 2000 --aug_iter 1
python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 2000 --aug_iter 1

python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 4000 --aug_iter 1
python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 4000 --aug_iter 1
python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 4000 --aug_iter 1

python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode random --rank_buffer_size 8000 --aug_iter 1
python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_style --rank_buffer_size 8000 --aug_iter 1
python train_iter.py --dataroot ./../mask_all/ --name LithoAll --load_iter 11 --continue_train --augmode adv_noise --rank_buffer_size 8000 --aug_iter 1