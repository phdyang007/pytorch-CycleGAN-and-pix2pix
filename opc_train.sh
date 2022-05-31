####################via
#python3 train_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 3
#python3 train_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --use_update_mask --lr_decay_iters 3 --update_train_round 1
#python3 train_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --use_update_mask --lr_decay_iters 3 --update_train_round 2
#python3 train_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --use_update_mask --lr_decay_iters 3 --update_train_round 3
#python3 train_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --use_update_mask --lr_decay_iters 3 --update_train_round 4



###################metal


#metalv5
#python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2
#python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2 --use_update_mask --update_train_round 1
#python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2 --use_update_mask --update_train_round 2
#python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2 --use_update_mask --update_train_round 3
#python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2 --use_update_mask --update_train_round 4
python3 train_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 16 --pool_size 25 --netG_A oinnopcgv2 --lr_policy step --lr_decay_iters 2 --use_update_mask --update_train_round 5


