###############via 
# test large tile
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt #--update_mask
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt

#update mask
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 0
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 1
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 2
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 3


###############metal
#python3 test_lt.py --dataroot ./datasets/metalv3 --name opc_oinnopcgv2_metalv4_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt #--update_mask
#python3 test_lt.py --dataroot ./datasets/metalv1 --name opc_oinnopcg_metalv1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcg --update_mask --update_train_round 0
#python3 test_lt.py --dataroot ./datasets/metalv4 --name opc_oinnopcgv2_metalv4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 0



#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 #--update_mask --update_train_round 0