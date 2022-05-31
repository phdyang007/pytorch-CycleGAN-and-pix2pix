###############via 
# test large tile
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt #--update_mask
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav4 --name opc_oinnopcgv2_viav2_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt
#python3 test_lt.py --dataroot ./datasets/viav3 --name opc_oinnopcgv2_viav2_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --lt

#update mask
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 0 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 0
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 1
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 2
#python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 3
python3 test_lt.py --dataroot ./datasets/viav2 --name opc_oinnopcgv2_viav2_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 4


###############metal
# test large tile
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best 
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best 
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best  
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best 
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best 
#python3 test_lt.py --dataroot ./datasets/metalv5a2 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch best 



#update mask

#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 0 --epoch best
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_1 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 1 --epoch best
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_2 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 2 --epoch best
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_3 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 3 --epoch best
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_4 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 4 --epoch best

#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --netG_A oinnopcgv2 --update_mask --update_train_round 5 --epoch best


#test convergence and overfitting
#sleep 1800
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 1
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 2
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 3
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 4
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 5
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 6
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 7
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 8
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 9
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 10
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 11
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 12
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 13
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 14
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 15
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 16
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 17
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 18
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 19
#python3 test_lt.py --dataroot ./datasets/metalv5 --name opc_oinnopcgv2_metalv5_update_5 --netD n_layers --n_layers_D 3 --model opc --gpu_ids 4 --batch_size 1 --phase test --lt_phase 1 --lt --netG_A oinnopcgv2 --epoch 20