# pretrain model
python train.py --dataroot ./mask_all/ --name Seed42_style --random_seed 42 --n_epochs 10

# non iterative mode on style
# generate data
python train_generate.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_noise --adv_loss_type pixel --random_seed 42 --load_iter initial --continue_train
# retrain model
python train_model.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_style --adv_loss_type pixel --random_seed 42 --load_iter initial --continue_train

# iterative mode on style
python train_generate_model.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_noise --adv_loss_type pixel --random_seed 42 --load_iter initial --continue_train

# iterative mode TOD on style
python train_generate_model_TOD.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_noise --adv_loss_type TOD --random_seed 42 --load_iter initial --continue_train
