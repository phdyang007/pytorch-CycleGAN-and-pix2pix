# pretrain model
python train.py --dataroot ./mask_all/ --name Seed42_style --random_seed 42

# non iterative mode on style
# generate data
python generate.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_style --adv_loss_type houdini --random_seed 42 --load_iter initial --continue_train
# retrain model
python train_model.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_style --adv_loss_type houdini --random_seed 42 --load_iter initial --continue_train

# iterative mode on style
python train_generate_model.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_style --adv_loss_type houdini --random_seed 42 --load_iter initial --continue_train

# iterative mode TOD on style
python train_generative_model_TOD.py --dataroot ./mask_all/ --name Seed42_style --augmode adv_style --adv_loss_type TOD --random_seed 42 --load_iter initial --continue_train
