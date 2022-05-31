"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from datetime import datetime
import numpy as np 
import cv2

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    datanum = len(dataset)

    model = create_model(opt)      # create a model given opt.model and other options
    if opt.lt_phase==1:
        model.save_dir = model.save_dir_pre
    else:
        model.save_dir = model.save_dir_post
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    #load golden results
    import numpy as np 
    import timeit


    designs = []
    result = np.zeros((datanum, 10)).astype(float)
    
    
    name=    open(os.path.join(opt.results_dir, opt.name, "name.txt"), 'w')
    total_data_count  = 0
    model_win_count = 0
    for i, data in enumerate(dataset):
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #    break
        model.set_input(data)  # unpack data from data loader
        start = timeit.default_timer()
        model.test()           # run inference
        end = timeit.default_timer()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        #if i % 5 == 0:  # save images to an HTML file
        print('processing (%04d)-th image... %s, runtime: %f' % (i, img_path[0], end-start))
        total_data_count+=1

        if opt.dump_imask:
            name.write(os.path.basename(img_path[0]))
            name.write("\n")
            result[i,0] = model.g_l2.cpu().detach().numpy()
            result[i,1] = model.g_pvb.cpu().detach().numpy()
            result[i,2] = model.l2_1.cpu().detach().numpy()
            result[i,3] = model.pvb_1.cpu().detach().numpy()
            result[i,4] = model.l2_2.cpu().detach().numpy()
            result[i,5] = model.pvb_2.cpu().detach().numpy()
            result[i,6] = model.l2_3.cpu().detach().numpy()
            result[i,7] = model.pvb_3.cpu().detach().numpy()
            result[i,8] = model.l2.cpu().detach().numpy()
            result[i,9] = model.pvb.cpu().detach().numpy()
        if opt.update_mask:
            train_next_path = os.path.join(opt.dataroot,"train_next_%g"%(opt.update_train_round+1))
            if not os.path.exists(train_next_path):
                os.mkdir(train_next_path)
            design = model.real_A.cpu().detach().numpy()[0,0,:,:]
            mask = model.fake_B.cpu().detach().numpy()[0,0,:,:]
            resist = model.nominal_C.cpu().detach().numpy()[0,0,:,:]
            g_mask = model.real_B.cpu().detach().numpy()[0,0,:,:]
            g_resist = model.real_C.cpu().detach().numpy()[0,0,:,:]
            #print(design.shape, mask.shape, resist.shape, g_mask.shape, g_resist.shape)
            if model.update_mask:
                model_win_count+=1
                filename = img_path[0].split('/')[-1][:-4]+'update.png'
                new_data = np.concatenate((design, mask, resist),axis=1)
            else:
                filename = img_path[0].split('/')[-1]
                new_data = np.concatenate((design, g_mask, g_resist),axis=1)
            print(np.max(new_data),np.min(new_data))
            new_data=new_data*255
            cv2.imwrite(os.path.join(train_next_path,filename),new_data)

        if opt.lt:
            print(opt.epoch, model.l2, model.a2_l2, model.pvb, model.a2_pvb, model.ml_epe_count, model.ilt_epe_count)
            result[i,0:6]= np.array([model.l2.cpu().detach().numpy(), model.ml_epe_count, model.pvb.cpu().detach().numpy(), model.a2_l2.cpu().detach().numpy(),model.a2_pvb.cpu().detach().numpy(), model.ilt_epe_count])


        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        if opt.lt and (not opt.epoch=='best'):
            break
    webpage.save()  # save the HTML
    name.close()
    if not opt.update_mask:
        np.savetxt(os.path.join(opt.results_dir, opt.name, "result%s.csv"%opt.epoch), result, delimiter=',', fmt='%d')
    
    if opt.update_mask:
        win_ratio = model_win_count *1.0 / total_data_count
        print(win_ratio)
