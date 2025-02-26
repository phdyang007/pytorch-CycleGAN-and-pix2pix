from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=10, help='how many test images to run')

        #ilt opt
        parser.add_argument('--max_ilt_step', type=int, default=20, help='how many test images to run')
        parser.add_argument('--ilt_lr', type=float, default=10, help='learning rate for ILT mask post processing')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        parser.add_argument('--dump_imask', action='store_true', help='dump the intermediate mask in each doinn step, only for doinn multi.')  
        parser.add_argument('--update_mask', action='store_true', help='do the inference on training set and get better mask')  
        # rewrite devalue values
        #parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
