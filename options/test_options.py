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
        parser.add_argument('--num_test', type=int, default=2000, help='how many test images to run')

        #ilt opt
        parser.add_argument('--max_ilt_step', type=int, default=20, help='how many test images to run')
        parser.add_argument('--ilt_lr', type=float, default=10, help='learning rate for ILT mask post processing')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        parser.add_argument('--dump_imask', action='store_true', help='dump the intermediate mask in each doinn step, only for doinn multi.')  

        #stylegan 
        parser.add_argument('--outdir', type=str, default='./out_style/', help='output results here.')
        parser.add_argument('--stylegan', type=str, default='./stylegan_model/network-snapshot-025000.pkl', help='stylegan model.')
        parser.add_argument('--num_gen', type=int, default=50, help='number of data to generate.')
        parser.add_argument('--aug_type', type=str, default='style', help='style|noise|random.')
        parser.add_argument('--lr_alpha', type=float, default=0.01, help='step size for attack.')
        parser.add_argument('--dist_norm', type=float, default=1.0, help='add distribution normalization.')
        parser.add_argument('--quantize_aware', action='store_true', help='use STE for legalization.')
        parser.add_argument('--loss_type', type=str, default='houdini', help='houdini|logprob|mse. valid only for style attack.')
        parser.add_argument('--attack_epoch', type=int, default=10, help='number of epochs for attack.')

        # rewrite devalue values
        #parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
