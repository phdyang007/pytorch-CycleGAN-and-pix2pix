from .base_options import BaseOptions


class AugOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=50000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # initial training parameters
        parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--n_epochs_decay_1', type=int, default=10, help='lithotwin phase 1: number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--n_epochs_decay_2', type=int, default=10, help='lithotwin phase 2: number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # augmentation parameters
        parser.add_argument('--augmode', type=str, default='rand', help='data augmentation mode. one of rand|adv_style|adv_noise|none.')
        parser.add_argument('--augroot', type=str, default='./augmentation', help='directory to save augmented data.')
        parser.add_argument('--gan_model', type=str, default='/workspace/stylegan2-ada-pytorch/stylegan_model/network-snapshot-025000.pkl', help='pickle file for pre-trained styleGAN model.')
        parser.add_argument('--augsize', type=int, default=200, help='number of new data generated each augmentation iteration.')
        parser.add_argument('--aug_iter', type=int, default=10, help='number of augmentation iteration')
        parser.add_argument('--aug_epochs', type=int, default=1, help='number of epochs for each iteration training')
        parser.add_argument('--aug_epochs_decay', type=int, default=2, help='number of epochs to linearly decay learning rate to zero')
        # set training to True
        self.isTrain = True
        return parser
