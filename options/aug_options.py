from .base_options import BaseOptions


class AugOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
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
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # augmentation parameters
        parser.add_argument('--augmode', type=str, default='random', help='data augmentation mode. one of random|adv_style|adv_noise|none.')
        parser.add_argument('--gan_batch_size', type=int, default=10, help='batch size used for finetune styleGAN.')
        parser.add_argument('--augroot', type=str, default='./augmentation', help='directory to save augmented data.')
        parser.add_argument('--gan_model', type=str, default='./stylegan_model/network-snapshot-025000.pkl', help='pickle file for pre-trained styleGAN model.')
        parser.add_argument('--rank_buffer_size', type=int, default=10, help='number of new data generated each augmentation iteration.')
        parser.add_argument('--aug_kimg', type=int, default=6, help='number of images seen by discriminator in GAN finetune.')
        parser.add_argument('--aug_iter', type=int, default=1, help='number of augmentation iteration')
        # set training to True
        self.isTrain = True
        return parser
