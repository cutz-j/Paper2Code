from .base_dicts import BaseOptions


class TrainOptions(BaseOptions):
    """
    This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # visdom and HTML visualization parameters
        parser.display_freq = 400
        parser.display_ncols = 4
        parser.display_id = 1
        parser.display_server = "http://localhost"
        parser.display_env = 'main'
        parser.display_port = 8888
        parser.update_html_freq = 1000
        parser.print_freq = 100
        parser.no_html = True
        # network saving and loading parameters
        parser.save_latest_freq = 5000
        parser.save_epoch_freq = 5
        parser.save_by_iter = True
        parser.continue_train = False
        parser.epoch_count = 1
        parser.phase = 'train'
        # training parameters
        parser.n_epochs = 100
        parser.n_epochs_decay = 100
        parser.beta1 = 0.5
        parser.lr = 0.0002
        parser.gan_mode = 'vanilla' # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        parser.pool_size = 50
        parser.lr_policy = 'linear' # learning rate policy. [linear | step | plateau | cosine]
        parser.lr_decay_iters = 50
        self.isTrain = True
        return parser