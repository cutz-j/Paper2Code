from easydict import EasyDict
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """
    This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""    
        # basic parameters
        parser.dataroot = ''
        parser.name = 'experiment_name'
        parser.gpu_ids = '0'  # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
        parser.checkpoints_dir = './checkpoints'
        # model parameters
        parser.model = 'pix2pix' # cycle_gan | pix2pix | test | colorization
        parser.input_nc = 3
        parser.output_nc = 3
        parser.ngf = 64
        parser.ndf = 64
        parser.netD = 'basic' # specify discriminator architecture [basic | n_layers | pixel]
        parser.netG = 'unet_128' # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        parser.n_layers_D = 3
        parser.norm = 'batch' # instance normalization or batch normalization [instance | batch | none]
        parser.init_type = 'normal' # network initialization [normal | xavier | kaiming | orthogonal]
        parser.init_gain = 0.02
        parser.no_dropout = True
        # dataset parameters
        parser.dataset_mode = 'algined' # chooses how datasets are loaded. [unaligned | aligned | single | colorization]
        parser.direction = 'AtoB' # AtoB or BtoA
        parser.serial_batches = True # if true, takes images in order to make batches, otherwise takes them randomly
        parser.num_threads = 4
        parser.batch_size = 1
        parser.load_size = 286
        parser.crop_size = 256
        parser.max_dataset_size = float("inf")
        parser.preprocess = 'resize_and_crop'
        parser.no_flip = True
        parser.display_winsize = 256
        # additional parameters
        parser.epoch = 'latest'
        parser.load_iter = 0
        parser.verbose = True
        parser.suffix = ''
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = EasyDict()
            parser = self.initialize(parser)

        # get the basic options
        opt = parser

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt = parser  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser[k]
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, args):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        for k in args:
            opt[k] = args[k]

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        

        self.opt = opt
        return self.opt