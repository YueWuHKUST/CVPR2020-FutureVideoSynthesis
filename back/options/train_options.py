from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')   
        self.parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        
        self.parser.add_argument('--pool_size', type=int, default=1, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--continue_train', action='store_true', help='whether continue train')
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of patch scales in each discriminator')        
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')  
        self.parser.add_argument('--lambda_feat', type=float, default=15.0, help='weight for feature matching')
        self.parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for feature matching')                 

        # for temporal
        self.parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for temporal loss')
        self.parser.add_argument('--lambda_F', type=float, default=10.0, help='weight for flow loss')
        self.parser.add_argument('--lambda_smooth', type=float, default=1.0, help='weight for smooth loss')
        self.parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for discriminator loss')
        self.parser.add_argument('--lambda_D_T', type=float, default=1.0, help='weight for temporal discriminator loss')

        self.isTrain = True
