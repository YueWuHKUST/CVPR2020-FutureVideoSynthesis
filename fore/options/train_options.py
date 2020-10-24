from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')        
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--TTUR', action='store_true', help='Use TTUR training scheme')        
        self.parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        self.parser.add_argument('--pool_size', type=int, default=1, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of patch scales in each discriminator')        
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')       

        # for temporal
        self.parser.add_argument('--lambda_D', type=float, default=4.0, help='weight for flow loss')
        self.parser.add_argument('--lambda_D_T', type=float, default=8.0, help='weight for flow loss')
        self.parser.add_argument('--lambda_scale', type=float, default=4.0, help='weight for scale penalty')
        self.parser.add_argument('--lambda_shear', type=float, default=2.0, help='weight for shear penalty')
        self.parser.add_argument('--lambda_rotation', type=float, default=2.0, help='weight for rotation penalty')
        self.parser.add_argument('--lambda_translationx', type=float, default=6.0, help='weight for translation penalty')
        self.parser.add_argument('--lambda_translationy', type=float, default=6.0, help='weight for translation penalty')
        self.parser.add_argument('--lambda_smooth', type=float, default=2.0, help='weight for smooth penalty')
        self.parser.add_argument('--lambda_image', type=float, default=30.0, help='weight for shear penalty')
        self.parser.add_argument('--n_frames_D', type=int, default=1, help='number of frames to feed into temporal discriminator')        
        self.parser.add_argument('--n_scales_temporal', type=int, default=3, help='number of temporal scales in the temporal discriminator')        
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=1, help='max number of frames to load into one GPU at a time')
        self.parser.add_argument('--max_frames_backpropagate', type=int, default=1, help='max number of frames to backpropagate') 
        self.parser.add_argument('--max_t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        self.parser.add_argument('--n_frames_total', type=int, default=30, help='the overall number of frames in a sequence to train with')                
        self.parser.add_argument('--niter_step', type=int, default=5, help='how many epochs do we change training batch size again')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='if specified, only train the finest spatial layer for the given iterations')
        self.parser.add_argument('--similar_input', action='store_true', help='..')
        self.parser.add_argument('--adjacent_r', action='store_true', help='..')
        self.isTrain = True
