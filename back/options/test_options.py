from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./result/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=300, help='how many test images to run')        
        self.parser.add_argument('--non_rigid_dir', type=str, default='', help='directory of non rigid objects')
        self.parser.add_argument('--small_object_mask_dir', type=str, default='', help='directory of non rigid objects')
        self.isTrain = False
