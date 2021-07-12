import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):                
        self.parser.add_argument('--ImagesRoot', type=str, default='')
        self.parser.add_argument('--SemanticRoot', type=str, default='')
        self.parser.add_argument('--InstanceRoot', type=str, default='/disk1/yue/cityscapes/cityscapes/InstanceMap_256p/')
        self.parser.add_argument('--StaticMapDir', type=str, default="", help='used when group static cars to background')
        
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--image_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--semantic_nc', type=int, default=19, help='# of input semantic channels')
        self.parser.add_argument('--flow_nc', type=int, default=2, help='# of output image channels')
          
        self.parser.add_argument('--ngf', type=int, default=128, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--n_blocks', type=int, default=9, help='number of resnet blocks in generator')
        self.parser.add_argument('--n_downsample_G', type=int, default=3, help='number of downsampling layers in netG')        

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset', type=str, default='cityscapes', help='chooses how datasets are loaded. [cityscapes | kitti]')
        self.parser.add_argument('--model', type=str, default='vid2vid', help='chooses which model to use. vid2vid, test')        
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')        
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')       
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')                    
    
        # for cascaded resnet        
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of resnet blocks in outmost multiscale resnet')        
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of cascaded layers')        

        # temporal
        self.parser.add_argument('--tIn', type=int, default=4, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        self.parser.add_argument('--tOut', type=int, default=5, help='number of output frames for prediction')      

        self.parser.add_argument('--static', action='store_true', help='whether add static cars to background')  
        # miscellaneous                
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')                
        self.parser.add_argument('--debug', action='store_true', help='if specified, use small dataset for debug')
        self.initialized = True
        
    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
    
        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
