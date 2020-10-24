import torch.utils.data

class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'
    
    def CreateDataset(self, opt):
        if opt.dataset == 'cityscapes':
            from data.temporal_dataset import TemporalDataset
            dataset = TemporalDataset()
        elif opt.dataset == 'kitti':
            from data.kitti_dataset import KittiDataset
            dataset = KittiDataset()
        print("dataset [%s] was created" % (dataset.name()))
        dataset.initialize(opt)
        return dataset


    def initialize(self, opt):
        self.opt = opt 
        self.dataset = self.CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=opt.isTrain,
            num_workers=0)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
