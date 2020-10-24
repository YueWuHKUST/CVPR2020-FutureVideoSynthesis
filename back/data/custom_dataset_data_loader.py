import torch.utils.data
from data.base_data_loader import BaseDataLoader
#sorted

def CreateDataset(opt):
    if opt.dataset == 'cityscapes' and opt.isTrain is True:
        from data.temporal_dataset import TemporalDataset
        dataset = TemporalDataset()
    elif opt.dataset == 'cityscapes' and opt.isTrain is False:
        from data.temporal_dataset_test import TemporalDataset
        dataset = TemporalDataset()
    elif opt.dataset == 'kitti' and opt.isTrain is True:
        from data.kitti_dataset import KittiDataset
        dataset = KittiDataset()
    elif opt.dataset == 'kitti' and opt.isTrain is False:
        from data.kitti_dataset_test import TestKittiDataset
        dataset = TestKittiDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, flownet):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=opt.isTrain,
            num_workers=1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
