import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt, flownet):
    dataset = None
    if opt.dataset == 'cityscapes' and opt.isTrain is True:
        from data.temporal_dataset import TemporalDataset
        dataset = TemporalDataset()
        dataset.initialize(opt)
    elif opt.dataset == 'cityscapes' and opt.isTrain is False and opt.next is False:
        from data.temporal_dataset_test_my_back import TestTemporalDataset
        dataset = TestTemporalDataset()
        dataset.initialize(opt, flownet)
    elif opt.dataset == 'cityscapes' and opt.isTrain is False and opt.next is True:
        from data.temporal_dataset_test_my_back_next import TestTemporalDataset
        dataset = TestTemporalDataset()
        dataset.initialize(opt, flownet)
    elif opt.dataset == 'kitti' and opt.isTrain is False:
        from data.temporal_dataset_test_my_back import TestTemporalDataset
        dataset = TestTemporalDataset()
        dataset.initialize(opt, flownet)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    
    return dataset



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, flownet=None):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, flownet)
        shuffle_flag = True if opt.isTrain is True else False
        workers=4 if opt.isTrain is True else 0
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuffle_flag,
            num_workers=workers)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
