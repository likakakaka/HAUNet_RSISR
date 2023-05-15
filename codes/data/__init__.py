
from torch.utils.data import DataLoader

def create_dataloaders(args):
    """create dataloader"""
    if args.dataset == 'AID':
        from data.aid import AIDataset
        training_set = AIDataset(args, root_dir='G:\datasets\yaogan\dataset\AID-dataset/train',
                                 train=True)
        val_set = AIDataset(args, root_dir='G:\datasets\yaogan\dataset\AID-dataset/val',
                            train=False)
        test_set = AIDataset(args, root_dir='G:\datasets\yaogan\dataset/AID-dataset/val',
                                   train=False)
    elif args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        training_set = UCMercedDataset(args, root_dir='G:\datasets\yaogan\dataset/UCMerced-dataset/train',
                                 train=True)
        val_set = UCMercedDataset(args, root_dir='G:\datasets\yaogan\dataset/UCMerced-dataset/val',
                            train=False)
        test_set = UCMercedDataset(args, root_dir='G:\datasets\yaogan\dataset/UCMerced-dataset/val',
                                  train=False)
    elif args.dataset == 'RSCNN7':
        from data.rscnn7 import RSCNN7Dataset
        training_set = RSCNN7Dataset(args, root_dir='G:\datasets\yaogan\dataset\RSSCN7/train',
                                       train=True)
        val_set = RSCNN7Dataset(args, root_dir='G:\datasets\yaogan\dataset\RSSCN7/val',
                                  train=False)
        test_set = RSCNN7Dataset(args, root_dir='G:\datasets\yaogan\dataset\RSSCN7/val',
                                   train=False)
    elif args.dataset == 'DIV2K':
        from data.div2k import DIV2KDataset
        training_set = DIV2KDataset(args, root_dir='G:\datasets\yaogan\dataset\DIV2K/train',
                                    train=True)
        val_set = DIV2KDataset(args, root_dir='G:\datasets\yaogan\dataset\DIV2K/val',
                               train=False)
        test_set = DIV2KDataset(args, root_dir='G:\datasets\yaogan\dataset/test\Set14',
                                   train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    dataloaders = {'train': DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0),  # args.n_threads
                   'val': DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0),
                   'test': DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=0),
                   }  # args.n_threads

    return dataloaders



















