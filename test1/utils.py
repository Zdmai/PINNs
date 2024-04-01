# Reading/Writing Data

from data import ImageDataset
from torch.utils.data import DataLoader


# def download(url, dir: str, filename: str):
#
#     if not os.path.exists(dir):
#         os.mkdir(dir)
#
#
#     if filename not in os.listdir(dir):
#         urllib.request.urlretrieve(url, dir+filename)

def build_loader(config):
    train_set, train_loader = None, None
    if config['train_root'] is not None:
        train_set = ImageDataset(root=config['train_root'], return_index=False)
        train_loader = DataLoader(train_set, shuffle=True, batch_size=config['batch_size'])

    val_set, val_loader = None, None
    if config['valild_root'] is not None:
        val_set = ImageDataset(root=config['valild_root'], return_index=False)
        val_loader = DataLoader(val_set, shuffle=True, batch_size=config['batch_size'])
        # only give the number of batch size data

    return train_loader, val_loader

# def get_dataset(config):
#     if config['train_root'] is not None:
#         train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
#         return train_set
#     return None



