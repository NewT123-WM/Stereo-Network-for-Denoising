import h5py
from .mydata import MyData
import torch



def load_data(train_path, val_path, train_batch_size, val_batch_size):
    f = h5py.File(train_path, "r")
    x_train = f['x_train']
    y_train = f['y_train']

    g = h5py.File(val_path, "r")
    x_val = g['x_val']
    y_val = g['y_val']

    trainset = MyData(x_train, y_train)
    valset = MyData(x_val, y_val)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)

    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size=val_batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True)

    return trainloader, valloader
