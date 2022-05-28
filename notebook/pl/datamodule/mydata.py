import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
import config
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
from pl_bolts.datamodules import SklearnDataset
import numpy as np
PIN_MEMORY=True


class MlpDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = config.DATA_DIR, batch_size: int = 64, num_workers: int = 20,fraction_rate: float = 0.8,val_fraction_rate: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fraction_rate = fraction_rate
        self.val_fraction_rate = val_fraction_rate
        
        data0 = torch.load(data_dir+'/embedding_dict_data0.pth')
        data0_ = [data0[30*i:30*(i+1),:].flatten().detach().numpy() for i in range(int(len(data0)/30))]
        data0 = torch.Tensor(data0_)
        data0 =torch.cat([data0,torch.zeros(len(data0),1)],1)
        data1 = torch.load(data_dir+'/embedding_dict_data1.pth')
        data1_ = [data1[30*i:30*(i+1),:].flatten().detach().numpy() for i in range(int(len(data1)/30))]
        data1 = torch.Tensor(data1_)
        data1 = torch.cat([data1,torch.ones(len(data1),1)],1)
        data = torch.cat([data0,data1],0)
        idx = list(range(len(data)))
        random.shuffle(idx)
        data = data[idx]
        self.data = data.detach().numpy()
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        train_test_split = int(self.fraction_rate*(len(self.data)))
        insample = self.data[:train_test_split,:]
        test_data = self.data[train_test_split:,:]
        train_val_split = int((1-self.val_fraction_rate)*(len(insample)))
        train_data = insample[:train_val_split,:]
        val_data  = insample[train_val_split:,:]
        self.dataset_train = SklearnDataset(X=train_data[:,:-1],y = train_data[:,-1].astype(int))
        self.dataset_val = SklearnDataset(X=val_data[:,:-1],y = val_data[:,-1].astype(int))
        self.dataset_test = SklearnDataset(X=test_data[:,:-1],y = test_data[:,-1].astype(int))
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY,shuffle=True)
    
    
class CNNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = config.DATA_DIR, batch_size: int = 64, num_workers: int = 20,fraction_rate: float = 0.8,val_fraction_rate: float = 0.1,pin_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2
        self.fraction_rate = fraction_rate
        self.val_fraction_rate = val_fraction_rate
        self.pin_memory = pin_memory
        data0 = torch.load(data_dir+'/embedding_dict_data0.pth')
        data0 =torch.cat([data0,torch.zeros(len(data0),1)],1)
        data1 = torch.load(data_dir+'/embedding_dict_data1.pth')
        data1 = torch.cat([data1,torch.ones(len(data1),1)],1)
        data = torch.cat([data0,data1],0)
        self.data = data.detach().numpy()
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        data = np.array([self.data[i*30:(i+1)*30,:-1] for i in range(int(len(self.data)/30))])
        y = self.data[::30,-1]
        idx = list(range(len(data)))
        random.shuffle(idx)
        data = data[idx]
        y = y[idx]
        train_test_split = int(self.fraction_rate*len(data))
        insample = data[:train_test_split]
        test_data = data[train_test_split:]
        train_val_split = int((1-self.val_fraction_rate)*len(insample))
        train_data = insample[:train_val_split]
        val_data  = insample[train_val_split:]
        y_train = y[:train_test_split][:train_val_split]
        y_val = y[:train_test_split][train_val_split:]
        y_test = y[train_test_split:]
        self.dataset_train,self.dataset_val,self.dataset_test = [
            SklearnDataset(X=i,y = np.array(j).astype(int)) for i,j in zip([train_data,val_data,test_data],[y_train,y_val,y_test])
        ]
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=self.pin_memory,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=self.pin_memory,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=self.pin_memory,shuffle=True)