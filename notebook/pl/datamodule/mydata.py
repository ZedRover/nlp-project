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
PIN_MEMORY=False


class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = config.DATA_DIR, batch_size: int = 64, num_workers: int = 20,fraction_rate: float = 0.8,val_fraction_rate: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10
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
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index,:]
        self.data = data.detach().numpy()
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        train_test_split = int(self.fraction_rate*len(self.data))
        insample = self.data[:train_test_split,:]
        test_data = self.data[train_test_split:,:]
        train_val_split = int((1-self.val_fraction_rate)*len(insample))
        train_data = insample[:train_val_split,:]
        val_data  = insample[train_val_split:,:]
        self.dataset_train = SklearnDataset(X=train_data[:,:-1],y = train_data[:,-1].astype(int))
        self.dataset_val = SklearnDataset(X=val_data[:,:-1],y = val_data[:,-1].astype(int))
        self.dataset_test = SklearnDataset(X=test_data[:,:-1],y = test_data[:,-1].astype(int))
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)
    
    
class CNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = config.DATA_DIR, batch_size: int = 64, num_workers: int = 20,fraction_rate: float = 0.8,val_fraction_rate: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10
        self.fraction_rate = fraction_rate
        self.val_fraction_rate = val_fraction_rate
        
        data0 = torch.load(data_dir+'/embedding_dict_data0.pth')
        data0 =torch.cat([data0,torch.zeros(len(data0),1)],1)
        data1 = torch.load(data_dir+'/embedding_dict_data1.pth')
        data1 = torch.cat([data1,torch.ones(len(data1),1)],1)
        data = torch.cat([data0,data1],0)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index,:]
        self.data = data.detach().numpy()
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        train_test_split = int(self.fraction_rate*len(self.data))
        insample = self.data[:train_test_split,:]
        test_data = self.data[train_test_split:,:]
        train_val_split = int((1-self.val_fraction_rate)*len(insample))
        train_data = insample[:train_val_split,:]
        val_data  = insample[train_val_split:,:]
        y_train,y_val,y_test  = [[i[30*(j),-1] for j in range(int(len(i)//30)) ]for i in [train_data,val_data,test_data] ]
        x_train,x_val,x_test = [[i[30*(j):30*(j+1),:-1] for j in range(int(len(i)//30)) ] for i in [train_data,val_data,test_data] ]
        self.dataset_train,self.dataset_val,self.dataset_test = [
            SklearnDataset(X=i,y = np.array(j).astype(int)) for i,j in zip([x_train,x_val,x_test],[y_train,y_val,y_test])
        ]
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=PIN_MEMORY)