from pytorch_lightning.core.datamodule import LightningDataModule

from typing import Dict,Optional,Tuple,Union
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch
from pathlib import Path
from torch import Tensor


def Make_Frame(
        normal_path:str,
        valid_path:str,
        normal_test_path:str,
        split:str,
        split_ratio:float,
    )->pd.DataFrame:

        if split == "train":
            data=pd.read_csv(normal_path)

        elif split == "valid":
            data=pd.read_csv(valid_path)

        elif split =="test":
            data=pd.read_csv(normal_test_path)

        return data


class One_Packet_dataset(Dataset):
    def __init__(
        self,
        normal_path:str,
        valid_path:str,
        normal_test_path:str,
        split:str,
        split_ratio:float,
        label=True,
    ):
        self.globa={'nan':0}
        self.label=label

        self.pd=Make_Frame(normal_path=normal_path,valid_path=valid_path,normal_test_path=normal_test_path,split=split,split_ratio=split_ratio)

    def __len__(self):
        return len(self.pd)

    def __getitem__(self,idx):
        
        item:Dict[str,Union[str,Tensor]] = {}

        x=torch.tensor(eval(self.pd["packet"][idx]),dtype=torch.float)
        item["image"] = x
        #no flow feature x1=torch.tensor(eval(self.pd["feature"][idx],self.globa),dtype=torch.float)
        if self.label:
            y1=torch.tensor(self.pd["h1_label"][idx],dtype=torch.float)
            item["label"] = y1

            return item
        else:
            return item    



class One_dimension_packet(LightningDataModule):
    """one dimension csv"""
    def __init__(
        self,
        root,
        normal_path:str,
        abnormal_path:str,
        normal_test_path:str,
        split_ratio:float,
        create_validtion_set:str,
        batch_size:int,
        test_batch_size:int,
        num_workers:int,
    )->None:
        super().__init__()

        self.root = root if isinstance(root,Path) else Path(root)
        self.normal_path = self.root / normal_path
        self.abnormal_path = self.root / abnormal_path
        self.normal_test_path = self.root / normal_test_path
        self.split_ratio=split_ratio
        self.create_validtion_set = create_validtion_set

        self.batch_size=batch_size
        self.num_workers=num_workers
        self.test_batch_size=test_batch_size


    def setup(self,stage:Optional[str]=None)->None:
        """
            Setup train,validtion and test data.
        
        Args:
            stage: Train/Val/Test  stages
        """
        if stage in (None,"fit"):
            self.train_data=One_Packet_dataset(
                normal_path=self.normal_path,
                valid_path=self.abnormal_path,
                normal_test_path=self.normal_test_path,
                split="train",
                split_ratio=self.split_ratio,
                label=False
            )
        
        if self.create_validtion_set:
            self.vali_data=One_Packet_dataset(
                normal_path=self.normal_path,
                valid_path=self.abnormal_path,
                normal_test_path=self.normal_test_path,
                split="valid",
                split_ratio=self.split_ratio,
                label=True
            )

        self.test_data=One_Packet_dataset(
            normal_path=self.normal_path,
            valid_path=self.abnormal_path,
            normal_test_path=self.normal_test_path,
            split="test",
            split_ratio=self.split_ratio,
            label=True
        )

        if stage == "predict":
            self.inference = self.test_data
    
    def train_dataloader(self):
        return DataLoader(self.train_data,shuffle=True,batch_size=self.batch_size,num_workers=self.num_workers)
    
    def val_dataloader(self):
        dataset = self.vali_data if self.create_validtion_set else self.test_data
        return DataLoader(dataset=dataset,shuffle=False,batch_size=self.batch_size,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,shuffle=False,batch_size=self.test_batch_size,num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.inference,shuffle=False,batch_size=self.test_batch_size,num_workers=self.num_workers)


