from torch import nn,Tensor
import numpy as np
import torch
import torch.nn.functional as F

class FlowLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self,dim_feature_vector,p_u,logdef_j):

        loss = - 0.5 * torch.sum(p_u**2,1) + logdef_j 
        loss = -F.logsigmoid(loss)

        loss=loss.mean()/dim_feature_vector

        return loss

class GeneratorLoss(nn.Module):
    def __init__(self,wcon,wadv) -> None:
        super().__init__()
        self.loss_con = nn.MSELoss() 
        self.loss_adv = nn.MSELoss()

        self.wcon = wcon
        self.wadv = wadv

    def forward(self,image,fake,pred_real,pred_fake):
        

        error_con = self.loss_con(fake,image)

        error_adv = self.loss_adv(pred_real,pred_fake)


        loss = self.wcon * error_con + self.wadv * error_adv

        return loss


class DiscriminatorLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_bce = nn.BCELoss()

    def forward(self,pred_real,pred_fake):

        error_discriminator_real = self.loss_bce(
            pred_real,torch.ones(size=pred_real.shape,dtype=torch.float32,device=pred_real.device)
        )

        error_discriminator_fake = self.loss_bce(
            pred_fake,torch.zeros(size=pred_fake.shape,dtype=torch.float32,device=pred_fake.device)
        )

        loss_discriminator = (error_discriminator_fake + error_discriminator_real) * 0.5
        return loss_discriminator


class ClassifierLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss_ce =  nn.CrossEntropyLoss() 
    
    def forward(self,pred_real,pred_fake):
       
        true_label = torch.zeros(size=pred_real.shape,dtype=torch.float32,device=pred_real.device)
        fake_label = torch.ones(size=pred_fake.shape,dtype=torch.float32,device=pred_fake.device)
        label = torch.cat((true_label, fake_label), dim=0)


        pred = torch.cat((pred_real, pred_fake), dim=0)

        loss = self.loss_ce(pred, label)

        return loss