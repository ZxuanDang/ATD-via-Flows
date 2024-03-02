from .torch_model import SemiFlowModel,DiscriminatorClassifier,Flow
from anomalib.models.components import AnomalyModule
from .loss import GeneratorLoss,FlowLoss,DiscriminatorLoss,ClassifierLoss
from torch import optim,Tensor
from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
import torch
import numpy as np

class Semiflow(AnomalyModule):
    def __init__(
        self,
        input_size,
        
        n_feature,
        latent_vec_size,
        num_input_channel: int = 1,
        add_final_conv_layer:bool = True,
        extra_layers:int = 0,
        n_dim: int = 128*50,
        condition_vector: int = 0,
        coupling_blocks: int = 8,
        clamp_alpha: int = 1.9,
        permute_soft: bool =False,

        wcon: int = 50,
        wadv: int = 1,
        lr:int = 0.0002,
        beta1:int = 0.5,
        beta2:int = 0.999,

        mu:float=0.0,
        sigmoid:float=0.1,        

    ):
        super().__init__()

        self.model: SemiFlowModel = SemiFlowModel(
            input_size=input_size,
            num_input_channel=1,
            n_feature=n_feature,
            latent_vec_size=latent_vec_size,
            add_final_conv_layer=add_final_conv_layer,
            extra_layers=extra_layers,
            n_dim=n_dim,
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            permute_soft=permute_soft,
            mu=mu,
            sigmoid=sigmoid
        ).cuda()

        self.model.load_state_dict(torch.load("./model_save/model_extractor_ex.pth"))

        self.flowmodel:Flow = Flow(
            n_feature=n_dim,
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            permute_soft=permute_soft,
            mu=mu,
            sigmoid=sigmoid
        ).cuda()

        self.flowmodel.load_state_dict(torch.load("./model_save/model_flow_ex.pth"))


        self.classifier:DiscriminatorClassifier = DiscriminatorClassifier(
            latent_vec_size=latent_vec_size
        )


        #self.classifier.load_state_dict(torch.load("./model_save/model_class_ex.pth"))


        self.real_embedding:List[Tensor] = []
        self.fake_embedding:List[Tensor] = []

        self.latent_vec=latent_vec_size

        self.generator_loss = GeneratorLoss(wcon=wcon,wadv=wadv)
        self.discriminator_loss = DiscriminatorLoss()
        self.flowloss = FlowLoss()
        self.classifierloss = ClassifierLoss()

        self.learning_rate=lr
        self.beta1 = beta1
        self.beta2 = beta2

    def configure_optimizers(self):
        optimizer_c = optim.Adam(
            self.classifier.parameters(),
            lr=self.learning_rate ,
            betas=(self.beta1,self.beta2)

        )

        return optimizer_c


    def training_step(self, batch, _):
        
        self.model.eval()
        re_batch,fake,latent_i = self.model(batch["image"].cuda())

        fake_latent,log_jac = self.flowmodel(latent_i,rev=True)

        #anomaly:nomal=1：2
        batch_size,channel,feature=fake_latent.shape
        fake_latent=fake_latent[0:int(batch_size/2)]

        pred_real,_ = self.classifier(latent_i)
        pred_fake,_ = self.classifier(fake_latent)

        loss = self.classifierloss(pred_real,pred_fake)

        return {"loss":loss}

    def validation_step(self, batch, batch_idx) -> dict:


        re_batch,fake,latent = self.model(batch["image"].cuda())

        classifier_score , _ = self.classifier(latent)

        batch["pred_scores"] = torch.max(classifier_score, dim=1).values

        return batch


class SemiflowLightning(Semiflow):
    def __init__(
        self, 
        hparams,
    ) -> None:
        super().__init__(
            input_size=hparams.model.input_size[0], 
            n_feature=hparams.model.n_features, 
            latent_vec_size = hparams.model.latent_vec_size, 
            n_dim=hparams.model.n_dim,


            condition_vector=hparams.model.condition_vector, 
            coupling_blocks=hparams.model.coupling_blocks, 
            clamp_alpha=hparams.model.clamp_alpha, 
            permute_soft=hparams.model.soft_permutation, 
            wcon=hparams.model.wcon, 
            wadv=hparams.model.wadv, 
            lr=hparams.model.lr, 
            beta1=hparams.model.beta1, 
            beta2=hparams.model.beta2,

            mu=hparams.model.mu,
            sigmoid=hparams.model.sigmoid,
        )
        self.hparams: Union[DictConfig,ListConfig]
        self.save_hyperparameters(hparams)

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]