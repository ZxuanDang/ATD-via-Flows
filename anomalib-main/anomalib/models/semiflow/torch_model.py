

from torch import Tensor
import torch.nn as nn
import torch
from typing import List,Callable,Tuple,Union
from anomalib.models.components.freia.framework import SequenceINN
from anomalib.models.components.freia.modules import AllInOneBlock
import einops
import math
import numpy as np

class Encoder(nn.Module):
    def __init__(
        self,
        input_size:int,
        num_input_channel,
        n_feature,
        
        latent_vec_size,
        add_final_conv_layer:bool = True,
        extra_layer:int=0, 
    ) -> None:
        super().__init__()

        self.input_layers=nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channel}-{n_feature}",
            nn.Conv1d(num_input_channel,n_feature,kernel_size=4,stride=2,padding=1,bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_feature}",nn.LeakyReLU(0.2,inplace=True))

        self.extra_layer=nn.Sequential()

        self.pyramid_features=nn.Sequential()
        pyramid_dim=input_size // 2
        while pyramid_dim > 100:
            in_features=n_feature
            out_features=n_feature * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv1d(in_features,out_features,kernel_size=4,stride=2,padding=1,bias=False)
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm1d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_feature=out_features
            pyramid_dim=pyramid_dim//2
        
        if add_final_conv_layer:
            self.final_conv_layer=nn.Conv1d(
                n_feature,
                latent_vec_size,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False
            )



    def forward(self,input_tensor:Tensor):
        output = self.input_layers(input_tensor)

        output = self.extra_layer(output)

        output = self.pyramid_features(output)

        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)
        return output

class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        n_features,
        latent_vec_size,
        num_input_channels,
        add_final_conv_layer:bool = False,
        extra_layers:int=0,

    ) -> None:
        super().__init__()
        
        self.latent_input=nn.Sequential()

        exp_factor = math.ceil(math.log(input_size // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose1d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm1d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = input_size // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 100:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose1d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm1d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        #extra_layer
        self.extra_layers=nn.Sequential()

        self.final_layers=nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-conv",
            nn.ConvTranspose1d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh",nn.Tanh())

    def forward(self,input_tensor):
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        output = self.final_layers(output)
        return output


def subnet_fc(dims_in:int,dims_out:int):
    """Subnetwork which predicts the affine coefficients
    Args:
        dims_in(int): input dimensions
        dims_out(int): output dimensions 
    """
    return nn.Sequential(
        nn.Linear(dims_in,2*dims_in),
        nn.ReLU(),
        nn.Linear(2*dims_in,dims_out)
    )

def cflow_head(
    condition_vector:int,coupling_blocks:int,clamp_alpha:float,n_feature:int,permute_soft:bool = False
) -> SequenceINN:

    coder=SequenceINN(n_feature)
    for _ in range(coupling_blocks):
        coder.append(
            AllInOneBlock,
            cond=None,    
            cond_shape=(condition_vector,), 
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=permute_soft,
    )
    return coder

class Flow(nn.Module):

    def __init__(
        self,
        n_feature:int = 128*50,
        condition_vector:int = 0,
        coupling_blocks:int = 8,
        clamp_alpha:int = 1.9,
        permute_soft:bool = False,

        #
        mu:float=0.0,
        sigmoid:float=0.1,
    ) -> None:
        super().__init__()
        
        self.decoder = cflow_head(
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            n_feature=n_feature,
            permute_soft=permute_soft,
        )

        self.mu=mu
        self.sigmoid=sigmoid

    def forward(self,batch,rev=False):
        """
        Args:
        batch: shape[batch,channel,feature]
        """

        batch_size,channel,features = batch.shape

        e_r = einops.rearrange(batch,"b c f -> b (c f)")

        p_u,log_jac_det = self.decoder(e_r)
        
        if rev == True:

            rand_sample = torch.tensor(np.random.normal(self.mu,self.sigmoid,size=e_r.shape).astype("float32"),device=batch.device)

            fake_e_r=p_u.detach() + rand_sample

            p_u,log_jac_det = self.decoder(fake_e_r,rev=rev)
            
            p_u=einops.rearrange(p_u,"b (c f) -> b c f",c=channel,f=features)

        return p_u,log_jac_det   


class PreNormResidual(nn.Module):
    def __init__(self,dim,fn) -> None:
        super().__init__()
        self.fn=fn
        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        return self.fn(self.norm(x)) + x



class DiscriminatorClassifier(nn.Module):

    def __init__(
        self,
        latent_vec_size:int = 64,
    ) -> None:
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv1d( latent_vec_size ,latent_vec_size * 2 , kernel_size=3 , stride=1,padding=1 ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(latent_vec_size * 2),

            nn.Conv1d(latent_vec_size * 2,latent_vec_size * 2, kernel_size=3 , stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(latent_vec_size * 2),

            nn.MaxPool1d(2)
        )



        self.classifier = nn.Sequential(
            nn.Linear(3360 ,1024 ),
            #nn.Dropout(p=0.05),
            nn.ReLU(inplace=True),


            nn.Linear(1024,256),
            #nn.Dropout(p=0.05),
            nn.ReLU(inplace=True),

            nn.Linear(256,128),
            #nn.Dropout(p=0.05),
            nn.ReLU(inplace=True),

            nn.Linear(128,2),
            #nn.Sigmoid()
        )   

    def forward(self,input_tensor):


        act1=self.layer1(input_tensor)

        act2_ = act1.view(act1.shape[0],-1)

        classifier = self.classifier(act2_)


        return classifier,act1




class Discriminator(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channels,
        n_features,
        extra_layers:int=0,
    ) -> None:
        super().__init__()
        encoder=Encoder(input_size=input_size,latent_vec_size=1,num_input_channel=num_input_channels,n_feature=n_features,extra_layer=extra_layers)
        layers=[]

        for block in encoder.children():
            if isinstance(block,nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)
        
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid",nn.Sigmoid())

    def forward(self,input_tensor):

        features=self.features(input_tensor)
        classifier=self.classifier(features)

        classifier=classifier.view(input_tensor.shape[0],-1).squeeze(1)

        classifier = torch.mean(classifier,dim=(1))

        return classifier,features


class Generator(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channel,
        n_features,
        latent_vec_size,
        add_final_conv_layer:bool=True,
        extra_layers:int =0,
    ) -> None:
        super().__init__()
        self.encoder1=Encoder(input_size=input_size,num_input_channel=num_input_channel,n_feature=n_features,add_final_conv_layer=add_final_conv_layer,latent_vec_size=latent_vec_size)
        self.decoder=Decoder(input_size=input_size,n_features=n_features,latent_vec_size=latent_vec_size,num_input_channels=num_input_channel,add_final_conv_layer=add_final_conv_layer)

    def forward(self,input_tensor):

        latent_i=self.encoder1(input_tensor)


        gen_image=self.decoder(latent_i)
        

        return latent_i,gen_image


class SemiFlowModel(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channel,

        n_feature,
        latent_vec_size,
        add_final_conv_layer:bool = True,
        extra_layers:int = 0,

    ) -> None:
        super().__init__()

        self.generator:Generator = Generator(
            input_size=input_size,
            num_input_channel=num_input_channel,
            n_features=n_feature,
            latent_vec_size=latent_vec_size,
            add_final_conv_layer=add_final_conv_layer,
        )


        self.discriminator:Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channel,
            n_features=n_feature,
            extra_layers=extra_layers,
        )



    def forward(self,batch):

        batch=batch.unsqueeze(1)


        latent_i,fake=self.generator(batch)


        return batch,fake,latent_i
        