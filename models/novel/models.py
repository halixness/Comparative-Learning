'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Credit: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
VAE Credit: https://github.com/AntixK/PyTorch-VAE/tree/a6896b944c918dd7030e7d795a8c13e5c6345ec7
Contrastive Loss: https://lilianweng.github.io/posts/2021-05-31-contrastive/
CLIP train: https://github.com/openai/CLIP/issues/83

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import clip
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from config import *
device = "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Model(nn.Module):
    
    def forward(self):
        raise NotImplementedError

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class HyperMLP(Model):
    def __init__(self, knob_dim:int, input_dim:int, output_dim:int, bias:bool=True):
        super(HyperMLP, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.bias = bias
        self.ff = nn.Linear(knob_dim, input_dim*output_dim)
        if self.bias: self.b = nn.Linear(knob_dim, output_dim)
        self.apply(self._init_weights)

    def forward(self, k:th.Tensor, x:th.Tensor) -> th.tensor:
        """
            Inputs:
                k:th.Tensor             hypernet conditioning input of the form (B, D)
                x:th.Tensor             input of the form (B, H)
            Outputs:
                h:th.Tensor             encoded examples of the form (B, H')
        """
        k = k.view(-1)
        w = self.ff(k).view(self.out_dim, self.in_dim)
        b = self.b(k).view(-1)
        h = F.linear(x, weight=w, bias=b)
        return h

class HyperEncoder(Model):

    def __init__(self, knob_dim:int=128, input_dim:int=512, hidden_dim:int=128, output_dim:int=16):
        super(HyperEncoder, self).__init__()
        self.down_mlp_1 = HyperMLP(knob_dim=knob_dim, input_dim=input_dim, output_dim=hidden_dim*2)
        self.down_mlp_2 = HyperMLP(knob_dim=knob_dim, input_dim=hidden_dim*2, output_dim=hidden_dim)
        self.up_mlp_2 = HyperMLP(knob_dim=knob_dim, input_dim=hidden_dim, output_dim=hidden_dim*2)
        self.up_mlp_1 = HyperMLP(knob_dim=knob_dim, input_dim=hidden_dim*2, output_dim=latent_dim)
        self.apply(self._init_weights)

    def forward(self, notion:th.Tensor, x:th.Tensor) -> th.Tensor:
        """
            Inputs:
                notion:th.Tensor        embedded concept to learn, eg. "red" or "spherical and plastic"
                x:th.Tensor             a batch of embedded visual examples of the shape (B, H)
            Outputs:
                h:th.Tensor             encoded examples in the notion's conceptual space
        """
        h_1 = F.gelu(self.down_mlp_1(notion, x)) # i -> h*2
        h_2 = F.gelu(self.down_mlp_2(notion, h_1)) # h*2 -> h
        h_3 = F.gelu(self.up_mlp_2(notion, h_2)) + h_1 # h -> h*2
        h_4 = F.gelu(self.up_mlp_1(notion, h_3)) # h*2 -> l
        return h_4

class HyperMem(Model):
    
    def __init__(self, lm_dim:int=512, knob_dim:int=128, input_dim:int=512, hidden_dim:int=128, output_dim:int=16, clip_model:object=None):
        super(HyperMem, self).__init__()
        """
            Inputs:
                lm_dim:int              embedding size of encoded sentence token with LM
                knob_dim:int            target embedding size of the modulating sentence token
                input_dim:int           embedding size of the examples to the AE
                hidden_dim:int          operating hidden size of the AE
                output_dim:int          output size of the AE
        """
        self._d = nn.Parameter(th.empty(0))
        self._d.requires_grad = False

        print(lm_dim)

        self.filter = nn.Linear(in_features=knob_dim, out_features=input_dim) # from baseline
        self.centroid = nn.Linear(in_features=knob_dim, out_features=latent_dim) # from baseline
        self.embedding = nn.Sequential(nn.Linear(in_features=lm_dim, out_features=lm_dim), nn.Linear(in_features=lm_dim, out_features=knob_dim), nn.ReLU()) # from paper
        self.encoder = HyperEncoder(knob_dim=knob_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        self.clip_model = clip_model

    def forward(self, notion:str, x:th.Tensor) -> (th.Tensor, th.Tensor):
        """
            Inputs:
                notion:str              embedded concept to learn, eg. "red" or "spherical and plastic"
                x:th.Tensor             a batch of embedded visual examples of the shape (B, H)
            Outputs:
                z:th.Tensor             encoded examples in the notion's conceptual space
                c:th.Tensor             centroid for the concept's conceptual space
        """
        # Notion embedding
        with th.no_grad():
            text_inputs = clip.tokenize(notion).to(self._d.device)
            e_notion = self.clip_model.encode_text(text_inputs).detach().type('torch.FloatTensor').to(self._d.device) # 1, 512
        e_notion = self.embedding(e_notion) # 1, 128
        # Encoding
        f = self.filter(e_notion)
        c = self.centroid(e_notion)
        h = x * f # B, 512
        z = self.encoder(e_notion, h)
        return z, c


