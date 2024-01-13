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
from transformers import BertTokenizer, BertModel
from config import *
device = "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLIP_AE_Encode(nn.Module):
	def __init__(self, hidden_dim, latent_dim, isAE=False):
		super(CLIP_AE_Encode, self).__init__()
		# Build Encoder
		self.fc1 = nn.Linear(512, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, latent_dim)
		self.relu = nn.ReLU(inplace=True)

		if isAE:
			self.filter = nn.Parameter(torch.ones((512)))
		else:
			self.filter = nn.Parameter(torch.rand((512)))

	def forward(self, clip_model, images):
		with torch.no_grad():
			emb = clip_model.encode_image(images).float()
		out = emb * self.filter
		out = self.relu(self.fc1(out))
		z = self.fc2(out)

		return z
     

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
        self.ff = nn.Sequential(
            nn.Linear(knob_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim*output_dim)
        )
        if self.bias: self.b = nn.Parameter(th.zeros((output_dim)))
        self.apply(self._init_weights)

    def get_weights(self, k:th.Tensor) -> th.Tensor:
        """
            Inputs:
                k:th.Tensor             hypernet conditioning input of the form (B, D)
            Outputs:
                w:th.Tensor             predicted MLP weights (1, in_dim*out_dim)
        """
        w = self.ff(k).view(-1)
        if self.bias: w = torch.hstack((w, self.b.view(-1))) 
        return w

    def forward(self, k:th.Tensor, x:th.Tensor) -> th.tensor:
        """
            Inputs:
                k:th.Tensor             hypernet conditioning input of the form (B, D)
                x:th.Tensor             input of the form (B, H)
            Outputs:
                h:th.Tensor             encoded examples of the form (B, H')
        """
        W = self.ff(k).view(self.out_dim, self.in_dim) # H', H
        h = (x @ th.t(W)) # B, H'
        if self.bias: h += self.b.repeat(h.shape[0], 1)
        return h
    

class HyperEncoder(Model):

    def __init__(self, knob_dim:int=128, input_dim:int=512, hidden_dim:int=128, output_dim:int=16):
        super(HyperEncoder, self).__init__()
        self.layers = nn.Sequential(
            HyperMLP(knob_dim=knob_dim, input_dim=input_dim, output_dim=256),
            HyperMLP(knob_dim=knob_dim, input_dim=256, output_dim=512),
            HyperMLP(knob_dim=knob_dim, input_dim=512, output_dim=256),
            HyperMLP(knob_dim=knob_dim, input_dim=256, output_dim=output_dim),
        )
        self.apply(self._init_weights)

    def get_weights(self, notion:th.Tensor) -> th.Tensor:
        return torch.hstack([layer.get_weights(notion) for layer in self.layers])

    def forward(self, notion:th.Tensor, x:th.Tensor) -> th.Tensor:
        """
            Inputs:
                notion:th.Tensor        embedded concept to learn, eg. "red" or "spherical and plastic"
                x:th.Tensor             a batch of embedded visual examples of the shape (B, H)
            Outputs:
                h:th.Tensor             encoded examples in the notion's conceptual space
        """
        for layer in self.layers: x = F.relu(layer(notion, x))
        return x


class HyperMem(Model):
    
    def __init__(self, lm_dim:int=768, knob_dim:int=128, input_dim:int=512, hidden_dim:int=128, output_dim:int=16):
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

        self.filter = nn.Linear(in_features=knob_dim, out_features=input_dim)
        self.centroid = nn.Linear(in_features=knob_dim, out_features=latent_dim)
        self.embedding = nn.Sequential(nn.Linear(lm_dim, lm_dim//2), nn.Linear(lm_dim//2, knob_dim))
        self.encoder = HyperEncoder(knob_dim=knob_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for name, param in self.bert.named_parameters(): param.requires_grad = False
        self.bert.eval()

    def get_weights(self, notion:str) -> dict:
        """
            Get all the hypernetwork weights predicted from an input task embedding
            Inputs:
                k:th.Tensor             hypernet conditioning input of the form (B, D)
            Outputs:
                {}:dict                 all the network weights
        """
        # Notion embedding
        with th.no_grad():
            t_notion = self.bert_tokenizer(notion, return_tensors="pt").to(self._d.device)
            e_notion = self.bert(t_notion.input_ids).last_hidden_state[:, 0]
            k = F.relu(self.embedding(e_notion)) # 1, 128
        # HyperNet
        w_centroid = self.centroid(k).view(-1)
        w_filt = self.filter(k).view(-1)
        w_enc = self.encoder.get_weights(k)
        return torch.hstack((w_filt, w_enc, w_centroid))

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
            t_notion = self.bert_tokenizer(notion, return_tensors="pt").to(self._d.device)
            e_notion = self.bert(t_notion.input_ids).last_hidden_state[:, 0]
        e_notion = F.relu(self.embedding(e_notion)) # 1, 128
        # Encoding
        filter = self.filter(e_notion)
        c = self.centroid(e_notion)
        h = x * filter # B, 512
        z = self.encoder(e_notion, h)
        return z, c


