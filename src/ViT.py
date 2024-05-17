import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torchvision.models import resnet18

'''
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")
ngpu= 1

print("驱动为：",device)
print("GPU型号： ",torch.cuda.get_device_name(0))
print("Hello World!")
'''

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
'''
import numpy as np

vitfile = np.load('./src/ViT-B_16.npz')
vitfile.files     #显示权重文件包含了数组名
vitfile['cls']    #查看数组cls的内容，cls为Numpy数组名
vitfile['cls'].shape

for item in vitfile.files:
    print(item)
'''

img = Image.open('./src/test.jpg')
'''
print("finish open the picture")
fig = plt.figure()
plt.imshow(img)
'''

#resize to ImageNet size
transfrom = Compose([Resize((224, 224)), ToTensor()])
x = transfrom(img)
x = x.unsqueeze(0)

patch_size = 16 # 16 pixels
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        # 生成一个维度为emb_size的向量当作cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b,_,_,_ = x.shape
        x = self.projection(x)
        # 将cls_token扩展b次
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # print(cls_tokens.shape)
        # print(x.shape)
        # 将cls_token在维度1扩展到输入上
        x = torch.cat([cls_tokens, x], dim = 1)
        # 添加位置编码
        # print(x.shape, self.positions.shape)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queiries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries  = rearrange(self.queiries(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim = -1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
patches_embedded = PatchEmbedding()(x)
TransformerEncoder()(patches_embedded).shape

model = resnet18()
# summary(model, input_size=[(3, 256, 256)], batch_size=2, device="cpu")

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        
class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size = emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )