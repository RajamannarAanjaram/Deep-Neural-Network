<br/>
<h1 align="center">Session 13: ViT
<br/>

<!-- toc -->
 
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)
  
### Objective
To train a **`ViT`** model for Cats and Dogs based on this [blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/).
    
### Dataset
Dataset is downloaded from Kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)<br/>
Dataset contains
    
```
Train Data: 25000
Test Data: 12500
```
### Model using vit-pytorch linformer

In this model the image is split into 7x7 patches.
    
**Model Parameters**
```
Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
```
**Model architecture**
```
ViT(
  (to_patch_embedding): Sequential(
    (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
    (1): Linear(in_features=3072, out_features=128, bias=True)
  )
  (transformer): Linformer(
    (net): SequentialSequence(
      (layers): ModuleList(
        (0): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
    .
    .   ###This repreats for another 10 times and we are using 12 Heads
    .
    .
   (11): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
  )
  (to_latent): Identity()
  (mlp_head): Sequential(
    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=128, out_features=2, bias=True)
  )
) 
```
**Training logs**<br/>

Model is trained for 20 epochs and only first 5 epoch logs are shown.
```
HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 1 - loss : 0.6979 - acc: 0.5000 - val_loss : 0.6959 - val_acc: 0.5028

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 2 - loss : 0.6948 - acc: 0.5008 - val_loss : 0.6939 - val_acc: 0.4986

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 3 - loss : 0.6944 - acc: 0.5028 - val_loss : 0.7073 - val_acc: 0.4972

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 4 - loss : 0.6939 - acc: 0.5074 - val_loss : 0.6988 - val_acc: 0.5000

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))
Epoch : 5 - loss : 0.6936 - acc: 0.5045 - val_loss : 0.6924 - val_acc: 0.5342
```
    
Notebook [link](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/13%20ViT/Cat%20Dogs/CatDogViT.ipynb):- [CatDogViT](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/13%20ViT/Cat%20Dogs/CatDogViT.ipynb)
    
### Model using timm library- Transfer Learing
 
**Model Parameters**<br/>
In this model the image is split into 14x14 patches.

```
dim=768  
seq_len=196+1,  # 14x14 patches + 1 cls-token
depth=12
heads=8
image_size=224
patch_size=16
num_classes=2
channels=3
```
**Model Architecture**<br/>

By Default while importing, the package has target of 1000 classes. Changing it to two using the floowing code`model.head = nn.Linear(768, 2)`
    
```
    VisionTransformer(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        (norm): Identity()
      )
      (pos_drop): Dropout(p=0.0, inplace=False)
      (blocks): Sequential(
        (0): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        
        ...** (the above block is repeated 11 times)**
                   
        (11): Block(
          (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): Attention(
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (pre_logits): Identity()
      (head): Linear(in_features=768, out_features=2, bias=True)
```
**Training Logs**<br/>
 
Attaching last 5 epochs
```
100%|██████████| 313/313 [07:48<00:00,  1.50s/it]
  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 15 - loss : 0.0361 - acc: 0.9854 - val_loss : 0.0550 - val_acc: 0.9792

100%|██████████| 313/313 [07:47<00:00,  1.49s/it]
  0%|          | 0/313 [00:00<?, ?it/s]
 Epoch : 16 - loss : 0.0318 - acc: 0.9872 - val_loss : 0.0543 - val_acc: 0.9816

100%|██████████| 313/313 [07:45<00:00,  1.49s/it]
  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 17 - loss : 0.0336 - acc: 0.9864 - val_loss : 0.0620 - val_acc: 0.9788

100%|██████████| 313/313 [07:44<00:00,  1.48s/it]
  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 18 - loss : 0.0312 - acc: 0.9882 - val_loss : 0.0515 - val_acc: 0.9802

100%|██████████| 313/313 [07:45<00:00,  1.49s/it]
  0%|          | 0/313 [00:00<?, ?it/s]
Epoch : 19 - loss : 0.0326 - acc: 0.9873 - val_loss : 0.0796 - val_acc: 0.9757

100%|██████████| 313/313 [07:44<00:00,  1.48s/it]
Epoch : 20 - loss : 0.0329 - acc: 0.9874 - val_loss : 0.0591 - val_acc: 0.9778
```
                                       
Notebook [link](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/13%20ViT/Cat%20Dogs/CatDogs_TransferLearning.ipynb):- [CatDogs-TransferLearning.ipynb](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/13%20ViT/Cat%20Dogs/CatDogs_TransferLearning.ipynb)
                                       
### References
Timm package:
library installation and model listing:- https://rwightman.github.io/pytorch-image-models/  <br/>
ViT model building:- https://rwightman.github.io/pytorch-image-models/models/vision-transformer/

### Contributors
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
