<br/>
<h1 align="center">Session 12: The Dawn of Transformers
<br/>

<!-- toc -->
 
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)


### Problem Statement
Objective is to train a Spatial transformer on **`CIFAR-10`** dataset for 50 epochs and a explanation on what Spatial Transformer does.

### Spatial Transformers
- CNNs are not invariant to rotation and scale and more general affine transformations(meaning if the input images has more variations the model performs poorly).
- This gives raise to a technique called **`Spatial Transformer Networks`** (STN for short).<br/>
- **STN** allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image.
- We can apply the STN module to the input data directly, or even to the feature maps (output of a convolution layer).
  
  ### Architecture
  
  Spatial transformer networks boils down to three main components :
    - The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
    ```
    (localization): Sequential(
    (0): Conv2d(3, 8, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(8, 10, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
    ```
    
    - The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
    - The sampler uses the parameters of the transformation and applies it to the input image.
<p align="center">
  <img width="400" height="280" src="./images/stn-arch.png">
<p/>
 
<br/>
 
**Colab Notebook** [link](https://colab.research.google.com/drive/1KPC3hC1GiV-Cogv-yo8_7qzcV1DgheBW#scrollTo=sGlfCagJJSaD)<br/>
 
**GitHub Notebook** [link](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/12%20Dawn%20of%20Transformers/Saptial%20Transformers/SpatialTransformer.ipynb)<br/>
 

### Model Architecture

```
Net(
  (conv1): Conv2d(3, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=500, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(3, 8, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(8, 10, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=160, out_features=32, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=32, out_features=6, bias=True)
  )
)
```
### Model Training logs
 
Adding last 5 epochs for training here
```
Train Epoch: 44 [0/50000 (0%)]	Loss: 1.421886
Train Epoch: 44 [32000/50000 (64%)]	Loss: 1.730358

Test set: Average loss: 1.1846, Accuracy: 5866/10000 (59%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 1.334839
Train Epoch: 45 [32000/50000 (64%)]	Loss: 1.357687

Test set: Average loss: 1.1469, Accuracy: 6005/10000 (60%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 1.443942
Train Epoch: 46 [32000/50000 (64%)]	Loss: 1.178336

Test set: Average loss: 1.2016, Accuracy: 5865/10000 (59%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 1.353830
Train Epoch: 47 [32000/50000 (64%)]	Loss: 1.309253

Test set: Average loss: 1.1982, Accuracy: 5742/10000 (57%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 1.397684
Train Epoch: 48 [32000/50000 (64%)]	Loss: 1.397693

Test set: Average loss: 1.2238, Accuracy: 5763/10000 (58%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 1.472536
Train Epoch: 49 [32000/50000 (64%)]	Loss: 1.431374

Test set: Average loss: 1.1607, Accuracy: 5908/10000 (59%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 1.189716
Train Epoch: 50 [32000/50000 (64%)]	Loss: 1.345627

Test set: Average loss: 1.1545, Accuracy: 6000/10000 (60%)

```

**complete model training and validation logs** can viewed [here](https://github.com/RajamannarAanjaram/TSAI-Assignment/blob/master/12%20Dawn%20of%20Transformers/Saptial%20Transformers/Logs.md)
 
### Model Results
 
<p align="center">
  <img width="400" height="280" src="./images/out.png">
<p/>

 
 


### Contributors
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
### References
 
https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html
