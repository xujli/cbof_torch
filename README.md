# Torch implementation of the CBoF method

This is an easy to use **Torch** (re)-implementation of the Convolutional Bag-of-Features (CBoF) pooling method (as presented in [Bag-of-Features Pooling for Deep Convolutional Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Passalis_Learning_Bag-Of-Features_Pooling_ICCV_2017_paper.html)). CBoF is a useful tool that allows for **decoupling the size of the representation extracted** from a deep CNN from both the **size** and the **number** of the extracted feature maps. Therefore, it allows for **reducing the size** of the used CNNs as well as  for **improving the scale-invariance** of the models. This repo is inherited by [Keras implementation of CBoF]https://github.com/passalis/keras_cbof

To use the CBoF pooling method, simply insert the *BoF_Pooling* layer between the last convolution and a fully connected layer:
```python

from cbof import BoF_Pooling, initialize_bof_layers
...
self.BoF_pooling = BoF_Pooling()
...
initialize_bof_layers(model, x_train)
```
Remember to initialize the BoF layer (the *initialize_bof_layers()* function automatically initializes all the BoF layers in a torch model). The number of codewords (that defines the dimensionality of the extracted representation) as well as the spatial level must be defined. Two spatial levels are current supported: 0 (no spatial segmentation) and 1 (spatial segmentation in 4 regions).
