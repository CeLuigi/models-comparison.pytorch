# Deep neural network architectures comparison
Code for the paper *Benchmark Analysis of Representative Deep Neural Network Architectures*

## Dependencies:
* Python 2.7
* PyTorch 4.0.0
* Torchvision

## Citation
If you use our code, please consider cite the following:
* Simone Bianco, Remi Cadene, Luigi Celona, and Paolo Napoletano. Benchmark Analysis of Representative Deep Neural Network Architectures. In arXiv preprint arXiv:*, 2018.
```
@article{bianco2018dnnsbench,
  title={Benchmark Analysis of Representative Deep Neural Network Architectures},
  author={Bianco, Simone and Cadene, Remi and Celona, Luigi and Napoletano, Paolo},
  journal={arXiv preprint arXiv:*},
  year={2018}
```

## Summary

- [Installation](https://github.com/CeLuigi/models-comparison.pytorch#installation)
- [Evaluation on ImageNet](https://github.com/CeLuigi/models-comparison.pytorch#evaluation-on-imagenet)
    - [Accuracy on valset](https://github.com/CeLuigi/models-comparison.pytorch#accuracy-on-validation-set)
    - [Reproducing results](https://github.com/CeLuigi/models-comparison.pytorch#reproducing-results)
- [Documentation](https://github.com/CeLuigi/models-comparison.pytorch#documentation)
    - [Available models](https://github.com/CeLuigi/models-comparison.pytorch#available-models)
        - [Model API](https://github.com/CeLuigi/models-comparison.pytorch#model-api)
        - [model.input_size](https://github.com/CeLuigi/models-comparison.pytorch#modelinput_size)
        - [model.input_space](https://github.com/CeLuigi/models-comparison.pytorch#modelinput_space)
        - [model.input_range](https://github.com/CeLuigi/models-comparison.pytorch#modelinput_range)
        - [model.mean](https://github.com/CeLuigi/models-comparison.pytorch#modelmean)
        - [model.std](https://github.com/CeLuigi/models-comparison.pytorch#modelstd)
        - [model.features](https://github.com/CeLuigi/models-comparison.pytorch#modelfeatures)
        - [model.logits](https://github.com/CeLuigi/models-comparison.pytorch#modellogits)
        - [model.forward](https://github.com/CeLuigi/models-comparison.pytorch#modelforward)
- [Reproducing porting](https://github.com/CeLuigi/models-comparison.pytorch#reproducing)
    - [ResNet*](https://github.com/CeLuigi/models-comparison.pytorch#hand-porting-of-resnet152)
    - [ResNeXt*](https://github.com/CeLuigi/models-comparison.pytorch#automatic-porting-of-resnext)
    - [Inception*](https://github.com/CeLuigi/models-comparison.pytorch#hand-porting-of-inceptionv4-and-inceptionresnetv2)

## Installation

1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)

### Install from pip

3. `pip install pretrainedmodels`

### Install from repo

3. `git clone https://github.com/CeLuigi/models-comparison.pytorch.git`
4. `cd pretrained-models.pytorch`
5. `python setup.py install`


## Quick examples

- To import `pretrainedmodels`:

```python
import pretrainedmodels
```

- To print the available pretrained models:

```python
print(pretrainedmodels.model_names)
> ['mobilenet', 'mobilenetv2', 'shufflenet', 'googlenet', 'fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4',
'inceptionresnetv2', 'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetalarge', 'nasnetamobile', 'cafferesnet101', 'senet154',  'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']
```

- To print the available pretrained settings for a chosen model:

```python
print(pretrainedmodels.pretrained_settings['nasnetalarge'])
> {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth', 'input_space': 'RGB', 'input_size': [3, 331, 331], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1000}, 'imagenet+background': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth', 'input_space': 'RGB', 'input_size': [3, 331, 331], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1001}}
```

- To load a pretrained models from imagenet:

```python
model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
```

**Note**: By default, models will be downloaded to your `$HOME/.torch` folder. You can modify this behavior using the `$TORCH_MODEL_ZOO` variable as follow: `export TORCH_MODEL_ZOO="/local/pretrainedmodels`

- To load an image and do a complete forward pass:

```python
import torch
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

path_img = 'data/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

output_logits = model(input) # 1x1000
```

- To extract features (beware this API is not available for all networks):

```python
output_features = model.features(input) # 1x14x14x2048 size may differ
output_logits = model.logits(output_features) # 1x1000
```

## Few use cases

### Compute imagenet logits

- See [examples/imagenet_logits.py](https://github.com/CeLuigi/models-comparison.pytorch/blob/master/examples/imagenet_logits.py) to compute logits of classes appearance over a single image with a pretrained model on imagenet.

```
$ python examples/imagenet_logits.py -h
> nasnetalarge, resnet152, inceptionresnetv2, inceptionv4, ...
```

```
$ python examples/imagenet_logits.py -a nasnetalarge --path_img data/cat.png
> 'nasnetalarge': data/cat.png' is a 'tiger cat' 
```

### Compute imagenet evaluation metrics

- See [examples/imagenet_eval.py](https://github.com/CeLuigi/models-comparison.pytorch/blob/master/examples/imagenet_eval.py) to evaluate pretrained models on imagenet valset. 

```
$ python examples/imagenet_eval.py /local/common-data/imagenet_2012/images -a nasnetalarge -b 20 -e
> * Acc@1 92.693, Acc@5 96.13
```
    
### Reproducing results

Please see [Compute imagenet validation metrics](https://github.com/CeLuigi/models-comparison.pytorch#compute-imagenet-validation-metrics)


## Documentation

### Model API

Once a pretrained model has been loaded, you can use it that way.

**Important note**: All image must be loaded using `PIL` which scales the pixel values between 0 and 1.

#### `model.input_size`

Attribut of type `list` composed of 3 numbers:

- number of color channels,
- height of the input image,
- width of the input image.

Example:

- `[3, 299, 299]` for inception* networks,
- `[3, 224, 224]` for resnet* networks.


#### `model.input_space`

Attribut of type `str` representating the color space of the image. Can be `RGB` or `BGR`.


#### `model.input_range`

Attribut of type `list` composed of 2 numbers:

- min pixel value,
- max pixel value.

Example:

- `[0, 1]` for resnet* and inception* networks,
- `[0, 255]` for bninception network.


#### `model.mean`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (substract "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.485, 0.456, 0.406]` for resnet* networks.


#### `model.std`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (divide "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.229, 0.224, 0.225]` for resnet* networks.


#### `model.features`

/!\ work in progress (may not be available)

Method which is used to extract the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
print(input_224.size())            # (1,3,224,224)
output = model.features(input_224) 
print(output.size())               # (1,2048,1,1)

# print(input_448.size())          # (1,3,448,448)
output = model.features(input_448)
# print(output.size())             # (1,2048,7,7)
```

#### `model.logits`

/!\ work in progress (may not be available)

Method which is used to classify the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
output = model.features(input_224) 
print(output.size())               # (1,2048, 1, 1)
output = model.logits(output)
print(output.size())               # (1,1000)
```

#### `model.forward`

Method used to call `model.features` and `model.logits`. It can be overwritten as desired.

**Note**: A good practice is to use `model.__call__` as your function of choice to forward an input to your model. See the example bellow.

```python
# Without model.__call__
output = model.forward(input_224)
print(output.size())      # (1,1000)

# With model.__call__
output = model(input_224)
print(output.size())      # (1,1000)
```

#### `model.last_linear`

Attribut of type `nn.Linear`. This module is the last one to be called during the forward pass.

- Can be replaced by an adapted `nn.Linear` for fine tuning.
- Can be replaced by `pretrained.utils.Identity` for features extraction. 

Example when the model is loaded using `fbresnet152`:

```python
print(input_224.size())            # (1,3,224,224)
output = model.features(input_224) 
print(output.size())               # (1,2048,1,1)
output = model.logits(output)
print(output.size())               # (1,1000)

# fine tuning
dim_feats = model.last_linear.in_features # =2048
nb_classes = 4
model.last_linear = nn.Linear(dim_feats, nb_classes)
output = model(input_224)
print(output.size())               # (1,4)

# features extraction
model.last_linear = pretrained.utils.Identity()
output = model(input_224)
print(output.size())               # (1,2048)
```

## Reproducing

### Hand porting of ResNet152

```
th pretrainedmodels/fbresnet/resnet152_dump.lua
python pretrainedmodels/fbresnet/resnet152_load.py
```

### Automatic porting of ResNeXt

https://github.com/clcarwin/convert_torch_to_pytorch

### Hand porting of NASNet, InceptionV4 and InceptionResNetV2

https://github.com/Cadene/tensorflow-model-zoo.torch


## Acknowledgement

Thanks to the deep learning community and especially to the contributers of the pytorch ecosystem.
