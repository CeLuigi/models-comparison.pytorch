# Benchmark Analysis of Representative Deep Neural Network Architectures ([IEEE Access](https://ieeexplore.ieee.org/document/8506339))

## Citation
If you use our code, please consider cite the following:
* Simone Bianco, Remi Cadene, Luigi Celona, and Paolo Napoletano. Benchmark Analysis of Representative Deep Neural Network Architectures. In _IEEE Access_, _volume 6_, _issue 1_, pp. 2169-3536, 2018.
```
@article{bianco2018dnnsbench,
 author = {Bianco, Simone and Cadene, Remi and Celona, Luigi and Napoletano, Paolo},
 year = {2018},
 title = {Benchmark Analysis of Representative Deep Neural Network Architectures},
 journal = {IEEE Access},
 volume = {6},
 pages = {64270-64277},
 doi = {10.1109/ACCESS.2018.2877890},
 ISSN = {2169-3536},
}
```

## Abstract
This paper presents an in-depth analysis of the majority of the deep neural networks (DNNs) proposed in the state of the art for image recognition. For each DNN, multiple performance indices are observed, such as recognition accuracy, model complexity, computational complexity, memory usage, and inference time. The behavior of such performance indices and some combinations of them are analyzed and discussed. To measure the indices, we experiment the use of DNNs on two different computer architectures, a workstation equipped with a NVIDIA Titan X Pascal, and an embedded system based on a NVIDIA Jetson TX1 board. This experimentation allows a direct comparison between DNNs running on machines with very different computational capacities. This paper is useful for researchers to have a complete view of what solutions have been explored so far and in which research directions are worth exploring in the future, and for practitioners to select the DNN architecture(s) that better fit the resource constraints of practical deployments and applications. To complete this work, all the DNNs, as well as the software used for the analysis, are available online.

## Architectures
We collect 40 state-of-the-art Deep neural network architectures already trained on ImageNet-1k.
- [AlexNet](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#alexnet)
- [BNInception](https://github.com/CeLuigi/models-comparison.pytorch/wiki/BNInception-model)
- [CaffeResNet101](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Other-models/#caffe-resnet)
- [DenseNet121](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#densenet)
- [DenseNet161](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#densenet)
- [DenseNet169](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#densenet)
- [DenseNet201](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#densenet)
- [DenseNet201](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#densenet)
- [DualPathNet68](https://github.com/CeLuigi/models-comparison.pytorch/wiki/DPN*-models)
- [DualPathNet92](https://github.com/CeLuigi/models-comparison.pytorch/wiki/DPN*-models)
- [DualPathNet98](https://github.com/CeLuigi/models-comparison.pytorch/wiki/DPN*-models)
- [DualPathNet107](https://github.com/CeLuigi/models-comparison.pytorch/wiki/DPN*-models)
- [DualPathNet131](https://github.com/CeLuigi/models-comparison.pytorch/wiki/DPN*-models)
- [FBResNet152](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Other-models#facebook-resnet)
- [GoogLeNet](https://github.com/CeLuigi/models-comparison.pytorch/wiki/GoogLeNet-model)
- [InceptionResNetV2](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Inception*-models)
- [InceptionV3](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Inception*-models)
- [InceptionV4](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Inception*-models)
- [MobileNet](https://github.com/CeLuigi/models-comparison.pytorch/wiki/MobileNet*-models)
- [MobileNetV2](https://github.com/CeLuigi/models-comparison.pytorch/wiki/MobileNet*-models)
- [NASNet-A-Large](https://github.com/CeLuigi/models-comparison.pytorch/wiki/NASNet*-models)
- [NASNet-A-Mobile](https://github.com/CeLuigi/models-comparison.pytorch/wiki/NASNet*-models)
- [ResNeXt101_32x4d](https://github.com/CeLuigi/models-comparison.pytorch/wiki/ResNeXt*-models)
- [ResNeXt101_64x4d](https://github.com/CeLuigi/models-comparison.pytorch/wiki/ResNeXt*-models)
- [ResNet101](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#resnet)
- [ResNet152](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#resnet)
- [ResNet18](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#resnet)
- [ResNet34](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#resnet)
- [ResNet50](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#resnet)
- [SENet154](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [SE-ResNet50](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [SE-ResNet101](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [SE-ResNet152](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [SE-ResNeXt50_32x4d](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [SE-ResNeXt101_32x4d](https://github.com/CeLuigi/models-comparison.pytorch/wiki/SENet*-models)
- [ShuffleNet](https://github.com/CeLuigi/models-comparison.pytorch/wiki/ShuffleNet-model)
- [SqueezeNet1_0](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#squeezenet)
- [SqueezeNet1_1](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#squeezenet)
- [VGG11](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG13](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG16](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG19](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG11_BN](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG13_BN](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG16_BN](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [VGG19_BN](https://github.com/CeLuigi/models-comparison.pytorch/wiki/TorchVision-models#vgg)
- [Xception](https://github.com/CeLuigi/models-comparison.pytorch/wiki/Xception-model)

## Acknowledgement
Thanks to the deep learning community and especially to the contributers of the PyTorch ecosystem.
