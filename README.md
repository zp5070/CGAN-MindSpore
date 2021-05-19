# CGAN-MindSpore

**Unofficial** implementation of [**Conditional Generative Adversarial Nets**](https://arxiv.org/abs/1411.1784v1), using [**MindSpore**](https://gitee.com/mindspore/mindspore).

## Preparation

- **Prerequisites**
    - MindSpore (1.2.0+)
    - Python 3.7.5 

- **Dataset**
    - MNIST Dataset
    - We follow the settings of infoGAN, kindly refer to [
    pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections) for more dataset preparation details



## Quick Start

Exemplar commands are listed here for a quick start.



### Training

```python
python train.py
```



### Distributed Training (Ascend 910 **Only**)

Still in progress