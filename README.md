<div align="center">
<img src="https://github.com/user-attachments/assets/dcf060f0-5adc-49f0-8a5a-8f2bb397ebc8" width="512" />
<h1><ins>Y</ins>et <ins>A</ins>nother <ins>R</ins>e-implementation <ins>R</ins>epository</h1>
<h3>AI/ML models and experiments, reimplemented for my own understanding.</h3>
</div>

# Overview

I'm a career software engineer with a computer science background, but do not work with AI/ML in my daily duties. This repository aims to take concepts, models, and code from papers and around the web, and re-implement them using the techniques and best practices I've picked up over the years, and hopefully learn some things in the process. If you're like me and learned to code long before you learned about ML, this repo could help you, too.

**All training, measurements, etc. performed on one desktop with an RTX 3090 Ti.**

# Implemented Models

| Model | Purpose | Paper |
| ----- | ------- | ----- |
| [Rectified Flow](#rectified-flow) | Image Synthesis | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, Esser et al. (2024)](https://arxiv.org/pdf/2403.03206) |

# Prerequisites

Assuming python and CUDA are installed, you can ensure necessary packages are available via `pip install -r requirements.txt`

# Rectified Flow

Models and optimizers available on [HuggingFace](https://huggingface.co/benjamin-paine/yarr/tree/main/rectified-flow).

## 1-Channel

### MNIST

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d6c5e94-299e-41d8-8cc5-6bc4718f3dbd" />
  <img src="https://github.com/user-attachments/assets/108c1b2a-35ed-4cfe-af02-4c0eb5765915" /><br />
  Trained for 100 epochs in ~45 minutes.
</div>

### Fashion MNIST

<div align="center">
  <img src="https://github.com/user-attachments/assets/088e0ed2-5fda-4014-8b14-5be3aa0eac1d" />
  <img src="https://github.com/user-attachments/assets/08c6a60e-ee2f-4642-84ec-5fb8d0efc72a" /><br />
  Trained for 100 epochs in ~45 minutes.
</div>

## 3-Channel

### CIFAR-10

<div align="center">
  <img src="https://github.com/user-attachments/assets/e8d5341a-f63e-4a13-a95b-0d9ef64d4dac" />
  <img src="https://github.com/user-attachments/assets/82b4225f-8b8c-4889-8ab8-5a882211d99a" /><br />
  Trained for 100 epochs in ~3 hours.
</div>

### CIFAR-100

<div align="center">
  <img src="https://github.com/user-attachments/assets/7fafee17-2202-4763-82fd-c5cff4b69515" />
  <img src="https://github.com/user-attachments/assets/499631dc-b8b8-4ce7-985c-31f41d258c0e" /><br />
  Trained for 250 epochs in ~7.5 hours.
</div>

### Training Commands

#### Basic Usage

*This should be run from the root of the repository.*

```sh
python -m yarr.trainers.rectified_flow
```

#### Options

*Note: [Weights and Biases](https://wandb.ai/) is a freemium service for monitoring AI/ML training runs, using it will allow you to see details and samples throughout training. Use `--wandb-entity` to pass your team name to use it.*

```
Usage: python -m yarr.trainers.rectified_flow [OPTIONS]

  Train a Rectified Flow model on either MNIST, Fashion-MNIST, CIFAR-10, or CIFAR-100.

Options:
  -lr, --learning-rate FLOAT  Learning rate for the optimizer.  [default: 0.001]
  -e, --num-epochs INTEGER    Number of epochs to train the model.  [default: 100]
  -w, --num-workers INTEGER   Number of workers for the data loader.  [default: 4]
  -b, --batch-size INTEGER    Batch size for training.  [default: 250]
  --wandb-entity TEXT         Weights and Biases entity.
  --resume                    Resume training from the latest checkpoint.
  --fashion-mnist             Use Fashion MNIST dataset instead of MNIST.
  --cifar-10                  Use CIFAR-10 dataset instead of MNIST.
  --cifar-100                 Use CIFAR-100 dataset instead of MNIST.
  --help                      Show this message and exit.
```

# License

All code in this repository is released into the public domain with no guarantees or warranty under [the unlicense.](https://github.com/painebenjamin/yarr/tree/main?tab=Unlicense-1-ov-file#readme)

# Contributions

While this repository is primarily for my own use and likely won't be useful as a library, I would welcome any recommendations others may have on other experiments to conduct or ways my implementations can be improved.

# Citations and Acknowledgments

[Simo Ryu (cloneofsimo), minRF](https://github.com/cloneofsimo/minRF) for the inspiration and reference implementation.

```
@misc{ryu2024minrf,
  author       = {Simo Ryu},
  title        = {minRF: Minimal Implementation of Scalable Rectified Flow Transformers},
  year         = 2024,
  publisher    = {Github},
  url          = {https://github.com/cloneofsimo/minRF},
}
```
