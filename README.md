# computer-vision-fine-tuning
Fine tune a computer vision model to solve your task locally, on HPC, or in the cloud!

## Picking Your Model

There are three leading models for computer vision at the moment:

1. Residual Neural Networks (ResNets)
2. You Only Look Once (YOLOs)
3. Vision Transformers (ViT)

ResNets and YOLOs are both based on convolutional neural networks. ViT models apply transformer model architecture. 

Currently, ResNets and YOLOs are the easiest to work with and the most broadly supported. They also can be fine tuned or even trained on surprisingly small datasets.
ViT models require enormous sums of data for fine tuning, which makes them slightly less attractive in research use cases where datasets are often on the smaller side. 
In this repository, I will first focus on demonstrating workflows for using ResNet and YOLO architectures.
YOLO models often excel in terms of inference / prediction speed, which is very attractive when considering possible production or deployment scenarios where computational resources may be at a premium or speed is a requirement.

## Data Preperation

Before you get started, you need to have images and annotions.

Some general advice:

- When collecting a dataset of images, work to make sure that your training data contains the same sort of variability that you would see in a deployment scenario. You want to make sure that your model is exposed to a diverse set of images that show your objects in a variety of different positions and sizes in the frame, on a wide variety of backgrounds.
- Try to have balanced classes. 1000 images per class is a great starting point.
- Include blank or "null" images, so that your model can get better at understanding what the background is.

Tools for making annotations:

- [makesense](https://www.makesense.ai/)
- [roboflow](https://roboflow.com/)
- CVAT

## Setup and install

TODO

### Locally

On MacOS/Linux
```bash
conda create -n finetune python=3.9

conda activate finetune

pip install -r requirements.txt
```

On Windows
```powershell
conda create -n finetune python=3.9

conda activate finetune

pip install -r requirements.txt
```

### Weights and Biases Logging

Use weights and biases for data logging and model pipeline version control.

### Grid

TODO: Show how to set up Grid project for this.

## Camera Traps Object Detection

Notebook on [colab](https://colab.research.google.com/drive/1Q2zV7kYRHT_j630_fKyF08eYEaob-N2e?usp=sharing).


