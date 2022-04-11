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

- [makesense](https://www.makesense.ai/): Pros: free, can keep your data private, built in COCO object detection. Cons: Less robust ecosystem, limited scalability.
- [roboflow](https://roboflow.com/): Pros: industry standard, well integrated with other platforms, scalable. Cons: you have to upload your data and it won't be private.
- CVAT

## Setup and install

TODO

### Locally

You will need a device with a CUDA GPU. 

Current testing has been on a desktop with:
- OS: Windows 10
- GPU: RTX 2070 Super 8gb vRAM
- RAM: 16gb 
- CPU: Ryzen 7 5800x 3.8Ghz 8-core

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

### Google Colab

Google Colab provides 12 hours a day of GPU node compute per day on the free tier; this represents one of the easiest ways to get access to low cost compute.
One major drawback is the hard 12 hour limit; you could potentially try to avoid this by running epochs in batchs and exporting checkpoints as you go. 

## Camera Traps Object Detection

Notebook on [colab](https://colab.research.google.com/drive/1Q2zV7kYRHT_j630_fKyF08eYEaob-N2e?usp=sharing).


