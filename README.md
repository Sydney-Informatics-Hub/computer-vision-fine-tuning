# computer-vision-fine-tuning
Fine tune a computer vision to solve your task locally, on HPC, or in the cloud!

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

### Weights and Biases Logging

Use weights and biases for data logging and model pipeline version control.

### Grid

TODO: Show how to set up Grid project for this.

## Camera Traps Object Detection

Notebook on [colab](https://colab.research.google.com/drive/1Q2zV7kYRHT_j630_fKyF08eYEaob-N2e?usp=sharing).


