# Computer Vision Fine Tuning
Fine tune a computer vision model to solve your task locally, on HPC, in a container, or in the cloud!

## Computer Vision Tasks

**1. Classification:** Classify what the image is.  The simplest task.
**2. Object Detection:** Detect and classify objects in images, using bounding boxes. 
**3. Segmentation:** Detect and classify objects in images, using segmentation masks.
**4. Panoptic Segmentation:** Detect and classify everything in an image, using segmentation masks for the whole image. The hardest class!

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

## Environments

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

### Docker

Recommended OS: Ubuntu/Debian (native or via WSL2).
This assumes you are using a device with CUDA.

Documentation: [Nvidia Docker](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) and [YOLOv5](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart).

1. Update your NVIDIA drivers.

2. Set up Docker

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```
3. Set up Nvidia Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install `nvidia-docker2`

```bash
suda apt-get update

sudo apt-get install -y nvidia-docker2
```

Restart docker

```bash
sudo systemctl restart docker
```

Test if the installation worked

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

4. Set up YOLO

```bash
sudo docker pull ultralytics/yolov5:latest
```

Run the container, giving it access to your GPU(s). You can also mount local files with `-v "$(pwd)"/datasets:/usr/src/datasets`.

```bash
sudo docker run --ipc=host -it --gpus all ultralytics/yolov5:latest
```

Once the container is up and running and you have your data mounted, you can then use all the YOLOv5 functions as normal.

### HPC 

TODO

### Google Colab

Google Colab provides 12 hours a day of GPU node compute per day on the free tier; this represents one of the easiest ways to get access to low cost compute.
One major drawback is the hard 12 hour limit; you could potentially try to avoid this by running epochs in batchs and exporting checkpoints as you go. 

## Data Logging

### Weights and Biases Logging

Use weights and biases for data logging and model pipeline version control.

### Grid

TODO: Show how to set up Grid project for this.

## Camera Traps Object Detection

Notebook on [colab](https://colab.research.google.com/drive/1Q2zV7kYRHT_j630_fKyF08eYEaob-N2e?usp=sharing).


