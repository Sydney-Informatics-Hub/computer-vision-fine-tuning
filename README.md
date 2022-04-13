# Computer Vision Fine Tuning
Fine tune a computer vision model to solve your task locally, on HPC, in a container, or in the cloud!

## Computer Vision Tasks

1. **Classification:** Classify what the image is.  The simplest task.
2. **Object Detection:** Detect and classify objects in images, using bounding boxes. 
3. **Segmentation:** Detect and classify objects in images, using segmentation masks.
4. **Panoptic Segmentation:** Detect and classify everything in an image, using segmentation masks for the whole image. The hardest class!

## Picking Your Model

There are three leading models for computer vision at the moment:

1. Residual Neural Networks (ResNet)
2. You Only Look Once ([YOLO](https://pjreddie.com/darknet/yolo/), [v5](https://github.com/ultralytics/yolov5))
3. Vision Transformers ([ViT](https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch))

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

# Environments

### Technical Requirements

While you can use CPUs for deep learning, you really should only use devices with CUDA capable GPUs.

For a local device, a good build would be:

- OS: Windows 11+WSL2 or Ubuntu (e.g. [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software))
- GPU: RTX 3080 or better
- RAM: 64gb or more
- SSD: biggest NVME you can afford
- CPU: Fast Intel i9 or AMD threadripper

The main priority is to have a very powerful GPU with a lot of CUDA cores and a large amount of fast vRAM. Your GPU's speed and size will determine the speed and complexity of the models that you can train, and will also control the batch size that you can work with.

You also will want to have a decent amount of fast RAM. RAM is very important for high throughput I/O operations, such as when your model is referencing numerous images and annotations. Many models also allow you to cache your images and annotations on RAM to accelerate training/fine tuning.

A large and fast SSD (e.g. NVME) can be a huge benefit, allowing faster I/O processes and supporting a high speed paging file if you do not have sufficient RAM. Caching on a fast SSD can also be a decent alternative if your RAM is not sufficient for caching your data.

For CPUs, our primary spec to target is 

## Locally

You will need a device with a CUDA GPU. 

Current testing has been on a desktop with:
- OS: Windows 10
- GPU: RTX 2070 Super 8gb vRAM
- RAM: 16gb DDR5
- CPU: Ryzen 7 5800x 3.8Ghz 8-core
- SSD: 1tb NVME

This desktop is approximately 2x faster than a GCP Tesla k80 VM.

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

## Docker

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

Run the container, giving it access to your GPU(s). 

You can also mount local files with `-v "$(pwd)"/datasets:/usr/src/datasets`.

```bash
sudo docker run --ipc=host -it --gpus all ultralytics/yolov5:latest
```

Once the container is up and running and you have your data mounted, you can then use all the YOLOv5 functions as normal.

## HPC 

TODO

## Google Colab

Google Colab provides 12 hours a day of GPU node compute per day on the free tier; this represents one of the easiest ways to get access to low cost compute.

There are two major drawbacks of Google Colab:
1. You have a maximum of 12 hours on the free tier.
2. You must have your computer open and running.

# Data Logging

## Weights and Biases Logging

Use weights and biases for data logging and model pipeline version control.

## Grid

TODO: Show how to set up Grid project for this.

# Examples

## Camera Traps Object Detection

### Context 

Camera traps are used to capture images of wildlife, with many applications in conservation biology and ecology. 
The New South Wales National Parks Service has been running a long term wildlife monitoring program called [Wildcount](https://www.environment.nsw.gov.au/-/media/OEH/Corporate-Site/Documents/Animals-and-plants/Native-animals/wildcount-broad-scale-long-term-monitoring-of-fauna-in-nsw-national-parks-2012-2016-200316.pdf). For this program they collected millions of images over several years from sites all over New South Wales, and recorded the species of animals found in these images.

The original data format is a series of directories structured like:

`year/set_of_10_sites/site/subsite`

Each of the final subsite folders contains all of the images captured for that location.

### Google Colab

Notebook on [colab](https://colab.research.google.com/drive/1Q2zV7kYRHT_j630_fKyF08eYEaob-N2e?usp=sharing).

## Weed-AI Integration

TODO: Google Colab showing how to pull a dataset from weed-ai and fine tune segmentation.
