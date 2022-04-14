#!/bin/bash

# PBS Script for running Yolov5s 
#PBS -P SIHsandbox 
#PBS -N run_150_01
#PBS -M nathaniel.butterworth@sydney.edu.au 
#PBS -m abe
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q defaultQ

#Load Artemis specific software modules
module load python/3.8.2 magma/2.5.3
module load freetype/2.5.5 libjpeg/6b libpng/1.6.16

# Activate the python env you have previously created with required packages
source /project/SIHsandbox/camtrap/yolo/bin/activate

cd $PBS_O_WORKDIR
#python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
python train.py --img 640 --batch 24 --epochs 300 --data data/datasets_150.yaml --weights yolov5s.pt

