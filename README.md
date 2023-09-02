<!-- Project Title -->
<h1 align="center">YOLOv3 Object Detection Training In Google Colab</h1>

<!-- Shields -->
<p align="center">
  <a href="https://github.com/RANJITHROSAN17/yolov3/stargazers">
    <img src="https://img.shields.io/github/stars/RANJITHROSAN17/yolov3?style=for-the-badge" alt="Stars">
  </a>
  <a href="https://github.com/RANJITHROSAN17/yolov3/issues">
    <img src="https://img.shields.io/github/issues/RANJITHROSAN17/yolov3?style=for-the-badge" alt="Issues">
  </a>
  <a href="https://github.com/RANJITHROSAN17/yolov3/network/members">
    <img src="https://img.shields.io/github/forks/RANJITHROSAN17/yolov3?style=for-the-badge" alt="Forks">
  </a>
</p>

<!-- Project Description -->
<p align="center">
  🚀 Train your own custom YOLOv3 object detection model with ease! 🌟
</p>

<!-- Screenshots or GIFs -->
<p align="center">
  <img src="animation.gif" alt="Demo Animation">
</p>

<!-- Table of Contents -->
## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)

<!-- About Section -->
## About

Welcome to the YOLOv3 Object Detection Training repository! This project provides a comprehensive guide and tools to train your own custom YOLOv3 model for object detection tasks. Whether you're working on surveillance, autonomous vehicles, or any other computer vision project, our project simplifies the process.

🔥 **Key Features:**

- 🖼️ **Custom Dataset Support:** Easily train on your own dataset.
- 🚁 **State-of-the-Art Accuracy:** Utilize the power of YOLOv3 architecture.
- 🧪 **Fine-Tuning Options:** Customize and fine-tune pre-trained models.
- 📈 **Monitoring and Visualization:** Track your training progress with rich visualizations.
- 🤝 **Community-Driven:** Active contributors and open to collaborations.

📚 For detailed documentation and examples, visit our [Wiki](wiki-link).

<!-- Getting Started Section -->
## Getting Started

Get your project up and running in no time! Follow these simple steps:

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- CUDA (for GPU acceleration)

1. Cheack Gpu :

```bash
!nvidia-smi
```

2. Mount Google drive :

```bash
from google.colab import drive
drive.mount('/content/gdrive')
```

3. Clone, configure & compile Darknet:

```bash
!git clone https://github.com/AlexeyAB/darknet
```

```bash
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
```

```bash
!make
```

4. Configure yolov3.cfg file:

```bash
!cp cfg/yolov3.cfg cfg/yolov3_training.cfg
```

```bash
!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 4000/' cfg/yolov3_training.cfg
!sed -i '610 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '689 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=21@' cfg/yolov3_training.cfg
```

5. Create .names and .data files:

```bash
!echo -e 'job\nbeam_number' > data/obj.names
!echo -e 'classes= 2\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /content/weight' > data/obj.data
```

6.  Save yolov3_training.cfg and obj.names files in Google drive:

```bash
!mkdir /content/gdrive/MyDrive/Yolo_v3/yolov3_testing.cfg
!mkdir /content/gdrive/MyDrive/Yolo_v3/classes.txt
```

```bash
!cp cfg/yolov3_training.cfg /content/gdrive/MyDrive/Yolo_v3/yolov3_testing.cfg
!cp data/obj.names /content/gdrive/MyDrive/Yolo_v3/classes.txt
```

7. Create a folder and unzip image dataset:

```bash
!mkdir data/obj
!unzip /content/gdrive/MyDrive/ocr_ds.zip -d data/obj
```

8. Create train.txt file:

```bash
import glob
images_list = glob.glob("data/obj/ocr_ds/*.jpg")
with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))
```

9. Download pre-trained weights for the convolutional layers file:

```bash
!wget https://pjreddie.com/media/files/darknet53.conv.74
```

10. Start training:

```bash
!./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show
# Uncomment below and comment above to re-start your training from last saved weights
#!./darknet detector train data/obj.data cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_training_last.weights -dont_show
```

