# FaceRecognizer

FaceRecognizer is a tool for searching and identifying persons on images. It consists of two major components:

- **face detector**. [Yolov5-based detector](https://github.com/deepcam-cn/yolov5-face) that infers face bounding boxes and facial landmarks. In order to use model with opencv-dnn it has to be preliminary converted to .onnx format.
- **face extractor**. [AdaFace](https://github.com/mk-minchul/AdaFace) model that represents face crop (inferred with face detector) in a 512-dimentional Euclidean embedding space. Operating within that space we can compute distances between different embeddings. LibTorch library was chosen for managing AdaFace in C++ code. The authors published AdaFace pretrained models in .ckpt format which is not readable for LibTorch so they need to be converted to .torchscript format.

The code depends on [OpenCV](https://github.com/opencv/opencv) and [LibTorch](https://pytorch.org/cppdocs/#torchscript) libraries.

## Optimization trick

In order to reduce CPU consumption and increase recognizer latency we may limit *face detector* performance by making it detect face only with some given frequency. However, the dummy limitation of detector activity will lead to "gaps" between face positions in a frame sequence. Generally, we want the face position to update smoothly, so I chose The Kalman Filter as a simple tool for filling up the gaps. It learns the dynamics of a moving face and predicts the missing positions. The state vector consists of center, scale and aspect ratio of the face bounding box as [SORT](https://arxiv.org/pdf/1602.00763.pdf) suggests. Unlike the original paper I give high attention to face detector results (low measurement noise) and not so high attention to filter results (higher process noise) because *face extractor* strongly depends on the accurate face localization.

At the current time the filter tracks only one face.

## Demos

Frontal result             |  Webcam result
:-------------------------:|:-------------------------:
![](dox/frontal_result.gif)  |  ![](dox/webcam_result.gif)

## Usage

Note that you should have some embeddings database holding person embeddings you want to identify. Without that base FaceRecognizer tool will simply act as a face detector. This base can be created with FaceCollector tool which extracts embeddings from given photos and stores it in .xml file.

Run FaceCollector
```bash
./FaceCollector -input path/to/input_dir -output path/to/embeddings.xml
```

Run FaceRecognizer without person embeddings
```bash
./FaceRecognizer -input path/to/video [-args]
```

Run FaceRecognizer with person embeddings
```bash
./FaceRecognizer -input path/to/video -persons_file path/to/embeddings.xml [-args]
```

## Acknowledgments

```
@article{YOLO5Face,
title = {YOLO5Face: Why Reinventing a Face Detector},
author = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
booktitle = {ArXiv preprint ArXiv:2105.12931},
year = {2021}
}
```

```
@inproceedings{kim2022adaface,
title={AdaFace: Quality Adaptive Margin for Face Recognition},
author={Kim, Minchul and Jain, Anil K and Liu, Xiaoming},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```

```
@inproceedings{Bewley2016_sort,
author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
title={Simple online and realtime tracking},
year={2016},
pages={3464-3468},
keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
doi={10.1109/ICIP.2016.7533003}
}
```
