# FaceRecognizer

FaceRecognizer is a tool for searching and identifying persons on images. It consists of two major components:

- **face detector**. [Yolov5-based detector](https://github.com/deepcam-cn/yolov5-face) that infers face bounding boxes and facial landmarks. In order to use model with opencv-dnn it has to be preliminary converted to .onnx format.
- **face extractor**. [AdaFace](https://github.com/mk-minchul/AdaFace) model that represents face crop (inferred with face detector) in a 512-dimentional embedding space. Operating within that space we can compute distances between different embeddings. LibTorch library was chosen for managing AdaFace in C++ code. The authors published AdaFace pretrained models in .ckpt format which is not readable for LibTorch so they need to be converted to .torchscript format.

The code depends on [OpenCV](https://github.com/opencv/opencv) and [LibTorch](https://pytorch.org/cppdocs/#torchscript) libraries.

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