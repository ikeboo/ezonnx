# EZONNX


<div align="center"><img src=./assets/top.jpg  width=640/> </div>
<br>

🤗 **Simple and unified API**  

```python
from ezonnx import DinoV3

model = DinoV3('vits16',size=768) # download & create model
res = model("image.jpg")          # inference

# Access output data
print(res.visualized_img)         # visualized image
print(res.class_token, res.patch_tokens)
```

## 📦 Models & Sample Notebooks
- [OCR](examples/ocr.ipynb)
    - PPOCRv5
- [Image classification](examples/image_classification.ipynb)  
    - ImageClassifier(ConvNeXt,ViT,EfficientNetV2,CoAtNet and others.)
    - SigLIP2  (zero-shot classification)
- [Object detection](examples/object_detection.ipynb)
    - YOLO26, YOLO11, YOLOv8
    - YOLO-OBB (v8,11)
    - RF-DETR
    - D-FINE
    - RTMDet (Person detection)
- [Segmentation](examples/segmentation.ipynb)
    - SAM2.1
    - EdgeTAM (SAM2 for mobile)
    - YOLOv8-seg,YOLO11-seg
    - RF-DETR Seg
- [Feature extraction](examples/feature_extraction.ipynb)
    - DINOv2, DINOv3
    - SigLIP2
- [Person pose estimation](examples/object_detection.ipynb)
    - DWPose (whole body)
    - RTMPose (17pts and whole body)
    - ViTPose+ (17pt)
    - RTMO (multi person pose estimation)
    - RTMW (whole body)
    - MotionBERT 3d (3d pose estimation)
- [Depth estimation](examples/image_processing.ipynb)
    - DepthAnythingV2
- [Background removal](examples/image_processing.ipynb)
    - RMBG1.4
    - ORMBG
- [Image inpainting](examples/image_processing.ipynb)
    - LaMa
- [Optical flow](examples/image_processing.ipynb)
    - NeuFlowV2
- [Super resolution](examples/image_processing.ipynb)
    - Real-ESRGAN
- [Image matching](examples/image_matching.ipynb)
    - ALIKED LightGlue
---

## 🛠️ Setup
Install with pip
```sh
pip install git+https://github.com/ikeboo/ezonnx.git
```

---

## ⚡ Quick Start

Ready-to-use sample code for image classification and object detection.

```python
from ezonnx import DinoV2,DinoV3,visualize_images

img_path = "images/cat.jpg"

dinov3 = DinoV3("vits16", size=768)
dinov2 = DinoV2("small", size=768)

out_v3 = dinov3(img_path)
out_v2 = dinov2(img_path)

visualize_images(["Original image","DinoV3 PCA","DinoV2 PCA"], 
                [out_v3.original_img[...,::-1],out_v3.visualized_img, out_v2.visualized_img])
```
<img src=./assets/quickstart.jpg  width=800/> 

Find more notebook samples in the [examples/](examples/) folder.

---

## 📚 Examples

- [Image Classification](examples/image_classification.ipynb)
- [Feature Extraction](examples/feature_extraction.ipynb)
- [Object Detection](examples/object_detection.ipynb)
- [Image Processing](examples/image_processing.ipynb)
- [Segmentation](examples/segmentation.ipynb)
- [OCR](examples/ocr.ipynb)
- [Image Matching](examples/image_matching.ipynb)

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---


## 🙏 Acknowledgements  

Special thanks to Hugging Face and the developers of each AI model integrated in EZONNX. Their open-source contributions and innovative research make it possible to provide powerful, easy-to-use tools for the community.