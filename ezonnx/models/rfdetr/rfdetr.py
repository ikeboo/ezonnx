from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ...data_classes.object_detection import ObjectDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio
from ...ops.postprocess import sigmoid,box_cxcywh_to_xyxy_numpy,xywh2xyxy

class RFDETR(Inferencer):
    """RF-DETR ONNX model for object detection.

    Args:
        identifier (str): Model identifier, e.g., "n","s","m".
        thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import RFDETR

            rfdetr = RFDETR("s")  # you can choose "n","s","m"
            result = rfdetr("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.classes)  # (N,) array of class labels
            print(result.scores)  # (N,) array of confidence scores
            print(result.visualized_img)  # (H, W, 3) image with
    
    """

    def __init__(self,
                 identifier:Optional[str]=None,
                 thresh:float=0.3,
                 onnx_path:Optional[str]=None):
        
        if onnx_path is None:
            self._check_backbone(identifier,["n","s","m"])
            # Initialize model
            repo_id = f"bukuroo/RF-DETR-ONNX"
            filename = f"rf-detr-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = self.sess.get_inputs()[0].shape[2]
        self.thresh = thresh
    
    def __call__(self,image:Union[str, np.ndarray])-> ObjectDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            ObjectDetectionResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        aspect = image.shape[1] / image.shape[0]
        input_tensor = self._preprocess(image)
        outputs = self.sess.run(None,
                            {self.input_name: input_tensor})
        boxes, classes, scores = self._postprocess(outputs)
        # denormalize boxes to original image size
        if aspect >= 1:
            boxes[:, [0, 2]] *= image.shape[1]
            boxes[:, [1, 3]] *= image.shape[0]*aspect
        else:
            boxes[:, [0, 2]] *= image.shape[1]/aspect
            boxes[:, [1, 3]] *= image.shape[0]
        return ObjectDetectionResult(
            original_img=image,
            boxes=boxes,
            classes=classes,
            scores=scores
        )
    
    def _preprocess(self,image:np.ndarray
                    )-> Tuple[np.ndarray,float]:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        padded_image, _ = resize_with_aspect_ratio(image,self.size)
        input_tensor = standard_preprocess(padded_image)
        return input_tensor
    
    def _postprocess(self,
                     outputs:np.ndarray
                     )-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            outputs (np.ndarray): outputs from the model containing bounding boxes and class scores.

        Returns:
            boxes (np.ndarray): Corresponding bounding boxes.
            classes (np.ndarray): Predicted class labels for each box.
            scores (np.ndarray): Confidence scores for each box.
        """

        boxes, logits = outputs
        prob = sigmoid(logits) 
        flat_prob = prob[0].flatten()
        topk_indexes = np.argsort(flat_prob)[::-1]
        topk_values = np.take_along_axis(flat_prob, topk_indexes, axis=0)
        scores = topk_values
        labels = topk_indexes % logits.shape[2]
        boxes = xywh2xyxy(boxes[0])

        thresh_filter = np.argmin(scores > self.thresh)
        scores = scores[:thresh_filter]
        labels = labels[:thresh_filter]
        boxes = boxes[:thresh_filter]

        return boxes, labels, scores
        
