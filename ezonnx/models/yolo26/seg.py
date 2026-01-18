from typing import List,Dict,Optional,Union,Tuple

import numpy as np
import cv2
from ...core.inferencer import Inferencer
from ...data_classes.object_detection import InstanceSegmentationResult
from ...ops.preprocess import (resize_with_aspect_ratio,
                                image_from_path,
                               standard_preprocess)
from ...ops.postprocess import sigmoid,COLORS

class YOLO26Seg(Inferencer):
    """YOLO26 seg model for object detection with ONNX.

    Args:
        onnx_path (str): Path to a local ONNX model file.
        conf_thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.45.
    
    Examples:
        Usage
        ::
            from ezonnx import YOLOSeg, visualize_images
            det = YOLOSeg("/path/to/yolo-seg.onnx") # Please use local weight
            ret = det("images/surf.jpg")
            visualize_images("Detection Result",ret.visualized_img)
    """
    def __init__(self,
                onnx_path:Optional[str]=None,
                conf_thresh:float=0.3,
                iou_thresh:float=0.45,
                size=640,
                ):
    
        if onnx_path is None:
            raise ValueError("Please provide the onnx_path for YOLOSeg model.Remote repo not available yet.")
            # self._check_backbone(identifier,[""])
            # self._check_quantize(quantize,
            #                     [None])
            # # Initialize model
            # repo_id = f""
            # filename = f""
            # self.sess = self._download_and_compile(repo_id, filename, quantize)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def __call__(self,image:Union[str, np.ndarray])-> InstanceSegmentationResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            InstanceSegmentationResult: Inference result containing boxes,classes,scores and masks.
        """
        img = image_from_path(image)
        tensor,r = self._preprocess(img)
        output=self.sess.run(None,{self.input_name:tensor})
        boxes,scores,classes,masks = self._postprocess(output,r)
        return InstanceSegmentationResult(
            original_img=img,
            boxes=boxes,
            classes=classes,
            scores=scores,
            masks=masks
        )

    def _preprocess(self,img:np.ndarray)-> Tuple[np.ndarray,float]:
        """Preprocess the input image for the model.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            Tuple[np.ndarray,float]: Preprocessed image tensor in shape (1, 3, H, W) and resize ratio.
        """

        self.img_height,self.img_width =img.shape[:2]
        padded_img,r = resize_with_aspect_ratio(img,self.size)
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        #単一バッチ、正規化
        tensor = padded_img[None, :]/255
        return tensor, r

    def _postprocess(self, output: List[np.ndarray], ratio) -> Tuple[List]:
        detections = output[0][0]
        if detections.size == 0:
            return [], [], [], np.array([])

        num_masks = output[1].shape[1]
        boxes = detections[:, :4] / ratio
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(int)
        mask_predictions = detections[:, 6:6 + num_masks]

        keep = scores >= self.conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_predictions = mask_predictions[keep]

        if len(scores) == 0:
            return [], [], [], np.array([])

        mask_maps = self._process_mask_output(boxes,
                                              mask_predictions,
                                              output[1])

        result = (boxes.astype(int),
                  scores,
                  class_ids,
                  mask_maps
                )
        return result
    
    def _process_mask_output(self, boxes, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Calculate letterbox parameters
        scale = min(self.size / self.img_height, self.size / self.img_width)
        scaled_height = int(self.img_height * scale)
        scaled_width = int(self.img_width * scale)

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(boxes), self.img_height, self.img_width))
        
        for i in range(len(boxes)):
            # Convert box coordinates to model input space (640x640 with letterbox)
            model_x1 = boxes[i][0] * scale
            model_y1 = boxes[i][1] * scale
            model_x2 = boxes[i][2] * scale
            model_y2 = boxes[i][3] * scale
            
            # Scale to mask dimensions
            scale_x1 = int(np.floor(model_x1 * mask_width / self.size))
            scale_y1 = int(np.floor(model_y1 * mask_height / self.size))
            scale_x2 = int(np.ceil(model_x2 * mask_width / self.size))
            scale_y2 = int(np.ceil(model_y2 * mask_height / self.size))

            # Ensure coordinates are within mask bounds
            scale_x1 = max(0, min(scale_x1, mask_width))
            scale_y1 = max(0, min(scale_y1, mask_height))
            scale_x2 = max(0, min(scale_x2, mask_width))
            scale_y2 = max(0, min(scale_y2, mask_height))

            # Original image coordinates
            x1 = int(np.floor(boxes[i][0]))
            y1 = int(np.floor(boxes[i][1]))
            x2 = int(np.ceil(boxes[i][2]))
            y2 = int(np.ceil(boxes[i][3]))

            # Ensure original coordinates are within image bounds
            x1 = max(0, min(x1, self.img_width))
            y1 = max(0, min(y1, self.img_height))
            x2 = max(0, min(x2, self.img_width))
            y2 = max(0, min(y2, self.img_height))

            if scale_x2 <= scale_x1 or scale_y2 <= scale_y1 or x2 <= x1 or y2 <= y1:
                continue  # skip invalid crops

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            
            if scale_crop_mask.size == 0:
                continue  # skip empty crops
                
            crop_mask = cv2.resize(scale_crop_mask,
                                (x2 - x1, y2 - y1),
                                interpolation=cv2.INTER_CUBIC)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps



