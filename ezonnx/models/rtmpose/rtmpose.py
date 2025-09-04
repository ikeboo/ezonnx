from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RTMPose(Inferencer):
    def __init__(self,
                 identifier:str="m",
                 person_detector:Inferencer=None,
                 kpt_thresh = 0.3,
                 iou_thresh = 0.45,
                 onnx_path:Optional[str]=None):
        # person detector
        if person_detector is None:
            self._person_det = RTMDet("m-person",iou_thresh=iou_thresh)
        else:
            self._person_det = person_detector
        
        # build
        self._check_backbone(identifier,["s","m","l","x",
                                         "m-hand","m-wholebody"])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/RTMPose-ONNX"
            filename = f"rtmpose-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        h,w = self.sess.get_inputs()[0].shape[2:]
        self.input_size_wh = (w,h)
        self.kpt_thresh = kpt_thresh

    def __call__(self,image:Union[str, np.ndarray])-> PoseDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            PoseDetectionResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        # detect person
        det_result:ObjectDetectionResult = self._person_det(image)
        boxes = det_result.boxes
        box_scores = det_result.scores
        if len(boxes)==0:
            kpts = np.empty((0,17,2))
            kpt_scores = np.empty((0,17))

        for box in boxes:
            kpt,kpt_score = self.estimate_keypoints(box,image)
            if 'kpts' in locals():
                kpts = np.vstack([kpts,kpt])
            else:
                kpts = kpt
            if 'kpt_scores' in locals():
                kpt_scores = np.vstack([kpt_scores,kpt_score])
            else:
                kpt_scores = kpt_score

        return PoseDetectionResult(
            original_img = image,
            boxes = boxes,
            kpts = kpts,
            scores = box_scores,
            kpt_scores=kpt_scores,
            kpt_thresh = self.kpt_thresh
        )

    def _preprocess(self,
        img: np.ndarray, 
        input_size_wh: Tuple[int, int] = (192, 256),
        padding=1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            input_size_wh (tuple): Input image size in shape (w, h).

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # get shape of image
        img_shape = img.shape[:2]
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

        # get center and scale
        center, scale = self._bbox_xyxy2cs(bbox, padding=padding)

        # do affine transformation
        resized_img, scale = self._top_down_affine(input_size_wh, scale, center, img)

        # normalize image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        resized_img = np.transpose(resized_img, [2, 0, 1])
        resized_img = resized_img.astype("float32")[None, :]

        return resized_img, center, scale

    def estimate_keypoints(self,
                           box:np.ndarray,
                           frame:np.ndarray
                           )-> Tuple[np.ndarray,np.ndarray]:
        """Crop the person area and estimate keypoints.
        
        Args:
            box (np.ndarray): Bounding box in shape (4,), formatted as
                (left, top, right, bottom)
            frame (np.ndarray): Original image.

        Returns:
            tuple:
            - keypoints (np.ndarray): Estimated keypoints in shape (1, n_keypoints, 2).
            - scores (np.ndarray): Model predict scores of each key points.
        """
        frame_cropped = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
        # h, w = self.sess.get_inputs()[0].shape[2:]
        # model_input_size = (w, h)

        # preprocessing
        input_tensor, center, scale = self._preprocess(frame_cropped, self.input_size_wh)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: input_tensor})
        # postprocessing
        keypoints, scores = self._postprocess(outputs, self.input_size_wh, center, scale)
        # convert coordinates to original
        keypoints[0][:,0]+=int(box[0])
        keypoints[0][:,1]+=int(box[1])
        return keypoints,scores


    def _postprocess(self,
                    outputs: List[np.ndarray],
                    model_input_size: Tuple[int, int],
                    center: Tuple[int, int],
                    scale: Tuple[int, int],
                    simcc_split_ratio: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # use simcc to decode
        simcc_x, simcc_y = outputs
        keypoints, scores = self._decode(simcc_x, simcc_y, simcc_split_ratio)

        # rescale keypoints
        keypoints = keypoints / model_input_size * scale + center - scale / 2

        return keypoints, scores

    def _bbox_xyxy2cs(self,bbox: np.ndarray,
                    padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the bbox format from (x,y,w,h) into (center, scale)

        Args:
            bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
                as (left, top, right, bottom)
            padding (float): BBox padding factor that will be multilied to scale.
                Default: 1.0

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
                (n, 2)
            - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
                (n, 2)
        """
        # convert single bbox from (4, ) to (1, 4)
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        # get bbox center and scale
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale


    def _fix_aspect_ratio(self,bbox_scale: np.ndarray,
                        aspect_ratio: float) -> np.ndarray:
        """Extend the scale to match the given aspect ratio.

        Args:
            scale (np.ndarray): The image scale (w, h) in shape (2, )
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.ndarray: The reshaped image scale in (2, )
        """
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                            np.hstack([w, w / aspect_ratio]),
                            np.hstack([h * aspect_ratio, h]))
        return bbox_scale


    def _rotate_point(self,pt: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate a point by an angle.

        Args:
            pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
            angle_rad (float): rotation angle in radian

        Returns:
            np.ndarray: Rotated point in shape (2, )
        """
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    
    def _get_3rd_point(self,a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): The 1st point (x,y) in shape (2, )
            b (np.ndarray): The 2nd point (x,y) in shape (2, )

        Returns:
            np.ndarray: The 3rd point.
        """
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def _get_warp_matrix(self,
                        center: np.ndarray,
                        scale: np.ndarray,
                        rot: float,
                        output_size: Tuple[int, int],
                        shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        """Calculate the affine transformation matrix that can warp the bbox area
        in the input image to the output size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: A 2x3 transformation matrix
        """
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        # compute transformation matrix
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        # get four corners of the src rectangle in the original image
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        # get four corners of the dst rectangle in the input image
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return warp_mat

    def _top_down_affine(self,
                        input_size: dict, bbox_scale: dict, bbox_center: dict,
                        img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bbox image as the model input by affine transform.

        Args:
            input_size (dict): The input size of the model.
            bbox_scale (dict): The bbox scale of the img.
            bbox_center (dict): The bbox center of the img.
            img (np.ndarray): The original image.

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: img after affine transform.
            - np.ndarray[float32]: bbox scale after affine transform.
        """
        w, h = input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        bbox_scale = self._fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

        # get the affine matrix
        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self._get_warp_matrix(center, scale, rot, output_size=(w, h))

        # do affine transform
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        return img, bbox_scale

    def _get_simcc_maximum(self,simcc_x: np.ndarray,
                        simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get maximum response location and value from simcc representations.

        Note:
            instance number: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
            simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

        Returns:
            tuple:
            - locs (np.ndarray): locations of maximum heatmap responses in shape
                (K, 2) or (N, K, 2)
            - vals (np.ndarray): values of maximum heatmap responses in shape
                (K,) or (N, K)
        """
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        # get maximum value locations
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        # get maximum value across x and y axis
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        # reshape
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

        return locs, vals

    def _decode(self,simcc_x: np.ndarray, simcc_y: np.ndarray,
            simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
        """Modulate simcc distribution with Gaussian.

        Args:
            simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
            simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
            simcc_split_ratio (int): The split ratio of simcc.

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
            - np.ndarray[float32]: scores in shape (K,) or (n, K)
        """
        keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio

        return keypoints, scores