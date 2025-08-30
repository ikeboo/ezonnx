import cv2
import numpy as np
from .result import Result

class OpticalFlowResult(Result):
    """Data class for optical flow results.

    Args:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        flow (np.ndarray): Optical flow in shape (H, W, 2). dtype=float32
        magnitude (np.ndarray): Magnitude of the optical flow in shape (H, W). dtype=float32
        angle (np.ndarray): Angle of the optical flow in shape (H, W). dtype=float32
        visualized_img (np.ndarray): Processed image with optical flow drawn.(H, W, 3). BGR
    """
    previous_img: np.ndarray
    flow: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        """Get the magnitude map of the optical flow.

        Returns:
            np.ndarray: Magnitude of the optical flow in shape (H, W). dtype=float32
        """
        return np.sqrt(np.sum(np.square(self.flow), axis=2))
    
    @property
    def angle(self) -> np.ndarray:
        """Get the angle map of the optical flow.

        Returns:
            np.ndarray: Angle of the optical flow in shape (H, W). dtype=float32
        """
        return np.arctan2(self.flow[..., 1], self.flow[..., 0])

    def _visualize(self) -> np.ndarray:
        """Return the RGB representation of the optical flow.

        Returns:
            np.ndarray: RGB representation of the optical flow in shape (H, W, 3). BGR
        """
        return _draw_flow(self.flow, self.original_img)
    
def _draw_flow(flow, image, boxes=None):
    flow_img = _flow_to_image(flow, 35)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

    combined = cv2.addWeighted(image, 0.5, flow_img, 0.6, 0)
    if boxes is not None:
        white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        new_image = cv2.addWeighted(image, 0.7, white_background, 0.4, 0)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            new_image[y1:y2, x1:x2] = combined[y1:y2, x1:x2]

        combined = new_image

    return combined

def _flow_to_image(flow, maxrad=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    rad = np.sqrt(u ** 2 + v ** 2)
    if maxrad is None:
        maxrad = max(-1, np.max(rad))

    eps = np.finfo(float).eps
    u = np.clip(u, -maxrad+5, maxrad-5)
    v = np.clip(v, -maxrad+5, maxrad-5)

    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    img = _compute_color(u, v)

    return np.uint8(img)


def _make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

colorwheel = _make_color_wheel()

def _compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img