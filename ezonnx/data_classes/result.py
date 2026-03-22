from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
import numpy as np

class Result(BaseModel,ABC):
    """Data class for inference results.

    Args:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR

    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    original_img: np.ndarray

    @property
    def visualized_img(self) -> np.ndarray:
        """Drawn original image.

        Returns:
            np.ndarray: Original image in shape (H, W, 3). BGR
        """
        return self._visualize()
    
    @abstractmethod
    def _visualize(self) -> np.ndarray:
        """Abstract method for visualization.
        Prcess original image and return visualized image.

        Returns:
            np.ndarray: Visualized image in shape (H, W, 3). BGR
        """
        pass
    
