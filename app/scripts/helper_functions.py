import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import List
from PIL import Image
from skimage import measure
import cv2
from typing import Tuple, List, Any, Dict, Union, Optional

import torch
import transformers
import accelerate

import os
import sys
import torch
from scripts.setup import GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, SAM_CHECKPOINT_PATH

from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from groundingdino.util.inference import  Model

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
###########################################################################################################
###########################################################################################################
# GROUNDING DINO
###########################################################################################################
###########################################################################################################
class GroundingDINO:
    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        
    def predict_with_captions(self, image: np.ndarray , text_prompt: str)-> Tuple[np.ndarray, List[str]]:
        # Ensure image is in BGR format
        if image.shape[2] == 3 and image.dtype == 'uint8':
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image  # Assume it's already in BGR format
        
        # Perform prediction
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=True):
                detections, phrases = self.model.predict_with_caption(
                    image=image_bgr,
                    caption=text_prompt,
                    box_threshold=self.BOX_THRESHOLD,
                    text_threshold=self.TEXT_THRESHOLD,
                )
        else:
            detections, phrases = self.model.predict_with_caption(
                image=image_bgr,
                caption=text_prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
            )
        
        # Print GPU memory usage if CUDA is available
        if torch.cuda.is_available():
            print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        return detections, phrases
    
###########################################################################################################
###########################################################################################################
# SAM
###########################################################################################################
###########################################################################################################

class SAMSegmenter:
    def __init__(self, checkpoint_path: str, model_type:str ='vit_h', device: str ='cuda')-> None:
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        
        # Instantiate SAM model
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        
        # Instantiate SAM predictor and mask generator
        self.sam_predictor = SamPredictor(self.sam_model)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
    
    def segment(self, image: np.ndarray, xyxy: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        # Convert xyxy to numpy array if it's a tensor
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        print(f"GPU memory before segmentation: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        print(f"GPU memory after segmentation: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        return np.array(result_masks)
    
    def make_sam_mask(self, boolean_mask: np.ndarray) -> List[List[float]]:
        binary_mask = boolean_mask.astype(int)
        contours = measure.find_contours(binary_mask, 0.5)
        mask_points = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            mask_points.append(segmentation)
        return mask_points
    
    def make_annotations(self, detections: 'Detections') -> Optional[List[Dict[str, Union[str, List[Dict[str, Union[str, float, List[List[float]]]]]]]]]:
        if len(detections.xyxy) == 0:
            return None

        annotations = [{"name": f"image id: {detections.tracker_id}", "data": []}]

        for i in range(len(detections.xyxy)):
            annotations[0]["data"].append({
                "label": detections.data[i],
                "score": round((detections.confidence[i] * 100), 2),
                "points": self.make_sam_mask(detections.mask[i]),
            })

        return annotations

###########################################################################################################
###########################################################################################################
# sTABLE DIFFUSION
###########################################################################################################
###########################################################################################################
class StableDiffusionInpainter:
    def __init__(self, pretrained_model_path: str, torch_dtype: torch.dtype = torch.float16, device: str = 'cuda') -> None:        
        self.device = device
        self.torch_dtype = torch_dtype
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
        ).to(device)
        
    def generate_image(self, image: Image.Image, mask: Image.Image, prompt: str, negative_prompt: str, seed: int) -> Image.Image:        # Resize for inpainting
        w, h = image.size
        in_image = image.resize((512, 512))
        in_mask = mask.resize((512, 512))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            image=in_image,
            mask_image=in_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
        )
        result = result.images[0]

        return result.resize((w, h))
###########################################################################################################
###########################################################################################################
# OTHER FUNCTIONS
###########################################################################################################
###########################################################################################################
def load_grounding_dino_model(config_path: str, checkpoint_path: str) -> 'Model':
    grounding_dino_model = Model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
    )
    return grounding_dino_model

def process_image_with_grounding_dino(grounding_dino: 'GroundingDINO', image: np.ndarray, text_prompt: str) -> 'Detections':
    detections, phrases = grounding_dino.predict_with_captions(image, text_prompt)
    for i, phrase in enumerate(phrases):
        detections.data[i] = {'class_id': phrase}
    return detections

def plot_images_grid(mask_dir: str, images: List[np.ndarray], original_image: np.ndarray, titles: List[str], grid_size: Tuple[int, int], cmap: str = "gray") -> None:
    nrows, ncols = grid_size

    if len(images) > nrows * ncols:
        raise ValueError("The number of images exceeds the grid size.")

    if nrows == 1 and ncols == 1:
        fig, ax = plt.subplots()
        blended_image = blend_image_and_mask(original_image, images[0])
        ax.imshow(blended_image)
        if titles is not None:
            ax.set_title(titles[0])
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(mask_dir, 'image_with_mask.png'))  # Save the blended image
        print(f"SAM mask saved at {mask_dir}")
    else:
        raise ValueError("The number of images exceeds 1.")

def blend_image_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend the original image and mask.
    :param image: Original image (RGB)
    :param mask: Binary mask
    :return: Blended image
    """
    # Convert original image to RGBA format
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    mask_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
    mask_rgba[:, :, 2] = mask * 100  # Dark blue channel for mask
    mask_rgba[:, :, 3] = mask * 128  # Alpha channel for mask
   
    # Blend images
    blended = cv2.addWeighted(image_rgba, 1.0, mask_rgba, 0.7, 0)

    # Convert blended image back to RGB format
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_RGBA2RGB)

    return blended_rgb


