import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import List
from PIL import Image
from skimage import measure
import cv2

import torch
import transformers
import accelerate

import os
import sys
import torch
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

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
    def __init__(self, model):
        self.model = model
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
    
    def predict_with_captions(self, image, text_prompt):
        # Assuming self.model.predict_with_caption returns detections and phrases
        detections, phrases = self.model.predict_with_caption(
            image=image,
            caption=text_prompt,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )
        return detections, phrases
    
    def visualize_detections(self, image, detections):
        # Visualize detections as before
        image_with_boxes = image.copy()
        for i, box in enumerate(detections.xyxy):
            x_min, y_min, x_max, y_max = map(int, box)
            class_id = detections.data[i]['class_id']
            cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, class_id, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boxes_rgb)
        plt.title('Image with Detections')
        plt.axis('off')
        plt.show()
###########################################################################################################
###########################################################################################################
# SAM
###########################################################################################################
###########################################################################################################

class SAMSegmenter:
    def __init__(self, checkpoint_path, model_type='vit_h', device='cuda'):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        
        # Instantiate SAM model
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        
        # Instantiate SAM predictor and mask generator
        self.sam_predictor = SamPredictor(self.sam_model)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
    
    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def make_sam_mask(self, boolean_mask):
        binary_mask = boolean_mask.astype(int)
        contours = measure.find_contours(binary_mask, 0.5)
        mask_points = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            mask_points.append(segmentation)
        return mask_points
    
    def make_annotations(self, detections):
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
    def __init__(self, pretrained_model_path, torch_dtype=torch.float16, device='cuda'):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch_dtype,
        ).to(device)
        
    def generate_image(self, image, mask, prompt, negative_prompt, seed):
        # Resize for inpainting
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
def load_grounding_dino_model(config_path, checkpoint_path):
    grounding_dino_model = Model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
    )
    return grounding_dino_model

def process_image_with_grounding_dino(grounding_dino, image, text_prompt):
    detections, phrases = grounding_dino.predict_with_captions(image, text_prompt)
    for i, phrase in enumerate(phrases):
        detections.data[i] = {'class_id': phrase}
    return detections

def plot_images_grid(mask_dir, images, original_image, titles, grid_size, cmap="gray"):
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

def blend_image_and_mask(image, mask):
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


