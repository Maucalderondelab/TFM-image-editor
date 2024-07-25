import os
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scripts.helper_functions import GroundingDINO, SAMSegmenter, StableDiffusionInpainter, load_grounding_dino_model, process_image_with_grounding_dino, plot_images_grid, blend_image_and_mask

# Run setup script
from scripts.setup import GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, SAM_CHECKPOINT_PATH
from typing import List, Tuple
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants

STABLE_DIFFUSION_MODEL_PATH = "stabilityai/stable-diffusion-2-inpainting"

def edit_image_func(image_path: str, prompt: str) -> Image.Image:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_OUTPUT_PATH = '/root/app/outputs/mask'
    EDITED_IMAGE_OUTPUT_PATH = '/root/app/outputs/edited_images'
    # Load GroundingDINO model
    grounding_dino_model: 'Model' = load_grounding_dino_model(
        config_path=GROUNDING_DINO_CONFIG_PATH,
        checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )
    
    grounding_dino: GroundingDINO = GroundingDINO(grounding_dino_model)
    
    # Load image
    image_bgr: np.ndarray = cv2.imread(image_path)
    image_rgb: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Process image with GroundingDINO
    prompt = prompt.strip().lower()
    text_prompt, paint_prompt = prompt.split("||")
    detections: 'Detections' = process_image_with_grounding_dino(grounding_dino, image_bgr, text_prompt)
    
    # Instantiate SAMSegmenter
    sam_segmenter: SAMSegmenter = SAMSegmenter(checkpoint_path=SAM_CHECKPOINT_PATH, model_type='vit_h', device=device)
    
    # Segment using SAM
    detections.mask = sam_segmenter.segment(image=image_bgr, xyxy=detections.xyxy)
    # Prepare for plotting
    titles: List[str] = [detections.data[i]['class_id'] for i in range(len(detections.mask))]
    
    # Initialize StableDiffusionInpainter
    sd_inpainter: StableDiffusionInpainter = StableDiffusionInpainter(STABLE_DIFFUSION_MODEL_PATH, device=device)
    
    # Plot images with masks overlaid
    plot_images_grid(
        mask_dir = MASK_OUTPUT_PATH,
        images=[detections.mask[0]],
        original_image=image_rgb,
        titles= titles,
        grid_size=(1, 1)
    )
    
    if text_prompt == 'background':
        # Prepare inputs for inpainting
        mask = detections.mask[0]
        # Generate outpainted image 
        image_source_pil = Image.fromarray(image_rgb)
        image_mask_pil = Image.fromarray(mask)

        p_negative_prompt = "low resolution, ugly"
        p_SEED = 42
        generated_image = sd_inpainter.generate_image(
            image=image_source_pil,
            mask=image_mask_pil,
            prompt=paint_prompt,
            negative_prompt=p_negative_prompt,
            seed=p_SEED,
        )
        # Return generated image
        return generated_image

        # Save generated image
        # edited_image_path = os.path.join(EDITED_IMAGE_OUTPUT_PATH, 'outpainted_image.png')
        # generated_image.save(edited_image_path)
        # print(f"Outpainted image saved at {edited_image_path}")
   
    else: 
        
        # Prepare inputs for inpainting
        mask = detections.mask[0]

        image_source_pil = Image.fromarray(image_rgb)
        image_mask_pil = Image.fromarray(mask)

        p_negative_prompt = "low resolution, ugly"
        p_SEED = 42
        # Generate inpainted image
        generated_image = sd_inpainter.generate_image(
            image=image_source_pil,
            mask=image_mask_pil,
            prompt=paint_prompt,
            negative_prompt=p_negative_prompt,
            seed=p_SEED,
        )
        # Return generated image
        return generated_image 
        # Save generated image
        # edited_image_path = os.path.join(EDITED_IMAGE_OUTPUT_PATH, 'outpainted_image.png')
        # generated_image.save(edited_image_path)
        # print(f"Outpainted image saved at {edited_image_path}")
