import os
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scripts.helper_functions import GroundingDINO, SAMSegmenter, StableDiffusionInpainter, load_grounding_dino_model, process_image_with_grounding_dino, plot_images_grid, blend_image_and_mask

# Run setup script
from scripts.setup import GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, SAM_CHECKPOINT_PATH

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # IMAGE_PATH = '/workspace/images/fox.jpg'
    IMAGE_PATH = '/workspace/test_images/1.jpg'
    MASK_OUTPUT_PATH = '/workspace/outputs/masks'
    EDITED_IMAGE_OUTPUT_PATH = '/workspace/outputs/edited_images'

    # Enter the text PROMPT
    # TEXT_PROMPT = "fox".strip().lower()
    # PAIN_PROMPT = "A brown bulldog".strip().lower()
    TEXT_PROMPT = "background".strip().lower()
    PAIN_PROMPT = "Midle of the road with rain in the background".strip().lower()

    # Load GroundingDINO model
    original_cwd = os.getcwd()
    os.chdir('GroundingDINO')
    from GroundingDINO.groundingdino.util.inference import (
        load_model,
        load_image,
        predict,
        annotate,
        Model,
    )
    os.chdir(original_cwd)
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )
    
    # Load GroundingDINO model
    grounding_dino = GroundingDINO(grounding_dino_model)
    # Print initial CUDA information
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("CUDA not available. Using CPU.")

    # Load image
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Process image with GroundingDINO
    detections = process_image_with_grounding_dino(grounding_dino, image_bgr, TEXT_PROMPT)

    # Instantiate SAMSegmenter
    sam_segmenter = SAMSegmenter(checkpoint_path=SAM_CHECKPOINT_PATH, model_type='vit_h', device=device)

    # Segment using SAM
    detections.mask = sam_segmenter.segment(image=image_bgr, xyxy=detections.xyxy)
    
    # Prepare flor plotting
    titles = []
    for i in range(len(detections.mask)):
        titles.append(detections.data[i]['class_id'])

    # Initialize StableDiffusionInpainter
        STABLE_DIFFUSION_MODEL_PATH = "stabilityai/stable-diffusion-2-inpainting"
        sd_inpainter = StableDiffusionInpainter(STABLE_DIFFUSION_MODEL_PATH, device=device)
        
    # Plot images with masks overlaid
    plot_images_grid(
        mask_dir = MASK_OUTPUT_PATH,
        images=[detections.mask[0]],
        original_image=image_rgb,
        titles= titles,
        grid_size=(1, 1)
    )
    
    if TEXT_PROMPT == 'background':
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
            prompt=PAIN_PROMPT,
            negative_prompt=p_negative_prompt,
            seed=p_SEED,
        )
        # Save generated image
        edited_image_path = os.path.join(EDITED_IMAGE_OUTPUT_PATH, 'outpainted_image.png')
        generated_image.save(edited_image_path)
        print(f"Outpainted image saved at {edited_image_path}")
   
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
            prompt=PAIN_PROMPT,
            negative_prompt=p_negative_prompt,
            seed=p_SEED,
        )

        # Save generated image
        edited_image_path = os.path.join(EDITED_IMAGE_OUTPUT_PATH, 'outpainted_image.png')
        generated_image.save(edited_image_path)
        print(f"Outpainted image saved at {edited_image_path}")