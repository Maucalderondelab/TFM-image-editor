from PIL import Image, ImageDraw, ImageFont
import logging
import os
from typing import Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def edit_image_func(input_image: Union[str, Image.Image], prompt: str, output_path: Optional[str] = None) -> Image.Image:
    """
    Edit an image by converting it to black and white and adding text.

    Args:
        input_image (Union[str, Image.Image]): Either a file path to an image or a PIL Image object.
        prompt (str): The text to be added to the image.
        output_path (Optional[str]): If provided, the path where the edited image will be saved.

    Returns:
        Image.Image: The edited PIL Image object.

    Raises:
        FileNotFoundError: If the input image file or font file is not found.
        IOError: If there's an error opening or processing the image.
        Exception: For any other unexpected errors during processing.
    """
    try:
        if isinstance(input_image, str):
            # If input is a file path
            logger.info(f"Opening image from path: {input_image}")
            with Image.open(input_image) as img:
                edited_image = img.convert('L')  # Convert to black and white
        else:
            # If input is already a PIL Image object
            logger.info("Converting PIL Image object to black and white")
            edited_image = input_image.convert('L')

        # Add text to the image
        draw = ImageDraw.Draw(edited_image)
        
        # Determine font size based on image size
        font_size = int(min(edited_image.width, edited_image.height) / 10)
        
        # Load a font (you may need to specify a different font file path)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            logger.warning(f"Font not found at {font_path}. Using default font.")
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, font_size)

        # Calculate text position (center of the image)
        left, top, right, bottom = draw.textbbox((0, 0), prompt, font=font)
        text_width = right - left
        text_height = bottom - top
        position = ((edited_image.width - text_width) / 2, (edited_image.height - text_height) / 2)

        # Draw the text
        draw.text(position, prompt, fill="white", font=font)

        if output_path:
            logger.info(f"Saving edited image to: {output_path}")
            edited_image.save(output_path)

        return edited_image
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"IO error processing the image: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in edit_image_func: {str(e)}")
        raise