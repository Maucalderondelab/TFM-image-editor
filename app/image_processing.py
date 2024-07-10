from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_black_and_white(input_image, output_path=None):
    try:
        if isinstance(input_image, str):
            # If input is a file path
            logger.info(f"Opening image from path: {input_image}")
            with Image.open(input_image) as img:
                bw_image = img.convert('L')
        else:
            # If input is already a PIL Image object
            logger.info("Converting PIL Image object to black and white")
            bw_image = input_image.convert('L')
        
        if output_path:
            logger.info(f"Saving black and white image to: {output_path}")
            bw_image.save(output_path)
        
        return bw_image
    except Exception as e:
        logger.error(f"Error in convert_to_black_and_white: {str(e)}")
        raise