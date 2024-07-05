from PIL import Image, ImageOps

def convert_to_black_and_white(image_path: str, output_path: str):
    with Image.open(image_path) as img:
        # Convert the image to black and white
        img = ImageOps.grayscale(img)
        # Save the edited image
        img.save(output_path)
