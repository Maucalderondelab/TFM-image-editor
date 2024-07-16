from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.image_processing import edit_image_func
import os
import base64
import io
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of the current file
current_dir: str = os.path.dirname(os.path.abspath(__file__))

# Mount static directories
style_dir: str = os.path.join(current_dir, "style")
app.mount("/style", StaticFiles(directory=style_dir), name="style")

test_images_dir: str = "/home/mauricio/Documents/Projects/TFM/test_images"
app.mount("/test_images", StaticFiles(directory=test_images_dir), name="test_images")

edited_images_dir: str = "/home/mauricio/Documents/Projects/TFM/edited_images"
app.mount("/edited_images", StaticFiles(directory=edited_images_dir), name="edited_images")

# Ensure edited_images directory exists
os.makedirs(edited_images_dir, exist_ok=True)

@app.get("/")
async def read_index() -> FileResponse:
    """Serve the index.html file."""
    return FileResponse(os.path.join(style_dir, 'index.html'))

@app.get("/images")
async def get_images() -> List[str]:
    """Fetch and return a list of image filenames."""
    logger.info("Fetching list of images")
    images: List[str] = [f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))]
    logger.info(f"Found {len(images)} images")
    return images

@app.post("/preview-edit")
async def preview_edit(image_name: str = Form(...), prompt: str = Form(...)) -> JSONResponse:
    """
    Preview an edited image based on the provided image name and prompt.

    Args:
        image_name (str): The name of the image file to edit.
        prompt (str): The text prompt to add to the image.

    Returns:
        JSONResponse: A JSON response containing the base64 encoded edited image data.

    Raises:
        HTTPException: If the image is not found or if there's an error during processing.
    """
    try:
        logger.info(f"Received preview edit request for image: {image_name} with prompt: {prompt}")
        image_path: str = os.path.join(test_images_dir, image_name)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        logger.info(f"Applying custom edit to image: {image_path}")
        edited_image = edit_image_func(image_path, prompt)

        buffered: io.BytesIO = io.BytesIO()
        edited_image.save(buffered, format="PNG")
        img_str: str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("Successfully edited and encoded image")
        return JSONResponse(content={"edited_image_data": f"data:image/png;base64,{img_str}"})
    except Exception as e:
        logger.error(f"Error in preview_edit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-edit")
async def save_edit(image_name: str = Form(...), edited_image_data: str = Form(...)) -> JSONResponse:
    """
    Save an edited image.

    Args:
        image_name (str): The name of the original image file.
        edited_image_data (str): The base64 encoded edited image data.

    Returns:
        JSONResponse: A JSON response containing the URL of the saved image.

    Raises:
        HTTPException: If there's an error during the saving process.
    """
    try:
        logger.info(f"Received save edit request for image: {image_name}")
        # Remove the data URL prefix
        image_data: str = edited_image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes: bytes = base64.b64decode(image_data)
        
        # Generate a filename for the saved image
        timestamp: int = int(time.time())
        saved_image_name: str = f"edited_{timestamp}_{image_name}"
        saved_image_path: str = os.path.join(edited_images_dir, saved_image_name)
        
        # Save the image
        logger.info(f"Saving edited image to: {saved_image_path}")
        with open(saved_image_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.info("Successfully saved edited image")
        return JSONResponse(content={"saved_image_url": f"/edited_images/{saved_image_name}"})
    except Exception as e:
        logger.error(f"Error in save_edit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/edited_images/{filename}")
async def get_edited_image(filename: str) -> FileResponse:
    """
    Serve an edited image file.

    Args:
        filename (str): The name of the edited image file.

    Returns:
        FileResponse: The requested image file.

    Raises:
        HTTPException: If the requested file is not found.
    """
    file_path: str = os.path.join(edited_images_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    logger.error(f"Edited image not found: {file_path}")
    raise HTTPException(status_code=404, detail="File not found")