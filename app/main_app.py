from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import time


import os
from app.image_processing import convert_to_black_and_white  # Import the new function

app = FastAPI()

# Get the directory of the current file (main_app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Mount the 'style' directory
style_dir = os.path.join(current_dir, "style")
app.mount("/style", StaticFiles(directory=style_dir), name="style")
print(style_dir)
edited_images_dir = "/home/mauricio/Documents/Projects/TFM/edited_images"
app.mount("/edited_images", StaticFiles(directory=edited_images_dir), name="edited_images")

# Mount the 'test_images' directory
test_images_dir = "/home/mauricio/Documents/Projects/TFM/test_images"
app.mount("/test_images", StaticFiles(directory=test_images_dir), name="test_images")

os.makedirs(edited_images_dir, exist_ok=True)  # Ensure directory exists

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/")
async def read_index():
    file_path = os.path.join(style_dir, 'index.html')
    return FileResponse(file_path)

@app.get("/images")
async def get_images():
    images = [f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))]
    return images


@app.post("/edit-image")
async def edit_image(image_name: str = Form(...)):
    print(f"Received image_name: {image_name}")
    image_path = os.path.join(test_images_dir, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Generate a new filename with a timestamp
    timestamp = int(time.time())
    edited_image_name = f"edited_{timestamp}_{image_name}"
    edited_image_path = os.path.join(edited_images_dir, edited_image_name)
    
    convert_to_black_and_white(image_path, edited_image_path)
    return JSONResponse(content={"edited_image_url": f"/edited_images/{edited_image_name}"})
@app.get("/edited_images/{filename}")
async def get_edited_image(filename: str):
    file_path = os.path.join(edited_images_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")