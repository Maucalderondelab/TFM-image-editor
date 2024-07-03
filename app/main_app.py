from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


import os
from image_processing import convert_to_black_and_white  # Import the new function

app = FastAPI()

# Directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
style_dir = os.path.join(current_dir, "style")
image_dir = "/home/mauricio/Documents/Projects/TFM/test_images"
edited_images_dir = "/home/mauricio/Documents/Projects/TFM/edited_images"
os.makedirs(edited_images_dir, exist_ok=True)  # Ensure directory exists


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Serve static files
app.mount("/test_images", StaticFiles(directory=image_dir), name="test_images")

@app.get("/")
async def read_index():
    file_path = os.path.join(style_dir, 'index.html')
    return FileResponse(file_path)

@app.get("/images")
async def get_images():
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    print(f"Found images: {images}")  # This will print to your terminal
    return images

@app.post("/edit-image")
async def edit_image(image_name: str = Form(...)):
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    edited_image_path = os.path.join(edited_images_dir, image_name)
    convert_to_black_and_white(image_path, edited_image_path)
    return RedirectResponse(url=f"/edited_images/{image_name}")

@app.get("/edited_images/{filename}")
async def get_edited_image(filename: str):
    file_path = os.path.join(edited_images_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
