from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# Get the current directory and form the path to the style directory
current_dir = os.path.dirname(os.path.abspath(__file__))
style_dir = os.path.join(current_dir, "style")

# Serve static files from the 'style' directory
app.mount("/style", StaticFiles(directory=style_dir), name="style")

class EditRequest(BaseModel):
    imageName: str
    prompt: str
@app.get("/")
async def read_index():
    file_path = os.path.join(style_dir, 'index.html')
    print(f"Serving file from: {file_path}")  # Log the file path
    return FileResponse(file_path)

@app.post("/edit-image")
async def edit_image(request: EditRequest):
    # Dummy edited images
    edited_images = [
        f"Edited {request.imageName} 1",
        f"Edited {request.imageName} 2",
        f"Edited {request.imageName} 3",
        f"Edited {request.imageName} 4"
    ]
    return edited_images
