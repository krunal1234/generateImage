from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os
import uuid
import shutil
import torch
import requests
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download

from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rmbg_from_file")
async def remove_background_from_file(file: UploadFile = File(...)):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        # Validate uploaded file content type
        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        # Save uploaded image to disk
        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Remove background
        remove_img_bg_local(temp_input_path, temp_output_path)

        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}

    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Output directory setup
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


def remove_img_bg_local(input_path: str, output_path: str):
    # Initialize model
    model = BriaRMBG()

    # Read image
    with Image.open(input_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        orig_im = np.array(pil_image)

    if orig_im is None or orig_im.size == 0:
        raise ValueError("Failed to load image")

    # Preprocess
    model_input_size = [1024, 1024]
    input_tensor = preprocess_image(orig_im, model_input_size)

    # Run model
    with torch.no_grad():
        result = model(input_tensor)

    # Postprocess
    result_image = postprocess_image(result[0][0], orig_im.shape[:2])
    mask = Image.fromarray(result_image).convert("L")
    orig_image = Image.open(input_path).convert("RGBA")

    no_bg_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=mask)
    no_bg_image.save(output_path)

    return output_path

@app.get("/rmbg_from_url")
def remove_background_from_url(image_url: str = Query(..., description="URL of the image")):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        # Download image
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, stream=True, headers=headers)
        if response.status_code != 200:
            return JSONResponse(content={"error": "Failed to download image"}, status_code=400)

        if "image" not in response.headers.get("Content-Type", ""):
            return JSONResponse(content={"error": "URL did not return an image"}, status_code=400)

        # Save image to file
        with open(temp_input_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        # Remove background
        remove_img_bg_local(temp_input_path, temp_output_path)

        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}

    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Old endpoint using local file
@app.get("/rmbg")
def remove_background(name: str, type="jpg", outSuffix="_no_bg.png"):
    file_path = f"./resource/{name}.{type}"
    output_path = f"./resource/{name}{outSuffix}"
    remove_img_bg_local(file_path, output_path)
    return {"path": output_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
