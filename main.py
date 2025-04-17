from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import uuid
import shutil
import torch
import requests
import numpy as np
from PIL import Image

from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/rmbg_from_file")
async def remove_background_from_file(file: UploadFile = File(...)):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        if not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Uploaded file is not an image"}, status_code=400)

        with open(temp_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        remove_img_bg_local(temp_input_path, temp_output_path)

        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/rmbg_from_url")
def remove_background_from_url(image_url: str = Query(...)):
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"{OUTPUT_DIR}/{temp_id}_input.png"
        temp_output_path = f"{OUTPUT_DIR}/{temp_id}_no_bg.png"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, stream=True, headers=headers)
        if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
            return JSONResponse(content={"error": "Invalid image URL"}, status_code=400)

        with open(temp_input_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        remove_img_bg_local(temp_input_path, temp_output_path)

        output_url = f"/output/{os.path.basename(temp_output_path)}"
        return {"success": True, "url": output_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def remove_img_bg_local(input_path: str, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BriaRMBG().to(device)
    net.eval()

    with Image.open(input_path) as pil_image:
        pil_image = pil_image.convert("RGB")
        orig_im = np.array(pil_image)

    model_input_size = [1024, 1024]
    image_tensor = preprocess_image(orig_im, model_input_size).to(device)
    result = net(image_tensor)  # simulate output

    result_image = postprocess_image(result[0][0], orig_im.shape[0:2])
    mask = Image.fromarray(result_image).convert("L")
    orig_image = Image.open(input_path).convert("RGBA")

    no_bg_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=mask)
    no_bg_image.save(output_path)

    return output_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)