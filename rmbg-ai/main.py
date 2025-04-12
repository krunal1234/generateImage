import os
import uuid
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from skimage import io

app = FastAPI()

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)


def remove_img_bg_local(input_path: str, output_path: str):
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
    net = BriaRMBG()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    model_input_size = [1024, 1024]
    orig_im = io.imread(input_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    result = net(image)
    result_image = postprocess_image(result[0][0], orig_im_size)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(input_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save(output_path)


@app.get("/generate_image_local")
def generate_image_local(prompt: str, remove_bg: bool = False):
    try:
        temp_id = str(uuid.uuid4())
        image_path = f"{OUTPUT_DIR}/{temp_id}_gen.png"
        final_path = f"{OUTPUT_DIR}/{temp_id}_final.png"

        # Generate image
        image: Image.Image = pipe(prompt).images[0]
        image.save(image_path)

        if remove_bg:
            remove_img_bg_local(image_path, final_path)
        else:
            final_path = image_path

        return {
            "success": True,
            "url": f"/output/{os.path.basename(final_path)}"
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
