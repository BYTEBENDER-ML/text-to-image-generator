from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import time, os

prompts = [
    "A family standing on the edge of a collapsing coastal village...",
    # add all 10 prompts here
]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)
    
os.makedirs("output_images", exist_ok=True)

for i, p in enumerate(prompts, start=1):
    img = pipe(
        prompt=p,
        negative_prompt="lowres, blurry, deformed, watermark, text",
        guidance_scale=8.5,
        height=768,
        width=768
    ).images[0]

    filename = f"image_{i}_{int(time.time())}.png"
    img.save(f"output_images/{filename}")
    print("Saved:", filename)
