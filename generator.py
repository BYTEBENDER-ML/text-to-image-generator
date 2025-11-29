import os, json, time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

def add_watermark(img, txt="AI-generated"):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.text((w-120, h-30), txt, fill=(255,255,255))
    return img

def save_metadata(img, path, prompt, neg, params):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "prompt": prompt,
        "negative_prompt": neg,
        "timestamp": ts,
        "params": params
    }
    json.dump(meta, open(str(path)+".json","w"), indent=2)
    img.save(str(path)+".png")

def generate_images(prompt, negative, num, gs, h, w, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(model_id, device)
    images = []

    for i in range(num):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=gs,
            height=h,
            width=w
        )
        img = result.images[0].convert("RGB")
        img = add_watermark(img)

        params = {"guidance_scale": gs, "height": h, "width": w}
        folder = Path("output_images")
        folder.mkdir(exist_ok=True)

        filename = folder / f"img_{int(time.time())}_{i}"
        save_metadata(img, filename, prompt, negative, params)

        images.append(img)

    return images
