import os, json, time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont


# Load pipeline (optimized)
def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


# Pillow 10+ compatible watermark
def add_watermark(img, txt="AI-generated"):
    draw = ImageDraw.Draw(img)

    # Default Pillow font
    font = ImageFont.load_default()

    # text size using textbbox (Pillow 10+)
    bbox = draw.textbbox((0, 0), txt, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    w, h = img.size

    # position: bottom-right with padding
    x = w - tw - 20
    y = h - th - 20

    # optional shadow for readability
    draw.text((x + 1, y + 1), txt, fill=(0, 0, 0), font=font)
    draw.text((x, y), txt, fill=(255, 255, 255), font=font)

    return img


# Save metadata + PNG
def save_metadata(img, path, prompt, neg, params):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    meta = {
        "prompt": prompt,
        "negative_prompt": neg,
        "timestamp": ts,
        "params": params
    }

    # JSON file
    json.dump(meta, open(str(path) + ".json", "w"), indent=2)

    # PNG file
    img.save(str(path) + ".png")


# Main image generation function
def generate_images(prompt, negative, num, gs, h, w, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load stable diffusion pipeline
    pipe = load_pipeline(model_id, device)

    images = []

    # Ensure output folder exists
    folder = Path("output_images")
    folder.mkdir(exist_ok=True)

    for i in range(num):

        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=gs,
            height=h,
            width=w
        )

        img = result.images[0].convert("RGB")

        # Add watermark
        img = add_watermark(img)

        # Save
        params = {"guidance_scale": gs, "height": h, "width": w}
        filename = folder / f"img_{int(time.time())}_{i}"
        save_metadata(img, filename, prompt, negative, params)

        images.append(img)

    return images
