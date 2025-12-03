# task2_runner.py
from article_reader import read_article
from prompt_extractor import article_to_prompts
from generator import generate_images
from pathlib import Path

MODEL_ID = "runwayml/stable-diffusion-v1-5"

ARTICLE_PATH = "articles/ML Internship Task Round 2.1- Google Docs.pdf"   # put your article file here

def main():
    print("Reading article...")
    text = read_article(ARTICLE_PATH)

    print("Extracting prompts...")
    prompts = article_to_prompts(text, n=6)

    print("Generated prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"{i}. {p}")

    print("\nGenerating images...\n")
    for p in prompts:
        generate_images(
            prompt=p,
            negative="lowres, blurry, deformed, watermark, text",
            num=1,
            gs=8.5,
            h=768,
            w=768,
            model_id=MODEL_ID
        )

    print("\nDone! Images saved in output_images/ folder.")


if __name__ == "__main__":
    main()
