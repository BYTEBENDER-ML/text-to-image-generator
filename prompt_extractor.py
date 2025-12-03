# prompt_extractor.py
from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)

# lightweight summarizer (low RAM, fast)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def article_to_prompts(text, n=6):
    # summarize first 2500 chars only (safe for low RAM)
    chunk = text[:2500]

    summary = summarizer(
        chunk,
        max_length=120,
        min_length=40,
        truncation=True
    )[0]["summary_text"]

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(summary)

    prompts = []

    for s in sentences[:n]:
        s_clean = s.strip()

        prompt = (
            f"{s_clean}, photorealistic, ultra-detailed, 8k, 50mm lens, "
            f"cinematic lighting, crisp textures, hyper-real clarity"
        )

        prompts.append(prompt)

    return prompts
