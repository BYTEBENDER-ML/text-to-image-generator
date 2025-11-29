# Local Text→Image Generator (open-source)

Overview
- Local text-to-image using Stable Diffusion (diffusers). Supports GPU (CUDA) and CPU fallback.
- Web UI: Streamlit. Optional REST API: Flask.
- Basic safety filtering + watermarking and metadata storage.

Quick start (Windows / VS Code)
1. Clone repo
2. Create and activate venv:
   python -m venv venv
   venv\Scripts\activate
3. Install PyTorch (choose CUDA or CPU from https://pytorch.org/get-started/locally)
   Example (CUDA 12.1): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   If no GPU: pip install torch torchvision torchaudio
4. Install remaining deps:
   pip install -r requirements.txt
5. Run:
   streamlit run app.py

Hardware requirements
- GPU recommended:
  - Minimum: 8 GB VRAM (SD 1.5 at 512×512 with optimizations).
  - Comfortable: 12–24 GB VRAM for larger models / SDXL.
- CPU path:
  - Any modern CPU works; expect large slowdown. Use smaller models (256–512).
  - Memory: 8+ GB recommended.

Usage examples (prompts)
- “a futuristic city at sunset, highly detailed, cinematic lighting, 4k”
- Negative prompt: “lowres, deformed, watermark, text”
- Style presets: photorealistic, oil painting, anime, cyberpunk

Files
- generator.py: model load + generation
- app.py: Streamlit UI
- utils/watermark.py: watermark helper
- utils/safety.py: banned keywords + optional NSFW classifier
- output_images/: saved images + JSON metadata

Ethical usage
- Built-in banned-keyword filter and optional NSFW check.
- All outputs watermarked with “AI-generated”.
- Do not generate illegal, violent, or copyrighted content requiring permission.

Limitations & future work
- CPU is slow; expect minutes-per-image for 512×512.
- Memory limits on small VRAM GPUs; use attention_slicing, float16.
- Future: model fine-tuning, multi-user queue, web deployment, style-transfer module.

License
- MIT (code). Check licenses for any model weights used.
