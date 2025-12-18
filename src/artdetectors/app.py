"""
Gradio UI for the Art Detectors pipeline.

This wraps the updated `ImageAnalysisPipeline` from `src/artdetectors/pipeline.py`
to provide:
- Fast analysis (CLIP styles + BLIP caption + SuSy source detection)
- Optional restyling via Stable Diffusion img2img

Run with:
    uvicorn app:demo  # or simply `python app.py`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import gradio as gr
from PIL import Image

# Avoid loading TensorFlow when pulling HF models
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Ensure local src/ is on PYTHONPATH when running without installation
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.append(str(SRC_PATH))

from artdetectors import ImageAnalysisPipeline  # noqa: E402

# Cache pipelines so models are loaded only once per configuration
_PIPELINES: Dict[str, ImageAnalysisPipeline] = {}


def _get_pipeline(use_transfer_learning: bool, enable_restyler: bool) -> ImageAnalysisPipeline:
    """
    Lazily create a pipeline for the requested configuration.

    We keep separate instances for:
      - Detectors only (faster; restyler disabled)
      - Restyling (loads SD img2img; heavier)
    """
    key = f"{'tl3' if use_transfer_learning else 'orig6'}_{'restyle' if enable_restyler else 'detect'}"
    if key not in _PIPELINES:
        _PIPELINES[key] = ImageAnalysisPipeline(
            use_transfer_learning=use_transfer_learning,
            enable_restyler=enable_restyler,
        )
    return _PIPELINES[key]


def _format_styles(styles: Union[List[Tuple[str, float]], Sequence[Dict[str, float]]]) -> str:
    if not styles:
        return "No styles predicted."

    lines: List[str] = []
    first = styles[0]
    if isinstance(first, dict):
        for item in styles:  # type: ignore[arg-type]
            name = item.get("style", "unknown")
            score = float(item.get("score", 0.0))
            lines.append(f"{name}: {score:.4f}")
    else:
        for name, score in styles:  # type: ignore[misc]
            lines.append(f"{name}: {float(score):.4f}")
    return "\n".join(lines)


def _format_susy(susy: Dict[str, object]) -> str:
    if not susy:
        return "No SuSy prediction."

    probs = susy.get("probs", {}) or {}
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    lines = [
        f"Prediction: {susy.get('pred_class')} ({susy.get('model_type')})",
        f"Confidence: {float(susy.get('confidence', 0.0)):.4f}",
        "Probabilities:",
    ]
    lines.extend([f"  - {k}: {float(v):.4f}" for k, v in sorted_probs])
    return "\n".join(lines)


def _safe_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def analyze_image(
    image: Image.Image,
    use_transfer_learning: bool,
    topk_styles: int,
    caption_prompt: str,
):
    if image is None:
        return "No image uploaded.", "", ""

    pipe = _get_pipeline(use_transfer_learning=use_transfer_learning, enable_restyler=False)
    result = pipe.analyze(
        image=image,
        topk_styles=int(topk_styles),
        caption_prompt=caption_prompt or None,
    )

    styles_formatted = _format_styles(result.get("styles", []))
    susy_formatted = _format_susy(result.get("susy", {}))
    caption = result.get("caption", "")

    return caption, styles_formatted, susy_formatted


def restyle_image(
    image: Image.Image,
    target_style: str,
    caption_prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed,
    use_transfer_learning: bool,
):
    if image is None:
        return None, "No image uploaded."

    target_style = (target_style or "").strip()
    if not target_style:
        return None, "Please provide a target style (e.g., 'ukiyo-e', 'cubism')."

    seed_val = _safe_int(seed)

    pipe = _get_pipeline(use_transfer_learning=use_transfer_learning, enable_restyler=True)
    out = pipe.restyle_image(
        image=image,
        target_style=target_style,
        caption_prompt=caption_prompt or None,
        negative_prompt=negative_prompt or None,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        seed=seed_val,
    )

    return out["restyled_image"], out["caption_used"]


with gr.Blocks() as demo:
    gr.Markdown("# MLAngelo — Style, Caption, Source Detection")
    gr.Markdown(
        "Run CLIP + BLIP + SuSy for fast analysis. Optionally restyle an image "
        "with Stable Diffusion img2img."
    )

    with gr.Tab("Analyze"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload artwork")
            with gr.Column():
                use_tl = gr.Checkbox(
                    label="Use transfer learning (3-class SuSy)",
                    value=True,
                )
                topk = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-k styles",
                )
                caption_prompt = gr.Textbox(
                    label="Optional caption prompt",
                    placeholder="e.g., 'a watercolor painting of ...'",
                )
                analyze_btn = gr.Button("Analyze")

        caption_out = gr.Textbox(label="BLIP Caption", lines=6, max_lines=12)
        styles_out = gr.Textbox(label="Top Predicted Styles", lines=8, max_lines=14)
        susy_out = gr.Textbox(label="Source (SuSy Probabilities)", lines=10, max_lines=16)

        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_in, use_tl, topk, caption_prompt],
            outputs=[caption_out, styles_out, susy_out],
        )

    with gr.Tab("Restyle"):
        with gr.Row():
            restyle_img_in = gr.Image(type="pil", label="Upload artwork to restyle")
            restyle_controls = gr.Column()

        with restyle_controls:
            restyle_use_tl = gr.Checkbox(
                label="Use transfer learning (for captioning)",
                value=True,
            )
            target_style = gr.Textbox(
                label="Target style",
                placeholder="e.g., ukiyo-e, cubism, impressionism",
            )
            caption_prompt_restyle = gr.Textbox(
                label="Optional caption prompt",
                placeholder="Custom prompt to guide caption (restyle uses caption + target style)",
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="blurry, distorted, noisy",
            )
            strength = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Strength (0 = preserve image, 1 = change more)",
            )
            guidance_scale = gr.Slider(
                minimum=2.0,
                maximum=10.0,
                value=5.0,
                step=0.5,
                label="Guidance scale",
            )
            num_steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=1,
                label="Inference steps",
            )
            seed = gr.Number(label="Seed (optional, int)", precision=0)
            restyle_btn = gr.Button("Restyle image")

        restyled_image_out = gr.Image(type="pil", label="Restyled image")
        caption_used_out = gr.Textbox(label="Caption used for restyle", lines=4, max_lines=8)

        restyle_btn.click(
            fn=restyle_image,
            inputs=[
                restyle_img_in,
                target_style,
                caption_prompt_restyle,
                negative_prompt,
                strength,
                guidance_scale,
                num_steps,
                seed,
                restyle_use_tl,
            ],
            outputs=[restyled_image_out, caption_used_out],
        )

# Enable share link if running on Colab/etc. Set share=True to enable public URL.
if __name__ == "__main__":
    demo.launch(share=True)

