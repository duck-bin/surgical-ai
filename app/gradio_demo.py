"""Gradio demo for the CVS assessment pipeline.

Displays, for an uploaded or sample frame: the original image, the segmentation
overlay, the 3 CVS criteria with confidence bars, and the composite 0-3 CVS
score.

Usage (needs trained checkpoints):
    python -m app.gradio_demo
"""
from __future__ import annotations

DISCLAIMER = "Research prototype, not for clinical use."


def build_demo(pipeline):
    """Construct the Gradio Blocks app around a CVSPipeline-like callable.

    ``pipeline`` is any callable mapping an RGB frame to
    ``(mask, criteria_probs, cvs_score)``.
    """
    import gradio as gr

    from src.data.endoscapes import CVS_CRITERIA
    from src.utils.viz import overlay_mask

    def assess(image):
        if image is None:
            return None, {}, ""
        mask, criteria_probs, score = pipeline(image)
        overlay = overlay_mask(image, mask)
        confidences = {name: float(prob)
                       for name, prob in zip(CVS_CRITERIA, criteria_probs)}
        return overlay, confidences, f"Composite CVS score: {score} / 3"

    with gr.Blocks(title="Surgical CVS Assessment") as demo:
        gr.Markdown(f"# Surgical CVS Assessment\n\n_{DISCLAIMER}_")
        with gr.Row():
            frame = gr.Image(type="numpy", label="Laparoscopic frame")
            overlay = gr.Image(type="numpy", label="Segmentation overlay")
        criteria = gr.Label(label="CVS criteria (achieved probability)")
        score = gr.Textbox(label="CVS score", interactive=False)
        gr.Button("Assess CVS").click(
            assess, inputs=frame, outputs=[overlay, criteria, score])
    return demo


if __name__ == "__main__":
    from src.inference.pipeline import CVSPipeline

    pipeline = CVSPipeline.from_checkpoints(
        seg_model_config="configs/model/sam2_lora.yaml",
        seg_checkpoint="outputs/sam2_lora/best.ckpt",
        cvs_checkpoint="outputs/cvs_classifier/best.ckpt",
    )
    build_demo(pipeline).launch()
