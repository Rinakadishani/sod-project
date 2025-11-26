import time
import torch
import numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms.functional as TF

from src.model import CNN_SOD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN_SOD().to(DEVICE)
model.load_state_dict(torch.load("final_model.pth", map_location=DEVICE))
model.eval()

def predict(image):
    start = time.time()

    img = Image.fromarray(image).convert("RGB")
    img_resized = img.resize((128, 128))
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    inference_time = time.time() - start

    pred_resized = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size)

    overlay = np.array(img).copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255
    mask = np.array(pred_resized) > 128
    overlay[mask] = overlay[mask] * 0.5 + red_mask[mask] * 0.5

    return (
        img,                
        pred_resized,       
        overlay,          
        f"{inference_time:.4f} seconds"
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an image"),
    outputs=[
        gr.Image(label="Input Image"),
        gr.Image(label="Predicted Mask"),
        gr.Image(label="Overlay (Mask + Image)"),
        gr.Textbox(label="Inference Time")
    ],
    title="Salient Object Detection Demo"
)

if __name__ == "__main__":
    demo.launch()
