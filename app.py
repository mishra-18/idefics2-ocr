import gradio as gr
from transformers import BitsAndBytesConfig, AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
import torch
from PIL import Image
from idefics2 import Idefics2FT

id2ft = Idefics2FT()
model_id = "smishr-18/Idefics2-OCR"

model = id2ft._load_model(model_id=model_id)
processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(image, text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain"},
                {"type": "image"},
                {"type": "text", "text": text}
            ]
        }
    ]

    image = Image.fromarray(image)

    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate texts
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]


gr_interface = gr.Interface(
    fn=inference,
    inputs=["image", "text"],
    outputs="text",
    title="Idefics-OCR Inference",
    description="Upload an image and enter a prompt to get model inference."
)

gr_interface.launch(debug=True)