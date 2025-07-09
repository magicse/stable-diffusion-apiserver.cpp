import gradio as gr
import requests
from pathlib import Path
import json

# A client to interact with a Stable Diffusion image generation server
class StableDiffusionClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url.rstrip('/')

    # Check the status of the server
    def check_status(self):
        try:
            response = requests.get(f"{self.server_url}/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    # Load a model by specifying the path
    def load_model(self, model_path):
        payload = {"model_path": model_path}
        try:
            response = requests.post(
                f"{self.server_url}/load_model",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200 and response.json().get("success"):
                return f"Model loaded: {model_path}"
            return f"Error: {response.text}"
        except Exception as e:
            return f"Exception: {e}"

    # Send a generation request to the server
    def generate_image(self, prompt, negative_prompt, width, height, steps, cfg_scale, seed, batch_count):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "batch_count": batch_count
        }

        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                return f"HTTP Error: {response.status_code}\n{response.text}", None

            result = response.json()
            if not result.get("success"):
                return f"Generation Error: {result}", None

            filenames = result.get("filenames", [])
            images = []
            for filename in filenames:
                image_url = f"{self.server_url}/image/{filename}"
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_path = f"temp_{filename}"
                    with open(image_path, "wb") as f:
                        f.write(image_response.content)
                    images.append(image_path)
                else:
                    return f"Image download error: {filename}", None

            return f"Successfully generated {len(images)} images", images

        except Exception as e:
            return f"Request Error: {e}", None

client = StableDiffusionClient()

# Load styles from styles.json
with open("styles.json", "r") as file:
    styles = json.load(file)

# Create a dictionary for quick style lookup
style_options = {style['name']: style for style in styles}

# Function to generate image from UI
def ui_generate(prompt, negative_prompt, width, height, steps, cfg_scale, seed, batch_count, selected_style):
    # If a style is selected, combine it with the input
    if selected_style in style_options:
        style = style_options[selected_style]
        prompt = style['prompt'].format(prompt=prompt)
        # Combine negative prompts: style first, then user
        if style['negative_prompt']:
            negative_prompt = f"{style['negative_prompt']}, {negative_prompt}".strip(", ")

    status, images = client.generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, batch_count)
    return status, images

# Function to load a model from UI
def ui_load_model(model_path):
    return client.load_model(model_path)

# Function to check server status from UI
def ui_check_status():
    return client.check_status()

# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Client Interface")

    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="mutation, deformed, disfigured, extra limbs, blurry, bad anatomy, text, low quality, "
                        "deformed iris, duplicate, morbid, mutilated, disfigured, poorly drawn hand, poorly drawn face, bad proportions, gross proportions, extra limbs, cloned face, "
                        "long neck, malformed limbs, missing arm, missing leg, extra arm, extra leg, fused fingers, too many fingers, extra fingers, mutated hands, blurry, bad anatomy, "
                        "out of frame, contortionist, contorted limbs, exaggerated features, disproportionate, twisted posture, unnatural pose, disconnected, disproportionate, warped, "
                        "misshapen, out of scale, text, low quality"
                )

                # Dropdown menu for selecting a style
                style_dropdown = gr.Dropdown(label="Select Style", choices=list(style_options.keys()), value="base")

                width = gr.Slider(256, 1024, step=64, value=512, label="Width")
                height = gr.Slider(256, 1024, step=64, value=512, label="Height")

                steps = gr.Slider(1, 100, step=1, value=20, label="Steps")
                cfg_scale = gr.Slider(1, 15, value=7.0, label="CFG Scale")

                seed = gr.Number(value=-1, step=1, label="Seed (-1 = random)")
                batch_count = gr.Slider(1, 5, step=1, value=1, label="Batch Size")

                btn_gen = gr.Button("Generate")
                status_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Results", columns=1, object_fit="contain", height=512)

        btn_gen.click(
            fn=ui_generate,
            inputs=[prompt, negative_prompt, width, height, steps, cfg_scale, seed, batch_count, style_dropdown],
            outputs=[status_output, gallery]
        )

    with gr.Tab("Model Loading"):
        model_path = gr.Textbox(label="Model Path")
        load_btn = gr.Button("Load Model")
        load_status = gr.Textbox(label="Load Status", interactive=False)

        load_btn.click(fn=ui_load_model, inputs=[model_path], outputs=[load_status])

    with gr.Tab("Server Status"):
        check_btn = gr.Button("Check")
        server_status = gr.JSON()

        check_btn.click(fn=ui_check_status, outputs=[server_status])

# Launch the interface
demo.launch(server_name="0.0.0.0", server_port=7860)
