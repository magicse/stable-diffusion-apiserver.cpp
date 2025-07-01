# curl -X POST http://localhost:8080/load_model -H "Content-Type: application/json" -d '{"model_path": "model.safetensors"}'
#!/usr/bin/env python3
"""
Python client for Stable Diffusion HTTP API
"""

import requests
import json
import time
import argparse
from pathlib import Path

class StableDiffusionClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url.rstrip('/')
        
    def check_status(self):
        """Check server status"""
        try:
            response = requests.get(f"{self.server_url}/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
 
    def load_model(self, model_path):
        """
        Load model on the server
        
        Args:
            model_path: Path to the model file (.safetensors)
        """
        payload = {"model_path": model_path}
        
        try:
            response = requests.post(
                f"{self.server_url}/load_model",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"Model successfully loaded: {model_path}")
                    return True
                else:
                    print(f"Model loading error: {result}")
            else:
                print(f"HTTP error when loading model: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False

    def generate_image(self, prompt, negative_prompt="", width=512, height=512, 
                      steps=20, cfg_scale=7.0, seed=-1, save_path=None):
        """
        Generate an image
        
        Args:
            prompt: Text description of the image
            negative_prompt: Negative prompt
            width: Image width
            height: Image height  
            steps: Number of denoising steps
            cfg_scale: CFG scale (influence on prompt adherence)
            seed: Seed for reproducibility (-1 for random)
            save_path: Path to save the generated image
        """
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed
        }
        
        print(f"Sending generation request...")
        print(f"Prompt: {prompt}")
        
        try:
            # Send generation request
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    filename = result.get("filename")
                    print(f"Image generated: {filename}")
                    
                    # Download the image
                    if filename:
                        image_response = requests.get(f"{self.server_url}/image/{filename}")
                        if image_response.status_code == 200:
                            # Determine save path
                            if save_path:
                                output_path = Path(save_path)
                            else:
                                output_path = Path(filename)
                            
                            # Save the image
                            with open(output_path, 'wb') as f:
                                f.write(image_response.content)
                            
                            print(f"Image saved: {output_path}")
                            return str(output_path)
                        else:
                            print(f"Download error: {image_response.status_code}")
                            return None
                else:
                    print(f"Generation error: {result}")
                    return None
            else:
                print(f"HTTP error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion HTTP API Client")
    parser.add_argument("--server", default="http://localhost:8080", 
                       help="API server URL")
    parser.add_argument("--prompt", required=True, 
                       help="Text description of the image")
    parser.add_argument("--negative", default="", 
                       help="Negative prompt")
    parser.add_argument("--width", type=int, default=512, 
                       help="Image width")
    parser.add_argument("--height", type=int, default=640, 
                       help="Image height")
    parser.add_argument("--steps", type=int, default=20, 
                       help="Number of steps")
    parser.add_argument("--cfg", type=float, default=7.0, 
                       help="CFG scale")
    parser.add_argument("--seed", type=int, default=-1, 
                       help="Generation seed")
    parser.add_argument("--output", 
                       help="Path to save the image")
    parser.add_argument("--status", action="store_true", 
                       help="Check server status")
    parser.add_argument("--load_model", 
                       help="Path to the .safetensors model file to load")                       
    
    args = parser.parse_args()
    
    client = StableDiffusionClient(args.server)

    # Default model path
    default_model_path = r"\models\checkpoints\v1-5-pruned-emaonly.gguf"

    # Load model (if explicitly specified or use default)
    #model_to_load = args.load_model if args.load_model else default_model_path
    #print(f"Loading model: {model_to_load}")
    #if not client.load_model(model_to_load):
    #    print("Model loading failed.")
    #    return
    
    # Check server status
    if args.status:
        status = client.check_status()
        print("Server status:", json.dumps(status, indent=2))
        return
    
    # Check server availability
    status = client.check_status()
    if "error" in status:
        print(f"Server unavailable: {status['error']}")
        return
    
    print("Server available:", status)
    
    # Generate image
    start_time = time.time()
    
    result = client.generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        save_path=args.output
    )
    
    if result:
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.1f} seconds")
        print(f"Result: {result}")
    else:
        print("Generation failed")

if __name__ == "__main__":
    main()

# Usage examples:
"""
# Simple generation
python sd_client.py --prompt "a beautiful landscape with mountains and lake"

# With additional parameters
python sd_client.py --prompt "portrait of a cat" --negative "blurry, ugly" --width 768 --height 768 --steps 30

# Check server status
python sd_client.py --status
"""
