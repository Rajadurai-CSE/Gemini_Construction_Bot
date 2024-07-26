import replicate
def generate_image_fun(prompt):
    """
    
    Image Generation Function to call external image generation api to yield image output to users.

    Args: prompt -> Generate a Prompt addressing user enquiry

    generate_image_fun(prompt)

    """
    input = {
    "width": 768,
    "height": 768,
    "prompt": prompt,
    "refine": "expert_ensemble_refiner",
    "apply_watermark": False,
    "num_inference_steps": 25
    }
    output = replicate.run(
    "stability-ai/stable-diffusion-3",
    input=input
    )
    return output