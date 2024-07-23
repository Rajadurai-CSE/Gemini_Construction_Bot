import replicate
def generate_image_fun(prompt):
    input = {
    "width": 768,
    "height": 768,
    "prompt": prompt,
    "refine": "expert_ensemble_refiner",
    "apply_watermark": False,
    "num_inference_steps": 25
    }
    output = replicate.run(
    "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    input=input
    )
    return output