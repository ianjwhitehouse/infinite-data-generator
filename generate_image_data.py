# Imports
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
import numpy as np
from PIL import Image


# Load diffusion models
background_model = AutoPipelineForText2Image.from_pretrained(
	"stabilityai/sdxl-turbo",
).to("cuda")

# inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting"
# ).to("cuda")
inpainting_model = AutoPipelineForInpainting.from_pipe(background_model).to("cuda")

default_negative_prompt = "cartoon drawing, dark"

# Generate background img or img(s)
def gen_background_img(prompt, negative_prompt=default_negative_prompt, num_imgs_per_prompt=8, width=640, height=480):
    return background_model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_imgs_per_prompt,
        width=width,
        height=height,
        # num_inference_steps=1
    ).images

# Generate bounding boxes (aspect_ratio is in width/height)
def gen_bounding_box_img(
    aspect_ratio_min=None, aspect_ratio_max=None,
    width_min=None, height_min=None, width_max=None, height_max=None,
    img_width=640, img_height=480
):
    # Set the width and height if it the aspect ratio is defined
    if aspect_ratio_min and aspect_ratio_max:
        assert not (width_min or height_min or width_max or height_max)
        aspect_ratio = np.random.uniform(low=aspect_ratio_min, high=aspect_ratio_max)

    if width_min and width_max and height_min and height_max:
        assert not (aspect_ratio_min or aspect_ratio_max)
        width = np.random.randint(width_min, width_max)
        height = np.random.randint(height_min, height_max)
    else:
        if aspect_ratio < 1:
            height = np.random.randint(img_height//4, img_height//2)
            width = height * aspect_ratio
        else:
            width = np.random.randint(img_width//4, img_width//2)
            height = width/aspect_ratio

    position_x = np.random.randint(0, img_width-width)
    position_y = np.random.randint(0, img_height-height)

    img = np.zeros((img_height, img_width, 1))
    print(position_y, int(position_y + height), position_x, int(position_x + width))
    img[position_y:int(position_y + height), position_x:int(position_x + width), :] = 1
    return img

# Generate inpainted images (all of these should be single values)
def inpaint_class_into_images(image, bounding_box_information, class_prompt, negative_prompt=default_negative_prompt, width=640, height=480):
    bounding_box = gen_bounding_box_img(**bounding_box_information)
    print(class_prompt)
    return inpainting_model(
        prompt=class_prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=bounding_box,
        width=width,
        height=height,
        strength=0.5,
        # num_inference_steps=4
    ).images, bounding_box
