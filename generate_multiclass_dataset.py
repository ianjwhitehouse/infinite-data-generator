# Imports
import pandas as pd
from generate_image_data import gen_background_img, inpaint_class_into_images
from random import randint
import numpy as np
from PIL import Image


# Load list of backgrounds
backgrounds = pd.read_csv("backgrounds.csv").to_dict("records")

# Generate backgrounds
background_imgs = []
for bg in backgrounds:
    background_imgs += gen_background_img(bg["prompt"], num_imgs_per_prompt=bg["num_imgs"])

# Load list of classes
classes = pd.read_csv("classes.csv")
classes = classes.replace(np.nan, None)
classes = classes.to_dict("records")

for i, img in enumerate(background_imgs):
    bbs = []
    img.save("output/bg_%d.png" % i)
    for cls in classes:
        cls_bbs = []
        cls_instances = randint(cls["min_appear"], cls["max_appear"])
        for num_cls in range(cls_instances):
            print({k: cls[k] for k in ["aspect_ratio_min", "aspect_ratio_max", "width_min", "height_min", "width_max", "height_max"]})
            new_img, bb = inpaint_class_into_images(
                img,
                {k: cls[k] for k in ["aspect_ratio_min", "aspect_ratio_max", "width_min", "height_min", "width_max", "height_max"]},
                cls["prompt"]
            )                
            img = new_img[0]
            cls_bbs.append(bb.tolist())
            
        if cls_instances < 1:
            bbs.append(np.zeros((768, 768)))
        else:
            bbs.append(np.any(np.array(cls_bbs), axis=0)[:, :, 0])
            print(np.any(np.array(cls_bbs), axis=0).shape)
        
    img.save("output/%d_x.png" % i)
    np.save("output/%d_y.npy" % i, np.stack(bbs, axis=-1))
