# Infinite Data Generator
### Utility for generating a vast majority of plausible, realistic-looking images for computer vision applications using Stable Diffusion  
  
## Goals
TO DO  
	
## Installation and Use
After downloading/cloning the repository, you can build the environment in Anaconda using `conda env create -f environment.yml`.The project is controlled through the `background.csv` and `classes.csv` files.  
  
## Process
This project uses Stable Diffusion text-to-image and in-painting to generate data that is probably realistic enough for early training but not realistic enough for fine-tuning.  
  
<p align="center">
  <img src="https://github.com/ianjwhitehouse/infinite-data-generator/assets/15909624/b52996f4-1fc1-458a-9757-79c0d9a0dc03" alt="Diagram showing the image generation process" />
</p>
  
The process uses the Stable Diffusion model twice.  First, it generates a set number of background images based on the prompt(s) in `background.png`.  
  
<p align="center">
  <img src="https://github.com/ianjwhitehouse/infinite-data-generator/assets/15909624/ca3c04b1-57f7-4308-bbc0-bde78671aff6" alt="Diagram showing Stable Diffusion generating two background images" />
</p>
  
Then it uses the prompt(s) in `classes.csv` to in-paint the class into set areas of the images.  These areas match the bounding boxes created based on the other parameters within the `classes.csv` file.
  
<p align="center">
  <img src="https://github.com/ianjwhitehouse/infinite-data-generator/assets/15909624/191b2f78-e6fd-4121-9a59-7371bb948499" alt="Diagram showing Stable Diffusion generating two background images" />
</p>
  
Finally, the resulting images and bounding boxes can be used to train a classification model.

## Results
Generate some example datasets and link them below.  
  
