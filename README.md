<h1 align="center">cold-diffusion-variant-models</h1>

<p align="center">
  This project is Yero's Cold Diffusion variant models. 
</p>

## About
I was inspired by the paper <b>Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise</b> https://arxiv.org/abs/2208.09392 and I wanted to create my own variant models based on this method. These models are not tuned for performance but a proof of concept that shows it works. Please read the paper that is linked/cited to learn more about Cold Diffusion, they have their own models and examples with code as well.

Bansal, A., Borgnia, E., Chu, H., Li, J.S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., Goldstein, T. (2022). Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise. arXiv preprint arXiv:2208.09392.

## Youtube Video
Here is a short video showcasing the visualizations of the examples below:
<br>
LINK GOES HERE WHEN READY

## Variant Models
- **Recolor**: Designed to recolor grayscaled images.
- **Spiral**: An inpainting method that fills gray areas in a spiral pattern.
  - **Golden Spiral**: Comes from the golden ratio which makes the golden angle that is used to create the spiral pattern.
  - **Metallic/Plastic Spirals**: There are other ratios that can be used to create spiral patterns and is analogous to the golden spiral. Additional spiral patterns has not been implemented yet.   

## Examples 64x64 Sized
The images in the gif below are as follow:
<br>
|Diffusion|Predicted Noise| x_0* | Initial | Original |
| ------------- | ------------- |------------- |------------- |------------- |
<p align="center">
  Recolor
  <br>
  <img src="https://github.com/user-attachments/assets/bf06b178-cd28-418b-a1b9-f4e0475a8367" alt="animated" />
  <br>
  <img src="https://github.com/user-attachments/assets/e1703a6f-03a7-4eb5-9f66-6aa0821f72a1" alt="animated" />
  <br>
  Spiral
  <br>
  <img src="https://github.com/user-attachments/assets/c0468721-32b5-4b2e-9560-4b9b7d5de5e5" alt="animated" />
  <br>
  <img src="https://github.com/user-attachments/assets/f315bc7e-2106-4d16-9a4e-da6b3d93e0ef" alt="animated" />
</p>
<br>
*: x_0 is the predicted reconstruction which gets closer to the final image diffused. 

## How To Run
- To get the required libraries.
```
pip install -r requirements.txt
```
- There are two main scripts to use:
  - run_to_train.py
    - This script is to train the diffision model,
    - 32x32 dataset is using CIFAR10 while 64x64 dataset is a custom dataset you have to manually put into the folder custom_data using the ImageFolder structure in pytorch.
    - Uncomment one of the function near the bottom of the script to train.
  - run_to_inference.py
      - This script is to do inference on a saved diffusion model.
      - Uncomment one of the functions near the bottom of the script to do inference.
        - In the function wanted, change the str variable, model_name, to the saved diffusion model without the '.pt' ending.
        - In the function wanted, change the str variable, model_name_file, to your choice to identify the output.

## Additional Notes:
- Recolor does not truly recolor grayscaled images as the model was trained to recolor a 99% equal weighted grayscaled image. Training it to do 100% causes issues as the possibilities for what color a pixel can be insreases drastically. A way to resolve this issue is to add context to what the 100% grayscaled image should be recolored to - text prompt. In the future I may test this method/model.
- The above examples were not used the models' training.  

## Copyright Information
Copyright 2025 Yeshua A. Romero

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
