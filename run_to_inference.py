import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


from diffusion_networks.diffusion_model_0 import Diffusion_Model_0

from utils.dataloaders import cifar10_32x32_loader
from utils.dataloaders import custom_32x32_loader
from utils.dataloaders import custom_64x64_loader

from utils.data_manipulation import count_spiral_dots

from utils.inference import inferencing


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

torch.use_deterministic_algorithms(False, warn_only=True)  # this is repeated two more times in the warning/profiler blocks

torch.set_float32_matmul_precision('medium')

'''
torch._dynamo.config.suppress_errors = True  resolves issues with torch.compile() failing
Many warnings will be shown but torch.compile will run properly
'''
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False



device = "cuda" if torch.cuda.is_available() else "cpu"



def recolor_cifar10_32x32_run():
    train_dataset, test_dataset = cifar10_32x32_loader()
    model = Diffusion_Model_0(input_resolution=(32, 32)).to(device)
    model_name = '_NAME_OF_MODEL_W/O_.pt'
    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    model_name_file = model_name + '_NAME_OF_IMAGE_'
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')


    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)

    timesteps_int = 20
    timesteps = torch.tensor([timesteps_int]).to(device)
    beta_schedule = torch.linspace(0, 0.99, timesteps_int).to(device)

    image_wanted = 0

    inferencing(model, test_loader, model_name_file, timesteps, beta_schedule, image_wanted, device)


def recolor_custom_64x64_run():
    train_dataset, test_dataset = custom_64x64_loader()
    model = Diffusion_Model_0(input_resolution=(64, 64)).to(device)
    model_name = '_NAME_OF_MODEL_W/O_.pt'
    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    model_name_file = model_name + '_NAME_OF_IMAGE_'
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)

    timesteps_int = 20
    timesteps = torch.tensor([timesteps_int]).to(device)
    beta_schedule = torch.linspace(0, 0.99, timesteps_int).to(device)

    image_wanted = 0

    inferencing(model, test_loader, model_name_file, timesteps, beta_schedule, image_wanted, device)


def spiral_cifar10_32x32_run():
    train_dataset, test_dataset = cifar10_32x32_loader()
    size = 32
    model = Diffusion_Model_0(input_resolution=(size, size)).to(device)
    model_name = '_NAME_OF_MODEL_W/O_.pt'
    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    model_name_file = model_name + '_NAME_OF_IMAGE_'
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)

    dot_radius = 2
    scale = 2.5
    angle_degrees = 137.5
    timesteps_int = count_spiral_dots(size=size, scale=scale, angle_degrees=angle_degrees)

    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)

    image_wanted = 0

    inferencing(model, test_loader, model_name_file, timesteps_int, beta_schedule, image_wanted, device, dot_radius, scale, angle_degrees)

def spiral_custom_64x64_run():
    train_dataset, test_dataset = custom_64x64_loader()
    size = 64
    model = Diffusion_Model_0(input_resolution=(size, size)).to(device)
    model_name = '_NAME_OF_MODEL_W/O_.pt'
    model_path = os.path.join("models_checkpointed", f"{model_name}.pt")
    model_name_file = model_name + '_NAME_OF_IMAGE_'
    if os.path.isfile(model_path):
        print(f'Loading saved model {model_name} to use...')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Saved model is loaded.')
    else:
        print('There was an issue loading the model, please check for correct spelling or placement of the model.')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)

    dot_radius = 2
    scale = 2.5
    angle_degrees = 137.5
    timesteps_int = count_spiral_dots(size=size, scale=scale, angle_degrees=angle_degrees)

    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)

    image_wanted = 0

    inferencing(model, test_loader, model_name_file, timesteps_int, beta_schedule, image_wanted, device, dot_radius, scale, angle_degrees)


# recolor_cifar10_32x32_run()
# recolor_custom_64x64_run()
# spiral_cifar10_32x32_run()
spiral_custom_64x64_run()


