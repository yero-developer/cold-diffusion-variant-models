import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.dataloaders import cifar10_32x32_loader
from utils.dataloaders import custom_64x64_loader
from utils.data_manipulation import count_spiral_dots
from utils.train import training

from diffusion_networks.diffusion_model_0 import Diffusion_Model_0


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

torch.use_deterministic_algorithms(False, warn_only=True)

torch.set_float32_matmul_precision('medium')

'''
torch._dynamo.config.suppress_errors = True  resolves issues with torch.compile() failing
Many warnings will be shown but torch.compile will run properly
'''
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False



device = "cuda" if torch.cuda.is_available() else "cpu"



def recolor_cifar10_32x32_run(epochs_limit=30, continue_train=False):
    train_dataset, test_dataset = cifar10_32x32_loader()
    model = Diffusion_Model_0(input_resolution=(32, 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5,
                                                           threshold=0.0001,
                                                           threshold_mode='abs')
    criterion = nn.MSELoss()
    timesteps_int = 20
    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)
    accumulation_steps = 1
    train_batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)
    training('recolor_cifar10_32x32_run', train_dataset, train_loader, test_loader, model,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             beta_schedule, continue_train)

def recolor_custom_64x64_run(epochs_limit=30, continue_train=False):
    train_dataset, test_dataset = custom_64x64_loader()
    model = Diffusion_Model_0(input_resolution=(64, 64)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5,
                                                           threshold=0.0001,
                                                           threshold_mode='abs')
    criterion = nn.MSELoss()
    timesteps_int = 20
    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)
    accumulation_steps = 2
    train_batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)
    training('recolor_custom_64x64_run', train_dataset, train_loader, test_loader, model,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             beta_schedule, continue_train)

def spiral_cifar10_32x32_run(epochs_limit=30, continue_train=False):
    train_dataset, test_dataset = cifar10_32x32_loader()
    size = 32
    model = Diffusion_Model_0(input_resolution=(size, size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5,
                                                           threshold=0.0001,
                                                           threshold_mode='abs')
    criterion = nn.MSELoss()

    accumulation_steps = 1
    train_batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)


    dot_radius = 2
    scale = 2.5
    angle_degrees = 137.5  # golden angle
    # angle_degrees = 210.9  # silver angle
    # angle_degrees = 251.0  # bronze angle
    # angle_degrees = 275.0  # copper angle
    # angle_degrees = 290.7  # nickel angle
    timesteps_int = count_spiral_dots(size=size, scale=scale, angle_degrees=angle_degrees)

    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)


    training('spiral_cifar10_32x32_run', train_dataset, train_loader, test_loader, model,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             beta_schedule, dot_radius, scale, angle_degrees, continue_train)

def spiral_custom_64x64_run(epochs_limit=30, continue_train=False):
    train_dataset, test_dataset = custom_64x64_loader()
    size = 64
    model = Diffusion_Model_0(input_resolution=(size, size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5,
                                                           threshold=0.0001,
                                                           threshold_mode='abs')
    criterion = nn.MSELoss()

    accumulation_steps = 2
    train_batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             persistent_workers=True)


    dot_radius = 2
    scale = 2.5
    angle_degrees = 137.5  # golden angle
    # angle_degrees = 210.9  # silver angle
    # angle_degrees = 251.0  # bronze angle
    # angle_degrees = 275.0  # copper angle
    # angle_degrees = 290.7  # nickel angle
    timesteps_int = count_spiral_dots(size=size, scale=scale, angle_degrees=angle_degrees)

    beta_schedule = torch.linspace(0, 1, timesteps_int).to(device)


    training('spiral_custom_64x64_run', train_dataset, train_loader, test_loader, model,
             optimizer, scheduler, criterion, accumulation_steps, epochs_limit, train_batch_size, device, timesteps_int,
             beta_schedule, dot_radius, scale, angle_degrees, continue_train)


# Call one of the above functions to train a model.

# recolor_cifar10_32x32_run(epochs_limit=100, continue_train=False)
# recolor_custom_64x64_run(epochs_limit=100, continue_train=False)
spiral_cifar10_32x32_run(epochs_limit=10, continue_train=False)
# spiral_custom_64x64_run(epochs_limit=100, continue_train=False)