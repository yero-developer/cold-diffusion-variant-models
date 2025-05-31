import torch
from torch.utils.data import DataLoader, Subset
import math
import random



def get_subset_loader(dataset, subset_size, batch_size):
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)



def add_grayscale(images, t, beta_schedule):
    B, C, H, W = images.shape
    assert C == 3, "Expected RGB images"

    beta_t = beta_schedule[t].view(-1, 1, 1, 1)
    beta_t = beta_t.expand(-1, 3, 1, 1)

    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=images.device).view(1, 3, 1, 1)
    gray = torch.sum(images * weights, dim=1, keepdim=True)
    true_gray_image = gray.repeat(1, 3, 1, 1)

    true_noise = (true_gray_image - images) * beta_t
    x_t = images + true_noise

    return x_t, true_noise, true_gray_image


def reverse_grayscale(x_t, t, gray_noise, beta_schedule, x_cond):
    x_0 = x_t - gray_noise
    if t > 0:
        beta_t_one_less = beta_schedule[t-1].view(-1, 1, 1, 1)
        x_s = x_0 + (x_cond - x_0) * beta_t_one_less

    else:
        x_s = x_0

    return x_s, x_0


def count_spiral_dots(size=64, scale=2.5, angle_degrees=137.5):
    center = size // 2
    max_radius = math.hypot(center, center)
    angle_rad = math.radians(angle_degrees)

    i = 0
    count = 0
    while True:
        r = scale * math.sqrt(i)
        if r > max_radius:
            break
        theta = i * angle_rad
        x = int(center + r * math.cos(theta))
        y = int(center + r * math.sin(theta))
        if 0 <= x < size and 0 <= y < size:
            count += 1
        i += 1
    return count


def create_spiral_mask_rgb_single(size=64, dot_radius=2, max_points=100, angle_degrees=137.5, scale=2.5):
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    center = size // 2
    angle_rad = math.radians(angle_degrees)
    max_radius = math.hypot(center, center)

    i = 0
    count = 0
    while count < max_points:
        r = scale * math.sqrt(i)
        if r > max_radius:
            break
        theta = i * angle_rad
        x = int(center + r * math.cos(theta))
        y = int(center + r * math.sin(theta))
        if 0 <= x < size and 0 <= y < size:
            for dy in range(-dot_radius, dot_radius + 1):
                for dx in range(-dot_radius, dot_radius + 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < size and 0 <= xx < size and dx**2 + dy**2 <= dot_radius**2:
                        mask[0, yy, xx] = 1.0
            count += 1
        i += 1

    rgb_mask = mask.repeat(3, 1, 1)
    return rgb_mask


def apply_spiral_masks_batch(images, max_points_tensor, dot_radius=2, angle_degrees=137.5, scale=2.5):
    B, C, H, W = images.shape
    masks = []

    for max_points in max_points_tensor.tolist():
        mask = create_spiral_mask_rgb_single(
            size=H,
            dot_radius=dot_radius,
            max_points=max_points,
            angle_degrees=angle_degrees,
            scale=scale
        )
        masks.append(mask)

    spiral_mask_batch = torch.stack(masks)
    return spiral_mask_batch


def reverse_spiral(x_t, t, pred_noise, spiral_mask, dot_radius, angle_degrees, scale):
    lower_spiral_mask_batch = spiral_mask

    x_0 = x_t + pred_noise * lower_spiral_mask_batch

    if t > 0:
        lower_spiral_mask_batch = apply_spiral_masks_batch(x_0,
                                                          torch.tensor([t-1]).repeat(x_0.shape[0]).to(x_0.device),
                                                          dot_radius=dot_radius,
                                                          angle_degrees=angle_degrees, scale=scale)
        lower_spiral_mask_batch = lower_spiral_mask_batch.to(x_0.device)
        x_s = x_0 * (1 - lower_spiral_mask_batch)

    else:
        x_s = x_0

    return x_s, x_0





def generate_spiral_mask_list(
    max_points_list,
    size=64,
    dot_radius=2,
    angle_degrees=137.5,
    scale=2.5,
    device='cpu'
):

    masks = []
    center = size // 2
    angle_rad = math.radians(angle_degrees)
    max_radius = math.hypot(center, center)

    if isinstance(max_points_list, torch.Tensor):
        max_points_list = max_points_list.tolist()

    for max_points in max_points_list:
        mask = torch.zeros((1, size, size), dtype=torch.float32, device=device)

        i = 0
        count = 0
        while count < max_points:
            r = scale * math.sqrt(i)
            if r > max_radius:
                break
            theta = i * angle_rad
            x = int(center + r * math.cos(theta))
            y = int(center + r * math.sin(theta))
            if 0 <= x < size and 0 <= y < size:
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < size and 0 <= xx < size and dx**2 + dy**2 <= dot_radius**2:
                            mask[0, yy, xx] = 1.0
                count += 1
            i += 1

        rgb_mask = mask.repeat(3, 1, 1)
        masks.append(rgb_mask)

    return masks
