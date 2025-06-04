import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from PIL import Image
from utils.data_manipulation import (add_grayscale,
                                     apply_spiral_masks_batch,
                                     reverse_spiral,
                                     generate_spiral_mask_list,
                                     reverse_grayscale)


def re_alter_image_list(images):
    images = [frame.permute(1, 2, 0) for frame in images]
    images = [frame.cpu().detach().numpy() for frame in images]

    images = [
        (np.nan_to_num(np.clip(img, 0.0, 1.0)) * 255).astype(np.uint8)
        for img in images
    ]
    return images


def inferencing(model, test_loader, model_name_file, timesteps, beta_schedule, image_wanted, device, dot_radius=None, scale=None, angle_degrees=None):
    compiled_mode = ''
    try:
        model = torch.compile(model, mode='default')
        compiled_mode = 'default'
    except Exception as e:
        model = model
        compiled_mode = 'no_compile'
    print(f'The model using compiled mode: {compiled_mode}')


    images = None
    for step, (image, label) in enumerate(test_loader):
        if step == image_wanted:
            images = image.to(device, non_blocking=True)
            break

    if dot_radius is None:
        x1, _1, _2 = add_grayscale(images, timesteps-1, beta_schedule)
        x1 = x1.to(device, non_blocking=True)
        x = x1.clone()
        x_cond = x1.clone()
    else:
        full_spiral_mask_batch = apply_spiral_masks_batch(images, torch.tensor([timesteps]).repeat(images.shape[0]).to(device, non_blocking=True), dot_radius=dot_radius,
                                                          angle_degrees=angle_degrees, scale=scale)
        full_spiral_mask_batch = full_spiral_mask_batch.to(device, non_blocking=True)
        x1 = images * (1 - full_spiral_mask_batch)

        x1 = x1.to(device, non_blocking=True)
        x = x1.clone()
        x_cond = x1.clone()

        all_spiral_numbers = list(range(timesteps))
        spiral_cache_mask = generate_spiral_mask_list(all_spiral_numbers,
                                                      size=next(iter(test_loader))[0].shape[-1],
                                                      dot_radius=dot_radius,
                                                      angle_degrees=angle_degrees,
                                                      scale=scale,
                                                      device=device)


    recolored_image_list = []
    pred_noise_list = []
    original_grayscaled_image = []
    original_color_image = []
    reconstructed_image = []


    model.eval()
    with torch.inference_mode():
        for t in reversed(range(timesteps)):
            original_grayscaled_image.append(x1.clone().detach().cpu().squeeze(0))
            aR, aG, aB = images[0][0], images[0][1], images[0][2]
            xa = torch.stack([aR, aG, aB], dim=0)
            original_color_image.append(xa.clone().detach().cpu())
            t_tensor = torch.full((1,), t, device=device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predicted_noise = model(x, t_tensor, x_cond)
            pred_noise_list.append(predicted_noise.clone().detach().cpu().squeeze(0))

            if dot_radius is None:
                x, x_0 = reverse_grayscale(x, torch.tensor([t]).to(device, non_blocking=True), predicted_noise, beta_schedule, x_cond)
            else:
                spiral_mask = spiral_cache_mask[t]
                x, x_0 = reverse_spiral(x, t, predicted_noise, spiral_mask, dot_radius, angle_degrees, scale)


            recolored_image_list.append(x.clone().detach().cpu().squeeze(0))
            reconstructed_image.append(x_0.clone().detach().cpu().squeeze(0))


    def unnormalize(tensor):
        for x in range(len(tensor)):
            tensor[x] = (tensor[x] + 1) / 2
        return tensor


    recolored_image_list = unnormalize(recolored_image_list)
    pred_noise_list = unnormalize(pred_noise_list)
    original_grayscaled_image = unnormalize(original_grayscaled_image)
    original_color_image = unnormalize(original_color_image)
    reconstructed_image = unnormalize(reconstructed_image)

    for _ in range(12):
        recolored_image_list = [recolored_image_list[0] ]+ recolored_image_list
        pred_noise_list = [pred_noise_list[0]] + pred_noise_list
        original_grayscaled_image = [original_grayscaled_image[0]] + original_grayscaled_image
        original_color_image = [original_color_image[0]] + original_color_image
        reconstructed_image = [reconstructed_image[0]] + reconstructed_image

    for _ in range(12):
        recolored_image_list.append(recolored_image_list[-1])
        pred_noise_list.append(pred_noise_list[-1])
        original_grayscaled_image.append(original_grayscaled_image[-1])
        original_color_image.append(original_color_image[-1])
        reconstructed_image.append(reconstructed_image[-1])

    recolored_image_list = re_alter_image_list(recolored_image_list)
    pred_noise_list = re_alter_image_list(pred_noise_list)
    original_grayscaled_image = re_alter_image_list(original_grayscaled_image)
    original_color_image = re_alter_image_list(original_color_image)
    reconstructed_image = re_alter_image_list(reconstructed_image)

    combined_frames = [
        np.hstack((img1, img2, img5, img3, img4))
        for img1, img2, img5, img3, img4 in zip(recolored_image_list, pred_noise_list, reconstructed_image, original_grayscaled_image, original_color_image)
    ]

    os.makedirs("models_epoch_inferences", exist_ok=True)
    for idx, frame in enumerate(combined_frames):
        img = Image.fromarray(frame)
        if idx == 0:
            img.save(f'models_epoch_inferences/{model_name_file}_initial.png')
        if idx == (len(combined_frames) - 1):
            img.save(f'models_epoch_inferences/{model_name_file}_final.png')

    combined_frames += combined_frames[::-1]
    num_frames = len(combined_frames)

    height, width = combined_frames[0].shape[:2]
    dpi = 100

    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(combined_frames[0])
    ax.axis('off')

    def update(frame_idx):
        im.set_array(combined_frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=True,
    )

    ani.save(f'models_epoch_inferences/{model_name_file}.mp4', fps=24, writer="ffmpeg")
    ani.save(f'models_epoch_inferences/{model_name_file}.gif', fps=24, writer="pillow")
    plt.close()
