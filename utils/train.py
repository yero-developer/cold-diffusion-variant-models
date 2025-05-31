import torch
import time
import os
import datetime


from utils.data_manipulation import (get_subset_loader,
                                     apply_spiral_masks_batch,
                                     generate_spiral_mask_list,
                                     add_grayscale)



def find_recent_model(folder_path: str, substring: str):
    matching_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if substring in f and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not matching_files:
        return None, None

    latest_file = max(matching_files, key=os.path.getmtime)
    file_name = os.path.basename(latest_file)
    return latest_file, file_name


def gradient_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

    loss_dx = torch.mean(torch.abs(pred_dx - target_dx))
    loss_dy = torch.mean(torch.abs(pred_dy - target_dy))

    return loss_dx + loss_dy

def training(model_type,
             train_dataset,
             train_loader,
             test_loader,
             model,
             optimizer,
             scheduler,
             criterion,
             accumulation_steps,
             epochs_limit,
             train_batch_size,
             device,
             timesteps_int,
             beta_schedule,
             dot_radius=None,
             scale=None,
             angle_degrees=None,
             continue_train=False):

    train_loaded = train_loader
    best_loss = None
    start_epoch = 0
    current_epoch = 0
    scaler = torch.GradScaler()
    os.makedirs("models_checkpointed", exist_ok=True)
    model_name = ''
    if continue_train == True:

        model_path, model_name = find_recent_model('models_checkpointed', model_type)

        if model_path is not None:
            if os.path.isfile(model_path):
                print(f'Loading saved model {model_name} to continue training...')
                checkpoint = torch.load(model_path, map_location='cuda')  # or 'cpu'
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
                start_epoch = checkpoint['epoch'] + 1  # continue from next epoch
                best_loss = checkpoint['loss']
                print('Saved model is loaded.')

                pre_model_name_file = model_name.split('_')[:-1]
                model_name_file = '_'.join(pre_model_name_file) + f'_{start_epoch}.pt'
            else:
                print('No recent models were found, starting a new training run.')
                now = datetime.datetime.now()
                formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                model_name = model_type + '_' + formatted_time + '_0'
        else:
            print('No recent models were found, starting a new training run.')
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            model_name = model_type + '_' + formatted_time + '_0'
    else:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = model_type + '_' + formatted_time + '_0'

        print('Starting a new training run.')


    print('Compiling model for faster training...')
    compiled_mode = ''
    try:
        model = torch.compile(model, mode='default')
        compiled_mode = 'default'
    except Exception as e:
        model = model
        compiled_mode = 'no_compile'
    print(f'The model using compiled mode: {compiled_mode}')


    model.train()
    train_losses = []

    if best_loss is None:
        best_loss = 100000


    start_time = time.time()
    print('Starting training.')
    acc_steps_time = 0

    if dot_radius is not None:
        all_spiral_numbers = list(range(timesteps_int))
        spiral_cache_mask = generate_spiral_mask_list(all_spiral_numbers,
                                  size=next(iter(train_loaded))[0].shape[-1],
                                  dot_radius=dot_radius,
                                  angle_degrees=angle_degrees,
                                  scale=scale,
                                  device=device)

    for epoch in range(start_epoch, epochs_limit):
        start_epoch_time = time.time()
        epoch_loss = 0

        print(f'Starting epoch: {epoch}')
        start_time_acc = time.time()
        for step, (images, labels) in enumerate(train_loaded):
            images = images.to(device, non_blocking=True)

            t = torch.randint(0, timesteps_int, (images.shape[0],)).to(device, non_blocking=True)

            if dot_radius is None:
                x_t, noise, x_cond = add_grayscale(images.clone(), t, beta_schedule)
            else:
                found_spiral_masks = [spiral_cache_mask[i] for i in t]
                full_spiral_mask = spiral_cache_mask[-1].unsqueeze(dim=0)
                random_spiral_mask_batch = torch.stack(found_spiral_masks, dim=0)

                x_cond = images * (1 - full_spiral_mask.to(images.device))
                noise = random_spiral_mask_batch.to(images.device)
                x_t = images * (1 - random_spiral_mask_batch.to(images.device))

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x_t, t, x_cond)
                if dot_radius is None:
                    loss = criterion(outputs, noise) + 0.1 * gradient_loss(outputs, noise)
                else:
                    loss = criterion(outputs * noise, images * noise)

                loss = loss / accumulation_steps  # Normalize for accumulation now

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loaded):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                if acc_steps_time < 10:
                    end_time_acc = time.time()
                    acc_time_elapsed = end_time_acc - start_time_acc
                    hours, rem = divmod(acc_time_elapsed, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print(f'Epoch {epoch} of a accumulated step took: {int(hours)}h {int(minutes)}m {seconds:.2f}s,')
                    acc_steps_time += 1
                    start_time_acc = time.time()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loaded)

        scheduler.step(avg_train_loss)
        train_losses.append(avg_train_loss)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            pre_model_name_file = model_name.split('_')[:-1]
            model_name_file = '_'.join(pre_model_name_file) + f'_{epoch}.pt'
            model_path = os.path.join("models_checkpointed", model_name_file)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # ← include this
                'loss': avg_train_loss,
            }, model_path)
            print('New lowest loss, saving model.')
            current_epoch = epoch

        end_epoch_time = time.time()
        epoch_time_elapsed = end_epoch_time - start_epoch_time
        hours, rem = divmod(epoch_time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Epoch {epoch} took: {int(hours)}h {int(minutes)}m {seconds:.2f}s, loss at {avg_train_loss}')

    end_time = time.time()
    es = end_time - start_time
    hours, rem = divmod(es, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'End training took: {int(hours)}h {int(minutes)}m {seconds:.2f}s')
