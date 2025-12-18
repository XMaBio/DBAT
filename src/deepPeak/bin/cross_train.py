#!/usr/bin/env python3
import torch
import numpy as np
import os
from tqdm import tqdm

from deepPeak.model.model import (
    setup_model,
    setup_training
)
from deepPeak.model.loss import RatioLoss
from deepPeak.data.cross_prepare_data import (
    prepare_datasets,
    create_data_loaders
)


def train(
    genome_path=None,
    ref_wt_path=None,
    ref_mut_path=None,
    output_dir=None,
    data_params=None,
    train_params=None
):

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dataset, val_dataset, test_dataset, all_chroms = prepare_datasets(
        genome_path, ref_wt_path, ref_mut_path, data_params, train_params
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, train_params['batch_size']
    )


    # Save statistics
    original_dataset = train_dataset
    all_stats = {
        'wt_stats': original_dataset.wt_stats,
        'mut_stats': original_dataset.mut_stats,
        'ratio_stats': original_dataset.ratio_stats,
        'global_ratio': original_dataset.global_ratio
    }
    np.savez(os.path.join(output_dir, "Ref_stats.npz"), **all_stats)

    # setup model
    model = setup_model(0, data_params['bin_size'], device)

    scaler, optimizer, scheduler = setup_training(
        model, train_params['use_amp'], device
    )

    criterion = RatioLoss(alpha=0.1)

    # Training loop
    training_loop(model, train_loader, val_loader, criterion, optimizer,
                   scaler, scheduler, device, train_params, output_dir)

    # Evaluation
    evaluation(model, test_loader, output_dir, criterion, device, train_params['use_amp'])



def training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    train_params,
    output_dir
):

    best_val_loss = float('inf')

    for epoch in range(train_params['epochs']):
        # Train and validate
        print(f"\nEpoch {epoch + 1}/{train_params['epochs']}")
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, train_params['use_amp'])

        avg_val_loss = validate_epoch(model, val_loader, criterion,
                    device, train_params['use_amp'])

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "Ref_model.pth"))

        print(f"Epoch {epoch + 1} summary:")
        print(f"- Train loss: {avg_train_loss:.4f}")
        print(f"- Validation loss: {avg_val_loss:.4f}")



def train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        device,
        use_amp
):
    model.train()
    train_loss = 0
    #train_mse = 0
    #train_penalty = 0

    train_bar = tqdm(train_loader, desc="Training", unit="seq")
    for batch_idx, batch in enumerate(train_bar):
        seq, target_ratio, _ = batch
        seq, target_ratio = seq.to(device), target_ratio.to(device)

        optimizer.zero_grad()

        # Forward Propagation with mixed precision
        if device.type == 'cuda' and use_amp:
            with torch.cuda.amp.autocast():
                pred_ratio = model(seq)
                total_loss, mse_loss, penalty_loss = criterion(pred_ratio, target_ratio)
        else:
            pred_ratio = model(seq)
            total_loss, mse_loss, penalty_loss = criterion(pred_ratio, target_ratio)

        if torch.isnan(total_loss).any():
            print(f"NaN detected in loss at batch {batch_idx}")
            continue

        # Backward Propagation
        if device.type == 'cuda' and use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss += total_loss.item()
        #train_mse += mse_loss.item()
        #train_penalty += penalty_loss.item()
        train_bar.set_postfix(loss=f"{total_loss.item():.4f}")

    return train_loss / len(train_loader)



def validate_epoch(
        model,
        val_loader,
        criterion,
        device,
        use_amp
):
    model.eval()
    val_loss = 0
    val_mse = 0
    #all_pred_ratios = []
    #all_true_ratios = []

    val_bar = tqdm(val_loader, desc="Validation", unit="seq")
    with torch.no_grad():
        for batch in val_bar:
            seq, target_ratio, _ = batch
            seq, target_ratio = seq.to(device), target_ratio.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    pred_ratio = model(seq)
                    total_loss, mse_loss, _ = criterion(pred_ratio, target_ratio)
            else:
                pred_ratio = model(seq)
                total_loss, mse_loss, _ = criterion(pred_ratio, target_ratio)

            val_loss += total_loss.item()
            val_mse += mse_loss.item()
            #all_pred_ratios.append(pred_ratio.cpu().numpy())
            #all_true_ratios.append(target_ratio.cpu().numpy())

            val_bar.set_postfix(loss=f"{val_loss:.4f}")

    avg_val_loss = val_loss / len(val_loader)
    #pred_ratios = np.concatenate(all_pred_ratios)
    #true_ratios = np.concatenate(all_true_ratios)

    return avg_val_loss



def evaluation(
    model,
    test_loader,
    output_dir,
    criterion,
    device,
    use_amp
):
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")

    # Load best model and run predictions
    state_dict = torch.load(os.path.join(output_dir, "Ref_model.pth"))
    model.load_state_dict(state_dict)
    model.eval()

    test_loss = 0
    test_bar = tqdm(test_loader, desc="Evaluation", unit="seq")
    with torch.no_grad():
        for seq, target_ratio, _ in test_bar:
            seq, target_ratio = seq.to(device), target_ratio.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    pred_ratio = model(seq)
                    total_loss, mse_loss, _ = criterion(pred_ratio, target_ratio)
            else:
                pred_ratio = model(seq)
                total_loss, mse_loss, _ = criterion(pred_ratio, target_ratio)

            test_loss += total_loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    
    print(f"\nOutput Files:")
    print(f"  Best model: {os.path.join(output_dir, 'Ref_model.pth')}")
    print(f"  Statistics: {os.path.join(output_dir, 'Ref_stats.npz')}")
    
    print(f"\nCross species training complete!")
    print("──────────────────────────────────────────────────────")
