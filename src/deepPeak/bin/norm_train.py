#!/usr/bin/env python3
import torch
import numpy as np
import os
from tqdm import tqdm

from deepPeak.model.model import (
    setup_model,
    setup_training
)
from deepPeak.model.loss import BoundaryAwareMSELoss
from deepPeak.data.norm_prepare_data import (
    prepare_datasets,
    create_data_loaders
)
from deepPeak.data.norm_save_data import (
    save_stats,
    denormalize,
    write_prediction,
    calculate_metrics,
)


def train(
    genome_path=None,
    train_path=None,
    output_dir=None,
    suppressive_paths=None,
    data_params=None,
    train_params=None
):

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dataset, val_dataset, test_dataset, all_chroms = prepare_datasets(
        genome_path, train_path, suppressive_paths, data_params, train_params
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, train_params['batch_size']
    )

    # Setup model and training
    num_suppression = len(suppressive_paths) if suppressive_paths else 0

    # Save statistics
    save_stats(train_dataset, output_dir, num_suppression)

    model = setup_model(num_suppression, data_params['bin_size'], device)

    scaler, optimizer, scheduler = setup_training(
        model, train_params['use_amp'], device
    )

    criterion = BoundaryAwareMSELoss(boundary_weight=5.0)

    # Training loop
    training_loop(model, train_loader, val_loader, criterion, optimizer,
                   scaler, scheduler, device, train_params, output_dir)

    # Evaluation
    evaluation(model, test_loader, output_dir, device,
                train_params, data_params)



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
            torch.save(model.state_dict(), os.path.join(output_dir, "Peak_model.pth"))

        print(f"Epoch {epoch + 1} summary:")
        print(f"- Train loss: {avg_train_loss:.4f}")
        print(f"- Val loss: {avg_val_loss:.4f}")



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

    train_bar = tqdm(train_loader, desc="Training", unit="seq")
    for batch_idx, batch in enumerate(train_bar):
        seq, coverage, _ = batch
        seq, coverage = seq.to(device), coverage.to(device)

        optimizer.zero_grad()

        # Forward Propagation with mixed precision
        if device.type == 'cuda' and use_amp:
            with torch.cuda.amp.autocast():
                pred = model(seq)
                loss = criterion(pred, coverage) + 1e-8
        else:
            pred = model(seq)
            loss = criterion(pred, coverage) + 1e-8

        loss = torch.clamp(loss, min=1e-8, max=1e8)

        if torch.isnan(loss).any():
            print(f"NaN detected in loss at batch {batch_idx}")
            continue

        # Backward Propagation
        if device.type == 'cuda' and use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

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
    all_preds = []
    all_targets = []

    val_bar = tqdm(val_loader, desc="Validation", unit="seq")
    with torch.no_grad():
        for batch in val_bar:
            seq, coverage, _ = batch
            seq, coverage = seq.to(device), coverage.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(seq)
                    loss = criterion(pred, coverage)
            else:
                pred = model(seq)
                loss = criterion(pred, coverage)

            val_loss += loss.item()
            #all_preds.append(pred.cpu().numpy())
            #all_targets.append(coverage.cpu().numpy())
            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader)
    #all_preds = np.concatenate(all_preds)
    #all_targets = np.concatenate(all_targets)

    return avg_val_loss



def evaluation(
    model,
    test_loader,
    output_dir,
    device,
    train_params,
    data_params,
):

    # Load best model and run predictions
    model.load_state_dict(torch.load(os.path.join(output_dir, "Peak_model.pth")))

    all_preds, all_targets, chrom_data = run_predictions(
        model, test_loader, output_dir, device, train_params['use_amp'], data_params
    )

    # Load coverage stats for denormalization
    coverage_stats = np.load(os.path.join(output_dir, "Peak_stats.npz"))

    # Convert predictions and targets to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Denormalize predictions and targets to original scale
    all_preds_orig = denormalize(all_preds, coverage_stats)
    all_targets_orig = denormalize(all_targets, coverage_stats)


    # Write predictions in BigWig format
    bigwig_path = os.path.join(output_dir, "test_chrs.bw")
    success = write_prediction(bigwig_path, chrom_data)

    # Calculate metrics
    test_metrics = calculate_metrics(all_preds_orig, all_targets_orig)

    # Print summary
    summary(output_dir, test_metrics, success, bigwig_path)



def run_predictions(
        model,
        test_loader,
        output_dir,
        device,
        use_amp,
        data_params
):
    model.eval()

    # Load coverage stats
    coverage_stats = np.load(os.path.join(output_dir, "Peak_stats.npz"))

    test_bar = tqdm(test_loader, desc="Predicting", unit="seq")
    all_preds = []
    all_targets = []
    chrom_data = {}

    with torch.no_grad():
        for batch in test_bar:
            seq, coverage, coords_batch = batch
            batch_size = seq.size(0)
            seq = seq.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(seq).cpu().numpy()
            else:
                preds = model(seq).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(coverage.numpy())

            # Process predictions for BigWig output
            process_for_bigwig(
                preds, coords_batch, chrom_data, coverage_stats, data_params
            )

    return all_preds, all_targets, chrom_data



def process_for_bigwig(
        preds,
        coords_batch,
        chrom_data,
        coverage_stats,
        data_params
):
    bin_size = data_params['bin_size']

    for i in range(preds.shape[0]):
        chrom = coords_batch[0][i]
        start = coords_batch[1][i]
        end = coords_batch[2][i]
        pred = preds[i]

        # Reverse normalization
        pred = pred * coverage_stats['std'] + coverage_stats['mean']
        if coverage_stats['transformation'] == 'log2':
            pred = np.exp2(pred) - 1
        else:
            pred = np.expm1(pred)

        pred = np.maximum(pred, 0)

        if chrom not in chrom_data:
            chrom_data[chrom] = {'starts': [], 'ends': [], 'values': []}

        # Add predictions for each bin
        for bin_idx, value in enumerate(pred):
            bin_start = start + bin_idx * bin_size
            bin_end = bin_start + bin_size
            chrom_data[chrom]['starts'].append(bin_start)
            chrom_data[chrom]['ends'].append(bin_end)
            chrom_data[chrom]['values'].append(float(value))


def summary(output_dir, test_metrics, bigwig_success, bigwig_path):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nPeak Detection Metrics:")
    print(f"  auPRC: {test_metrics['auprc']:.4f}")
    print(f"  auROC: {test_metrics['auroc']:.4f}")
    print(f"  Peak Threshold (90th %ile): {test_metrics['peak_threshold']:.4f}")

    print(f"\nDistribution Statistics:")
    print(f"  Predictions - Mean: {test_metrics['predictions_mean']:.4f}, Std: {test_metrics['predictions_std']:.4f}")
    print(f"  Targets - Mean: {test_metrics['targets_mean']:.4f}, Std: {test_metrics['targets_std']:.4f}")

    print(f"\nOutput Files:")
    print(f"  Best model: {os.path.join(output_dir, 'Peak_model.pth')}")
    print(f"  Statistics: {os.path.join(output_dir, 'Peak_stats.npz')}")
    print(f"  Predictions: {bigwig_path if bigwig_success else 'FAILED'}")

    print(f"\nTraining complete!")
    print("──────────────────────────────────────────────────────")