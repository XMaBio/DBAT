#!/usr/bin/env python3
import numpy as np
import os
import pyBigWig
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def save_stats(train_dataset, output_dir, num_suppressive_channels):
    stats_path = os.path.join(output_dir, "Peak_stats.npz")

    if hasattr(train_dataset, 'dataset'):
        stats = train_dataset.dataset.coverage_stats
    else:
        stats = train_dataset.coverage_stats

    extra_info = {
        'num_suppressive_channels': num_suppressive_channels,
    }

    # Save both stats and metadata
    np.savez(stats_path, **stats, **extra_info)



def denormalize(data, coverage_stats):
    data_orig = data * coverage_stats['std'] + coverage_stats['mean']

    if coverage_stats['transformation'] == 'log2':
        data_orig = np.exp2(data_orig) - 1
    else:
        data_orig = np.expm1(data_orig)

    return np.maximum(data_orig, 0)  # Ensure non-negative



def write_prediction(bigwig_path, chrom_data):
    # Get chromosome lengths
    chrom_lengths = estimate_chr_len(chrom_data)

    success = write_bigwig(bigwig_path, chrom_data, chrom_lengths)

    if not success:
        print(f"WARNING: Failed to write BigWig file {bigwig_path}")
    return success



def estimate_chr_len(chrom_data):
    chrom_lengths = {}
    for chrom, data in chrom_data.items():
        if data['ends']:
            chrom_lengths[chrom] = int(max(data['ends']) + 1000)  # Add buffer
        else:
            chrom_lengths[chrom] = 1000000  # Default length (already int)

    return chrom_lengths



def write_bigwig(output_path, chrom_data, chrom_lengths):
    print(f"\nWriting BigWig file: {output_path}")

    total_entries = 0
    valid_chroms = []

    for chrom in chrom_data:
        if (chrom in chrom_lengths and
                'starts' in chrom_data[chrom] and
                'ends' in chrom_data[chrom] and
                'values' in chrom_data[chrom] and
                len(chrom_data[chrom]['starts']) > 0 and
                len(chrom_data[chrom]['ends']) > 0 and
                len(chrom_data[chrom]['values']) > 0):
            total_entries += len(chrom_data[chrom]['starts'])
            valid_chroms.append(chrom)

    if total_entries == 0:
        print("WARNING: No data to write to BigWig file")
        return False

    try:
        with pyBigWig.open(output_path, "w") as bw:
            # Add header with chromosome lengths
            header = []
            for chrom in valid_chroms:
                if chrom in chrom_lengths:
                    length = chrom_lengths[chrom]
                    header.append((chrom, int(length)))

            if not header:
                print("WARNING: No valid chromosomes for BigWig header")
                return False

            bw.addHeader(header)

            # Add entries for each chromosome
            for chrom in valid_chroms:
                starts = chrom_data[chrom]['starts']
                ends = chrom_data[chrom]['ends']
                values = chrom_data[chrom]['values']

                # Ensure all arrays have same length
                min_len = min(len(starts), len(ends), len(values))

                if min_len > 0:
                    # Convert to lists and ensure correct data types
                    starts_list = [int(x) for x in starts[:min_len]]
                    ends_list   = [int(x) for x in ends[:min_len]]
                    values_list = [float(x) for x in values[:min_len]]

                    # Sort by start position (required by BigWig format)
                    sorted_indices = np.argsort(starts_list)
                    starts_sorted = [starts_list[i] for i in sorted_indices]
                    ends_sorted = [ends_list[i] for i in sorted_indices]
                    values_sorted = [values_list[i] for i in sorted_indices]

                    bw.addEntries(
                        [chrom] * min_len,
                        starts_sorted,
                        ends=ends_sorted,
                        values=values_sorted
                    )
                    print(f"  Found {min_len} entries for chromosome {chrom}")

        print(f"Successfully write BigWig file with {total_entries} entries")
        return True

    except Exception as e:
        print(f"Error writing BigWig file: {e}")
        import traceback
        traceback.print_exc()
        return False



def calculate_metrics(predictions, targets):
    # Peak detection metrics
    auprc, auroc, peak_threshold = calculate_auprc(
        targets, predictions, threshold_percentile=90
    )

    return {
        'auprc': auprc,
        'auroc': auroc,
        'peak_threshold': peak_threshold,
        'predictions_mean': np.mean(predictions),
        'predictions_std': np.std(predictions),
        'targets_mean': np.mean(targets),
        'targets_std': np.std(targets)
    }



def calculate_auprc(true_coverage, pred_coverage, threshold_percentile=90):
    true_flat = true_coverage.flatten()
    pred_flat = pred_coverage.flatten()

    if len(true_flat) > 1e6:
        idx = np.random.choice(len(true_flat), int(1e6), replace=False)
        true_flat = true_flat[idx]
        pred_flat = pred_flat[idx]

    threshold = np.percentile(true_flat, threshold_percentile)
    true_binary = (true_flat > threshold).astype(int)

    precision, recall, _ = precision_recall_curve(true_binary, pred_flat)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(true_binary, pred_flat)

    return auprc, auroc, threshold


