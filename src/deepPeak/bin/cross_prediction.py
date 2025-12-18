#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import pyBigWig

from deepPeak.model.model import PerformerUNet
from deepPeak.data.genome import get_chr_len, Genome
from deepPeak.data.cross_seq_data import (
    load_bigwig, genome_wide_stats, compute_coverage
)


def prediction(
    genome_path=None,
    targ_wt_path=None,
    model_path=None,
    stats_path=None,
    output_dir=None,
    data_params=None,
    train_params=None
):

    os.makedirs(output_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_params['use_amp']

    seq_len  = data_params['seq_length']
    bin_size = data_params['bin_size']
    num_bins = seq_len // bin_size
    batch_size = train_params['batch_size']

    # Load statistics file
    loaded = np.load(stats_path, allow_pickle=True)
    all_stats = {key: loaded[key].item() for key in loaded.files}

    wt_stats = all_stats['wt_stats']
    mut_stats = all_stats['mut_stats']
    ratio_stats = all_stats['ratio_stats']
    ref_global_ratio = all_stats['global_ratio']

    # Load model directly
    model = PerformerUNet(0, bin_size).to(device)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()


    # Create prediction dataset
    pred_dataset = RatioData(genome_path, targ_wt_path, data_params)

    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Predicting ratios for {len(pred_dataset)} regions...")

    # Predict ratios
    all_pred_ratios = []
    all_wt_signals = []
    all_regions = []

    with torch.no_grad():
        for seq, wt_signal, regions in tqdm(pred_loader, desc="Predicting"):
            seq = seq.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    pred_ratio = model(seq)
            else:
                pred_ratio = model(seq)

            all_pred_ratios.append(pred_ratio.cpu().numpy())
            all_wt_signals.append(wt_signal.numpy())
            all_regions.extend(zip(regions[0], regions[1], regions[2]))

    # Concatenate results
    pred_ratios = np.concatenate(all_pred_ratios)
    wt_signals = np.concatenate(all_wt_signals)

    # Convert back to original ratio scale
    print("Converting predictions to original ratio scale...")
    pred_ratios_original = (pred_ratios * ratio_stats['std']) + ratio_stats['mean']

    # ratio = log2((1+mut_B)/(1+wt_B)) => (1+mut_B) = (1+wt_B) * 2^pred_ratio
    pred_mut_linear = (1 + wt_signals) * np.exp2(pred_ratios_original) - 1

    # global_mean(targ_mut) / global_mean(targ_wt) = global_ratio
    targ_wt_global_mean = pred_dataset.wt_stats['global_mean']
    targ_mut_global_mean = np.mean(pred_mut_linear)

    scaling_factor = ref_global_ratio * targ_wt_global_mean / targ_mut_global_mean

    #print(f"  Global targ_wt mean: {targ_wt_global_mean:.4f}")
    #print(f"  Global predicted targ_mut mean (before scaling): {targ_mut_global_mean:.4f}")
    #print(f"  Target ratio (ref_mut/ref_wt): {ref_global_ratio:.4f}")
    #print(f"  Required scaling factor: {scaling_factor:.4f}")

    pred_mut_scaled = pred_mut_linear * scaling_factor / 5

    # Save results
    output_file = os.path.join(output_dir, "Targ_species_mut.bw")
    print(f"Saving predictions to {output_file}...")

    chrom_sizes = get_chr_len(genome_path)
    # Convert predictions to list of 1D arrays per region
    pred_mut_list = [pred_mut_scaled[i] for i in range(len(pred_mut_scaled))]

    #print(f"Number of regions: {len(all_regions)}")
    #print(f"Shape of first value array: {pred_mut_list[0].shape}")
    #print(f"Sample values: {pred_mut_list[0][:5]}")  # First 5 values


    # Check if any arrays have wrong shape
    wrong_shape = [i for i, arr in enumerate(pred_mut_list) if arr.shape[0] != num_bins]
    if wrong_shape:
        print(f"Arrays with wrong shape at indices: {wrong_shape[:10]}")  # Show first 10


    save_to_bigwig(
        chrom_sizes=chrom_sizes,
        values_list=pred_mut_list,
        regions_list=all_regions,
        num_bins=num_bins,
        bin_size=bin_size,
        output_path=output_file
    )



class RatioData(Dataset):

    def __init__(self, genome_path, targ_wt_path, data_params, chroms=None):

        self.seq_len = data_params['seq_length']
        self.bin_size = data_params['bin_size']
        self.num_bins = self.seq_len // self.bin_size

        self.genome = Genome(genome_path, chroms)

        if chroms is None:
            chroms = list(self.genome.keys())
        self.chroms = chroms

        # Load WT seq data from BigWig
        self.wt_coverage = {}
        self.wt_coverage = load_bigwig('Target species WT', targ_wt_path, self.chroms)

        # Compute target species WT statistics
        self.wt_stats = genome_wide_stats(self.wt_coverage, "Target species WT")

        # Precompute all valid regions
        self.regions = []
        for chrom in self.chroms:
            if chrom not in self.genome:
                continue
            chrom_len = len(self.genome[chrom])
            if chrom_len < self.seq_len:
                continue
            for start in range(0, chrom_len - self.seq_len, self.seq_len):
                self.regions.append((chrom, start, start + self.seq_len))

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        seq = str(self.genome[chrom][start:end])
        seq_onehot = self._one_hot_encode(seq)

        # Get WT coverage
        wt_coverage = compute_coverage(
            start, end, self.wt_coverage[chrom], self.num_bins, self.bin_size
        )

        # Input DNA sequence
        input_tensor = seq_onehot  # [4, SEQ_LENGTH]

        # Convert regions to strings and integers (not tensors)
        return (torch.tensor(input_tensor),
                torch.tensor(wt_coverage),
                (chrom, start, end))  # Keep as regular Python types

    def _one_hot_encode(self, seq):
        mapping = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25]
        }
        encoded = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq.upper()):
            encoded[:, i] = mapping.get(base, mapping['N'])

        return encoded



def save_to_bigwig(
        chrom_sizes,
        values_list,
        regions_list,
        num_bins,
        bin_size,
        output_path
):
    """
    Save predicted signal to BigWig file.
    """
    bw = pyBigWig.open(output_path, "w")
    bw.addHeader(list(chrom_sizes.items()))

    # Collect all entries
    all_entries = []

    for i, region in enumerate(regions_list):
        chrom, start, end = region

        # Convert tensors to Python types if needed
        if hasattr(chrom, 'item'):
            chrom = str(chrom.item()) if hasattr(chrom, 'dtype') and chrom.dtype == torch.int64 else str(chrom)
        if hasattr(start, 'item'):
            start = int(start.item())
        if hasattr(end, 'item'):
            end = int(end.item())

        vals = values_list[i]

        for bin_idx in range(num_bins):
            bin_start = start + bin_idx * bin_size
            bin_end = bin_start + bin_size
            val = float(vals[bin_idx])

            if not np.isnan(val) and not np.isinf(val):
                all_entries.append((str(chrom), bin_start, bin_end, val))

    # Sort by chromosome and position
    chrom_order = {chrom: idx for idx, chrom in enumerate(chrom_sizes.keys())}
    all_entries.sort(key=lambda x: (chrom_order.get(x[0], 999999), x[1]))

    # Write all entries at once
    if all_entries:
        chroms = [e[0] for e in all_entries]
        starts = [e[1] for e in all_entries]
        ends = [e[2] for e in all_entries]
        values = [e[3] for e in all_entries]
        bw.addEntries(chroms, starts, ends=ends, values=values)

    bw.close()
    print(f"\nOutput Files:")
    print(f"    Write {len(all_entries)} entries to {output_path}")

    print(f"\nCross-species prediction complete!")
    print("──────────────────────────────────────────────────────")