#!/usr/bin/env python3
import numpy as np
import torch
from torch.utils.data import Dataset
import pyBigWig

from .genome import Genome


class SeqDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            fasta_path,
            ref_wt_path,
            ref_mut_path,
            chroms=None,
            **kwargs
    ):

        self.params = kwargs
        self.seq_len = self.params['seq_length']
        self.bin_size = self.params['bin_size']
        self.num_bins = self.seq_len // self.bin_size

        # Load reference genome
        self.genome = Genome(fasta_path, chroms)

        # Load WT and mut BigWig files
        #print("Loading WT BigWig file ...")
        self.wt_coverage = {}
        self.wt_coverage  = load_bigwig(dataset_type, ref_wt_path, chroms)
        #print("Loading mut BigWig file ...")
        self.mut_coverage = {}
        self.mut_coverage = load_bigwig(dataset_type, ref_mut_path, chroms)


        # check chromosomes in all sources (genome, WT, mut)
        genome_chroms = set(self.genome.keys())
        usable_chroms = [
            c for c in chroms
            if c in genome_chroms and c in self.wt_coverage and c in self.mut_coverage
        ]
        if not usable_chroms:
            raise ValueError(
                "Check chromosome IDs in genome, WT and mut tracks!"
            )
        self.chroms = usable_chroms


        # Precompute valid regions
        self.regions = []
        for chrom in self.chroms:
            if chrom not in self.genome:
                continue
            chrom_len = len(self.genome[chrom])
            if chrom_len < self.seq_len:
                continue
            # Create non-overlapping windows
            for start in range(0, chrom_len - self.seq_len, self.seq_len):
                self.regions.append((chrom, start, start + self.seq_len))

        # Compute genome-wide statistics for WT and mut
        self.wt_stats  = genome_wide_stats(self.wt_coverage, "WT")
        self.mut_stats = genome_wide_stats(self.mut_coverage, "mut")

        # Compute ratio statistics using all regions
        self.ratio_stats = self._compute_ratio_stats()

        # Calculate global ratio for scaling factor
        self.global_ratio = self._compute_global_ratio()

    def _compute_ratio_stats(self):
        """
        Compute statistics for log2((1+mut)/(1+wt)) ratio
        using ALL regions
        """
        all_ratios = []

        for idx in range(len(self.regions)):
            chrom, start, end = self.regions[idx]

            # Compute coverage for this region
            wt_cov = compute_coverage(
                start, end, self.wt_coverage[chrom], self.num_bins, self.bin_size
            )
            mut_cov = compute_coverage(
                start, end, self.mut_coverage[chrom], self.num_bins, self.bin_size
            )

            # Calculate ratio: log2((1+mut)/(1+wt))
            ratio = np.log2((1 + mut_cov) / (1 + wt_cov))
            all_ratios.extend(ratio)

        all_ratios = np.array(all_ratios)
        ratio_mean = np.mean(all_ratios)
        ratio_std  = np.std(all_ratios)

        #print(f"Ratio stats - Mean: {ratio_mean:.4f}, Std: {ratio_std:.4f}")
        #print(f"Number of ratio values: {len(all_ratios)}")

        return {
            'mean': ratio_mean,
            'std': ratio_std
        }

    def _compute_global_ratio(self):
        """
        Compute global mut_A/wt_A ratio
        for cross-species scaling
        """
        wt_global_mean = self.wt_stats['global_mean']
        mut_global_mean = self.mut_stats['global_mean']

        if wt_global_mean > 0:
            global_ratio = mut_global_mean / wt_global_mean
        else:
            global_ratio = 1.0

        #print(f"Global ratio (mut_A/wt_A): {global_ratio:.4f}")
        return global_ratio

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

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        seq = str(self.genome[chrom][start:end])
        seq_onehot = self._one_hot_encode(seq)

        # Get WT and mut coverage
        wt_coverage = compute_coverage(
            start, end, self.wt_coverage[chrom], self.num_bins, self.bin_size
        )
        mut_coverage = compute_coverage(
            start, end, self.mut_coverage[chrom], self.num_bins, self.bin_size
        )

        # log2((1+mut)/(1+wt)) ratio, normalized
        target_ratio = np.log2((1 + mut_coverage) / (1 + wt_coverage))
        target_ratio = (target_ratio - self.ratio_stats['mean']) / self.ratio_stats['std']

        # Handle NaN/Inf values
        target_ratio = np.nan_to_num(target_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # Create input tensor: DNA sequence only
        input_tensor = seq_onehot  # [4, seq_length]

        return (torch.tensor(input_tensor),
                torch.tensor(target_ratio),
                (chrom, start, end))



def load_bigwig(data_type, bigwig_path, chroms):
    """
    Load coverage data from BigWig file.
    Args:
        bigwig_path: Path to BigWig file
        chroms: List of chromosomes to load
    Returns:
        Dictionary with chr as key and list of (beg, end, val) tuples as value
    """
    coverage_data = {chrom: [] for chrom in chroms}

    bw = pyBigWig.open(bigwig_path)

    # Get available chromosomes
    bw_chroms = {chrom: size for chrom, size in bw.chroms().items()}

    for chrom in chroms:
        if chrom not in bw_chroms:
            print(f"Warning: Chromosome {chrom} not found in BigWig file, skipping")
            continue

        #chrom_length = bw_chroms[chrom]
        try:
            # Get all intervals for this chromosome
            intervals = bw.intervals(chrom)
            if intervals is None:
                continue

            # transform to non-negative
            for start, end, value in intervals:
                if value < 0:
                    value = 0.0
                coverage_data[chrom].append((start, end, float(value)))

        except Exception as e:
            print(f"Error processing chromosome {chrom}: {e}")
            continue

    bw.close()

    total_intervals = sum(len(intervals) for intervals in coverage_data.values())
    #print(f"{data_type} data loaded {total_intervals} intervals.")

    return coverage_data


def genome_wide_stats(coverage_data, label):
    all_values = []
    for chrom in coverage_data:
        for _, _, value in coverage_data[chrom]:
            if value >= 0:  # Only include non-negative values
                all_values.append(value)

    all_values = np.array(all_values, dtype=np.float32)

    if len(all_values) == 0:
        print(f"WARNING: No {label} values found, using defaults")
        return {
            'mean': 0.0,
            'std': 1.0,
            'global_mean': 0.0,
            'global_std': 1.0
        }

    # Calculate statistics
    global_mean = np.mean(all_values)
    global_std = np.std(all_values)

    return {
        'global_mean': global_mean,
        'global_std': global_std
    }



def compute_coverage(start, end, coverage_list, num_bins, bin_size):
    coverage = np.zeros(num_bins, dtype=np.float32)

    for r_start, r_end, value in coverage_list:
        if r_end <= start or r_start >= end:
            continue
        bin_start_idx = max(0, (r_start - start) // bin_size)
        bin_end_idx = min(num_bins, (r_end - start) // bin_size + 1)
        for bin_idx in range(bin_start_idx, bin_end_idx):
            bin_start = start + bin_idx * bin_size
            bin_end = bin_start + bin_size
            overlap_start = max(r_start, bin_start)
            overlap_end = min(r_end, bin_end)
            overlap = max(0, overlap_end - overlap_start)
            coverage[bin_idx] += value * (overlap / bin_size)

    return coverage
