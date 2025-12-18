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
        bigwig_path,
        chroms,
        suppression_paths=None,
        **kwargs
    ):
        self.genome = Genome(fasta_path, chroms)
        self.chroms = chroms
        self.bigwig_path = bigwig_path
        try:
            self.bigwig = pyBigWig.open(bigwig_path)
            # Verify BigWig is readable
            _ = self.bigwig.chroms()
        except Exception as e:
            print(f"Error opening bigWig file {bigwig_path}: {e}")
            raise

        self.suppression_paths = suppression_paths if suppression_paths else []
        self.suppression_bigwig_paths = self.suppression_paths
        self.suppression_bigwigs = []
        for path in self.suppression_paths:
            try:
                bw = pyBigWig.open(path)
                _ = bw.chroms()  # Test reading
                self.suppression_bigwigs.append(bw)

            except Exception as e:
                print(f"Error opening suppression bigWig {path}: {e}")
                self.suppression_bigwigs.append(None)

        self.params = kwargs
        self.seq_len = self.params['seq_length']
        self.bin_size = self.params['bin_size']
        self.num_bins = self.seq_len // self.bin_size

        # Precompute valid regions across specified chromosomes
        self.regions = []
        for chrom in self.chroms:
            chrom_len = len(self.genome[chrom])

            # Create non-overlapping windows
            for start in range(0, chrom_len - self.seq_len, self.seq_len):
                self.regions.append((chrom, start, start + self.seq_len))

        # Precompute coverage statistics for standardization
        self.coverage_stats = self._compute_coverage_stats()

        # Precompute suppressive data statistics
        self.suppression_stats = []
        for idx in range(len(self.suppression_paths)):
            self.suppression_stats.append(self._compute_suppression_stats(idx))

        print(f"{dataset_type} dataset initialized with {len(self.regions)} regions")

    def _compute_coverage_stats(self):
        # Use the helper function for bigWig
        stats = compute_genomewide_stats(
            self.bigwig_path,
            "coverage",
            self.chroms
        )

        return {
            'mean': stats['mean'],
            'std': stats['std'],
            'transformation': 'log2'
        }

    def _compute_suppression_stats(self, suppression_idx):
        """Compute genome-wide statistics for suppressive dataset"""
        stats = compute_genomewide_stats(
            self.suppression_bigwig_paths[suppression_idx],
            f"suppression_{suppression_idx + 1}",
            self.chroms
        )

        return {
            'mean': stats['mean'],
            'std': stats['std'],
            'transformation': 'log2'
        }

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        seq = str(self.genome[chrom][start:end])
        seq_onehot = one_hot_encode(seq)

        # Create input tensor: DNA sequence + suppression signals
        num_inputs = 4 + len(self.suppression_paths)
        input_tensor = np.zeros((num_inputs, self.seq_len), dtype=np.float32)
        input_tensor[:4] = seq_onehot  # First 4 channels are DNA sequence

        # Add suppression signals as negative values
        for idx in range(len(self.suppression_paths)):
            signal = self._get_suppression_signal(
                chrom, start, end, idx
            )
            input_tensor[4 + idx] = signal

        # Get coverage as target from bigWig
        coverage = compute_coverage_bigwig(
            self.bigwig, chrom, start, end, self.num_bins, self.bin_size
        )

        # Apply log2(1+x) transformation and standardize
        coverage = np.log2(1 + coverage)
        coverage = (coverage - self.coverage_stats['mean']) / self.coverage_stats['std']

        if np.isnan(coverage).any() or np.isinf(coverage).any():
            coverage = np.nan_to_num(coverage, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(input_tensor), torch.tensor(coverage), (chrom, start, end)

    def _get_suppression_signal(self, chrom, start, end, suppression_idx):
        """Get suppression signal from bigWig"""
        if (suppression_idx >= len(self.suppression_bigwigs) or
                self.suppression_bigwigs[suppression_idx] is None):
            return np.zeros(self.seq_len, dtype=np.float32)

        try:
            values = bigwig_values(
                self.suppression_bigwigs[suppression_idx], chrom, start, end
            )
            if values is None:
                return np.zeros(self.seq_len, dtype=np.float32)

            values = np.nan_to_num(values, nan=0.0)

            # Ensure correct length
            if len(values) < self.seq_len:
                values = np.pad(
                    values, (0, self.seq_len - len(values)), constant_values=0
                )
            elif len(values) > self.seq_len:
                values = values[:self.seq_len]

            # Store as negative value for suppression
            signal = -values

            # Apply transformation and standardization
            abs_signal = np.abs(signal)
            transformed = np.log2(1 + abs_signal)
            standardized = (transformed - self.suppression_stats[suppression_idx]['mean']) / \
                           self.suppression_stats[suppression_idx]['std']

            # Restore original sign
            standardized = np.sign(signal) * np.abs(standardized)
            return standardized

        except Exception:
            return np.zeros(self.seq_len, dtype=np.float32)

    def __del__(self):
        """Close bigWig files when dataset is destroyed"""
        if hasattr(self, 'bigwig'):
            try:
                self.bigwig.close()
            except:
                pass
        for bw in getattr(self, 'suppression_bigwigs', []):
            if bw is not None:
                try:
                    bw.close()
                except:
                    pass



def one_hot_encode(seq):
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



def compute_genomewide_stats(bw_path, label, chroms):
    all_values = []
    try:
        with pyBigWig.open(bw_path) as bw:
            bw_chroms = list(bw.chroms().keys()) # Get all chromosomes

            bw_chroms = [chr for chr in bw_chroms if chr in chroms]

            for chr in bw_chroms:
                chrom_length = bw.chroms()[chr]

                # Sample regions to estimate statistics
                num_samples = min(1000, chrom_length // 1000)
                sample_starts = np.random.randint(
                    0, max(1, chrom_length - 1000), num_samples
                )

                for start in sample_starts:
                    end = min(start + 1000, chrom_length)
                    try:
                        values = bigwig_values(bw, chr, start, end)
                        if values is not None:
                            values = np.nan_to_num(values, nan=0.0)
                            values = values[values >= 0] # non-negative
                            all_values.extend(values)
                    except:
                        continue

    except Exception as e:
        print(f"Error processing bigWig file {bw_path}: {e}")
        return {
            'mean': 0.0,
            'std': 1.0,
            'global_mean': 0.0,
            'global_std': 1.0
        }

    if len(all_values) == 0:
        print(f"WARNING: No {label} values found, using defaults")
        return {
            'mean': 0.0,
            'std': 1.0,
            'global_mean': 0.0,
            'global_std': 1.0
        }

    all_values = np.array(all_values, dtype=np.float32)

    # Apply log2(1+x) transformation
    log_values = np.log2(1 + all_values)

    # Calculate statistics
    global_mean = np.mean(all_values)
    global_std  = np.std(all_values)
    log_mean    = np.mean(log_values)
    log_std     = np.std(log_values)

    return {
        'mean': log_mean,
        'std': log_std,
        'global_mean': global_mean,
        'global_std': global_std
    }



def compute_coverage_bigwig(bw_file, chrom, start, end, num_bins, bin_size):
    try:
        values = bigwig_values(bw_file, chrom, start, end)
        if values is None:
            return np.zeros(num_bins, dtype=np.float32)

        # Replace NaN with 0
        values = np.nan_to_num(values, nan=0.0)

        # Ensure the right length
        expected_length = end - start
        if len(values) < expected_length:
            values = np.pad(values, (0, expected_length - len(values)), constant_values=0)
        elif len(values) > expected_length:
            values = values[:expected_length]

        # Reshape to bins and take mean
        if len(values) >= num_bins * bin_size:
            coverage = values.reshape(num_bins, bin_size).mean(axis=1)
        else:
            # Handle case where we don't have enough values
            coverage = np.zeros(num_bins, dtype=np.float32)
            min_bins = min(num_bins, len(values) // bin_size)
            if min_bins > 0:
                coverage[:min_bins] = values[:min_bins * bin_size].reshape(min_bins, bin_size).mean(axis=1)

        return coverage.astype(np.float32)

    except Exception:
        return np.zeros(num_bins, dtype=np.float32)



def bigwig_values(bw_file, chrom, start, end):
    try:
        # Check if chromosome exists in BigWig file
        if chrom not in bw_file.chroms():
            return None

        chrom_length = bw_file.chroms()[chrom]

        # Adjust coordinates to be within chromosome bounds
        start = max(0, start)
        end   = min(chrom_length, end)

        if start >= end:
            return None

        values = bw_file.values(chrom, start, end)
        return values

    except RuntimeError:
        return None
