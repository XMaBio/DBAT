#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from deepPeak.data.genome import Genome
from deepPeak.bin import process_for_bigwig
from deepPeak.data.norm_save_data import write_prediction
from deepPeak.model.model import PerformerUNet


def prediction(
    genome_path=None,
    model_path=None,
    stats_path=None,
    output_dir=None,
    data_params=None,
    train_params=None
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load coverage stats
    coverage_stats = np.load(stats_path)
    num_suppressive = int(coverage_stats.get('num_suppressive_channels', 0))

    # Load model
    model = PerformerUNet(num_suppressive,data_params['bin_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Create dataset and loader
    dataset = PredictionDataset(
        genome_path,
        data_params['seq_length'],
        num_suppressive_channels=num_suppressive
    )
    loader  = DataLoader(
        dataset, batch_size=train_params['batch_size'], shuffle=False
    )

    # run predictions
    all_preds, all_targets, chrom_data = run_predictions(
        model, loader, coverage_stats, device, train_params['use_amp'], data_params
    )


    # Write predictions in BigWig format
    bigwig_path = os.path.join(output_dir, "predictions.bw")
    write_prediction(bigwig_path, chrom_data)

    print(f"\nOutput Files:")
    print(f"  Predictions: {bigwig_path}")
    print(f"\nPrediction complete!")
    print("──────────────────────────────────────────────────────")



def run_predictions(
        model,
        loader,
        coverage_stats,
        device,
        use_amp,
        data_params
):
    model.eval()

    bar = tqdm(loader, desc="Predicting", unit="seq")
    all_preds = []
    all_targets = []
    chrom_data = {}

    with torch.no_grad():
        for batch in bar:
            seq, coords_batch = batch
            seq = seq.to(device)

            if device.type == 'cuda' and use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(seq).cpu().numpy()
            else:
                preds = model(seq).cpu().numpy()

            all_preds.append(preds)

            # Process predictions for BigWig output
            process_for_bigwig(
                preds, coords_batch, chrom_data, coverage_stats, data_params
            )

    return all_preds, all_targets, chrom_data



class PredictionDataset(Dataset):
    def __init__(self, genome_path, seq_length, num_suppressive_channels=0):
        self.genome = Genome(genome_path)
        self.seq_length = seq_length
        self.num_suppressive = num_suppressive_channels
        self.regions = []

        for chrom in self.genome.keys():
            chrom_len = len(self.genome[chrom])
            if chrom_len < self.seq_length:
                continue
            for start in range(0, chrom_len - self.seq_length, self.seq_length):
                self.regions.append((chrom, start, start + self.seq_length))

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        seq = str(self.genome[chrom][start:end])
        dna = self._one_hot_encode(seq)

        # Add suppressive channels as zeros (not provided in prediction)
        if self.num_suppressive > 0:
            zeros = np.zeros((self.num_suppressive, dna.shape[1]), dtype=np.float32)
            x = np.concatenate([dna, zeros], axis=0)
        else:
            x = dna

        return torch.tensor(x), (chrom, start, end)

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
