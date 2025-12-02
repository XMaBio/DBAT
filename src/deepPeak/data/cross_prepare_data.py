#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys

from .genome import get_chr_len
from .cross_seq_data import SeqDataset

def prepare_datasets(
        genome_path,
        ref_wt_path,
        ref_mut_path,
        data_params,
        train_params
):
    #seq_length = data_params['seq_length']
    #bin_size = data_params['bin_size']
    #batch_size = train_params['batch_size']
    #epochs = train_params['epochs']
    #use_amp = train_params['use_amp']
    #test_prop = train_params['test_prop']
    #val_prop = train_params['val_prop']

    # Get chromosome lengths and filter
    chrom_length = get_chr_len(genome_path)
    all_chroms = list(chrom_length.keys())

    if len(all_chroms) < 3:
        sys.exit("ERROR: At least 3 chromosomes required.")
        #return prepare_region_data(
        #    all_chroms,
        #    genome_path,
        #    ref_wt_path,
        #    ref_mut_path,
        #    data_params,
        #    train_params
        #)
    else:
        return prepare_chrom_data(
            all_chroms,
            genome_path,
            ref_wt_path,
            ref_mut_path,
            data_params,
            train_params
        )



def prepare_region_data(
        all_chroms,
        genome_path,
        ref_wt_path,
        ref_mut_path,
        data_params,
        train_params
):
    # Create a single dataset with all chromosomes
    full_dataset = SeqDataset(
        'Train',
        genome_path,
        ref_wt_path,
        ref_mut_path,
        chroms=all_chroms,
        **data_params
    )

    all_indices = list(range(len(full_dataset)))

    # Split regions
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=train_params['test_prop'],
        random_state=42
    )

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=train_params['val_prop'] / (1 - train_params['test_prop']),
        random_state=42
    )

    # Create train, validation and test datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, all_chroms


def prepare_chrom_data(
        all_chroms,
        genome_path,
        ref_wt_path,
        ref_mut_path,
        data_params,
        train_params
):
    # Chromosome-based splitting
    train_val_chroms, test_chroms = train_test_split(
        all_chroms,
        test_size=train_params['test_prop'],
        random_state=42
    )

    train_chroms, val_chroms = train_test_split(
        train_val_chroms,
        test_size=train_params['val_prop'] / (1 - train_params['test_prop']),
        random_state=42
    )
    print(f"Training chromosomes: {', '.join(train_chroms[:])}.")
    print(f"Validation chromosomes: {', '.join(val_chroms[:])}.")
    print(f"Test chromosomes: {', '.join(test_chroms[:])}.")
    print()

    # Create datasets
    train_dataset = SeqDataset(
        'Train',
        genome_path,
        ref_wt_path,
        ref_mut_path,
        chroms=train_chroms,
        **data_params
    )
    val_dataset = SeqDataset(
        'Validation',
        genome_path,
        ref_wt_path,
        ref_mut_path,
        chroms=val_chroms,
        **data_params
    )
    test_dataset = SeqDataset(
        'Test',
        genome_path,
        ref_wt_path,
        ref_mut_path,
        chroms=test_chroms,
        **data_params
    )

    return train_dataset, val_dataset, test_dataset, all_chroms


def create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size
):

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    return train_loader, val_loader, test_loader
