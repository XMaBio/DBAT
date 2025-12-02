#!/usr/bin/env python3

class Genome:
    def __init__(self, fasta_path, chroms=None):
        self.sequences = {}
        current_seq = []
        current_chrom = None

        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_chrom is not None:
                        seq = ''.join(current_seq).upper()
                        if chroms is None or current_chrom in chroms:
                            self.sequences[current_chrom] = seq
                        current_seq = []
                    current_chrom = line[1:].split()[0]

                else:
                    current_seq.append(line)

        if current_chrom is not None and current_seq:
            seq = ''.join(current_seq).upper()
            if chroms is None or current_chrom in chroms:
                self.sequences[current_chrom] = seq

    def __contains__(self, chrom):
        return chrom in self.sequences

    def __getitem__(self, chrom):
        return self.sequences[chrom]

    def keys(self):
        return self.sequences.keys()
    
    
def get_chr_len(fasta_path):
    chrom_lengths = {}
    current_chrom = None

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_chrom is not None:
                    chrom_lengths[current_chrom] = current_len
                current_chrom = line[1:].split()[0]
                current_len = 0

            elif current_chrom is not None and line:
                current_len += len(line)

        if current_chrom is not None:
            chrom_lengths[current_chrom] = current_len

    return chrom_lengths