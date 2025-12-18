import numpy as np
import pandas as pd
import joblib
import warnings
import os
import gc
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

warnings.filterwarnings('ignore')



def process_data(input_file, output_dir):

    print("\nProcessing data ...")
    dfs = collapse_replicate(input_file)

    data_types = list(dfs.keys())

    # Separate N and non-N features
    n_features = {}
    non_n_features = {}
    for dt, df in dfs.items():
        n_mask = df['feature'] == 'N'
        n_features[dt] = df[n_mask]
        non_n_features[dt] = df[~n_mask]

    # Find common IDs for non-N features (training)
    common_non_n = set.intersection(*[
        set(zip(df['ID'], df['feature'], df['sample']))
        for df in non_n_features.values() if not df.empty
    ])

    # Find common IDs for N features (prediction)
    common_n = set.intersection(*[
        set(zip(df['ID'], df['feature'], df['sample']))
        for df in n_features.values() if not df.empty
    ])
    print(f"Non-N common IDs: {len(common_non_n)}, N common IDs: {len(common_n)}")

    # Align non-N data
    aligned_non_n = {}
    for dt in non_n_features:
        df = non_n_features[dt]
        mask = df[['ID', 'feature', 'sample']].apply(tuple, axis=1).isin(common_non_n)
        aligned_non_n[dt] = df[mask].sort_values(['ID', 'feature', 'sample'])
    train_non_n_dfs = []
    for dt, df in aligned_non_n.items():
        df = df.copy()
        df['data'] = dt
        train_non_n_dfs.append(df)

    # save training (nonN) data to output dir
    train_nonN_file = os.path.join(output_dir, "nonN_train_data.csv")
    pd.concat(train_non_n_dfs).to_csv(train_nonN_file, index=False)
    print(f"Saved non-N training data to {train_nonN_file}")


    # Encode feature labels
    le_feature = LabelEncoder()
    all_features = pd.concat([df['feature'] for df in aligned_non_n.values()])
    le_feature.fit(all_features)

    # save fitted feature labels to output dir
    label_file = os.path.join(output_dir, 'feature_label_encoder.pkl')
    joblib.dump(le_feature, label_file)


    # Align N data using N common IDs
    aligned_n = {}
    for dt in n_features:
        df = n_features[dt]
        mask = df[['ID', 'feature', 'sample']].apply(tuple, axis=1).isin(common_n)
        aligned_n[dt] = df[mask].sort_values(['ID', 'feature', 'sample'])
    test_n_dfs = []
    for dt, df in aligned_n.items():
        df = df.copy()
        df['data'] = dt
        test_n_dfs.append(df)

    # save testing (N) data to output dir
    test_N_file = os.path.join(output_dir, "N_test_data.csv")
    pd.concat(test_n_dfs).to_csv(test_N_file, index=False)
    print(f"Saved N test data to {test_N_file}")

    return data_types



def collapse_replicate(input_file):

    # get sequencing data types 
    data_types = get_data_types(input_file)

    with open(input_file, 'r') as f:
        header = f.readline().strip().split(',')
    pos_cols = [col for col in header if col.startswith('pos')]
    num_pos = len(pos_cols)

    sum_dicts = {dt: defaultdict(lambda: np.zeros(num_pos, dtype=np.float32))
                 for dt in data_types}
    count_dicts = {dt: defaultdict(int) for dt in data_types}

    chunk_iter = pd.read_csv(
        input_file,
        dtype={'ID': 'str',
               'feature': 'str',
               'data': 'str',
               'sample': 'str'},
        chunksize=5000
    )

    chunk_cnt = 0
    for chunk in chunk_iter:
        chunk_cnt += 1
        if chunk_cnt % 50 == 0:
            print(f"Total rows scanned: {chunk_cnt * len(chunk) :,} ")

        chunk[pos_cols] = chunk[pos_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        chunk[pos_cols] = chunk[pos_cols].astype('float32')

        for data_type in data_types:
            sub_chunk = chunk[chunk['data'] == data_type]
            if sub_chunk.empty:
                continue

            grouped = sub_chunk.groupby(['ID', 'feature', 'sample'])
            for (id_, feature_, sample_), subgroup in grouped:
                key = (id_, feature_, sample_)
                sum_vals = subgroup[pos_cols].sum().values.astype(np.float32)
                sum_dicts[data_type][key] += sum_vals
                count_dicts[data_type][key] += len(subgroup)

        # memory cleanup
        del grouped, chunk
        if chunk_cnt % 10 == 0:
            gc.collect()

    # Build combined dataFrame
    combined_dfs = {}
    for data_type in data_types:
        sum_dict = sum_dicts[data_type]
        count_dict = count_dicts[data_type]

        rows = []
        for key in sum_dict:
            id_, feature, sample = key
            mean_vals = sum_dict[key] / count_dict[key]
            mean_vals = np.round(mean_vals, 2)
            rows.append([id_, feature, sample] + mean_vals.tolist())

        if rows:
            df = pd.DataFrame(
                data=rows,
                columns=['ID', 'feature', 'sample'] + pos_cols
            )
        else:
            df = pd.DataFrame()
        combined_dfs[data_type] = df

    return combined_dfs


    
def get_data_types(input_file):

    uni_types = set()
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                fields = line.strip().split(',')
                if len(fields) > 2:
                    uni_types.add(fields[2])
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {input_file}")
    except Exception as e:
        raise RuntimeError(f"Reading file error: {e}")

    return list(uni_types)



class MultiModalDataset(Dataset):
    def __init__(self, data_types, output_dir, dataframe, is_train=True):
        self.data_types = data_types
        self.output_dir = output_dir
        self.is_train = is_train
        self.pos_cols = self._detect_position_columns(dataframe)
        self.P = len(self.pos_cols)
        self.M = len(data_types)
        self.samples = self._create_samples(dataframe)
        self.S = self._validate_sample_structure()

        if self.is_train:
            self._fit_scalers()
            self._fit_label_encoder(dataframe)
        else:
            self._load_scalers()
            self._load_label_encoder()
            self._process_labels(dataframe)

    def _fit_scalers(self):
        """Fit standardization scalers on training data"""
        all_data = np.concatenate([s['data'] for s in self.samples.values()])
        self.scaler = StandardScaler()
        self.scaler.fit(all_data.reshape(-1, self.P))

        scalar_file = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scalar_file)

    def _fit_label_encoder(self, df):
        """Encode target labels using sklearn's LabelEncoder"""
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(
            [s['feature'] for s in self.samples.values()]
        )
        label_file = os.path.join(self.output_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_file)

    def _load_scalers(self):
        scalar_file = os.path.join(self.output_dir, 'scaler.pkl')
        self.scaler = joblib.load(scalar_file)

    def _load_label_encoder(self):
        label_file = os.path.join(self.output_dir, 'label_encoder.pkl')
        self.label_encoder = joblib.load(label_file)


    ### Data Processing Methods

    def _process_labels(self, dataframe):
        """Handle label processing for test/validation data"""
        if 'feature' in dataframe.columns:
            features = [s['feature'] for s in self.samples.values()]
            self.labels = self.label_encoder.transform(features)
        else:
            self.labels = []


    ### Validation Methods

    def _detect_position_columns(self, df):
        """Automatically detect position columns (pos1, pos2, ...)"""
        pos_columns = []
        for col in df.columns:
            if re.match(r'^pos\d+$', col):
                pos_columns.append(col)

        if not pos_columns:
            raise ValueError(
                "Position columns not found. Expected format: pos1, pos2, ..."
            )

        # Sort columns numerically (pos1 before pos10)
        return sorted(pos_columns, key=lambda x: int(x[3:]))

    def _validate_sample_structure(self):
        """Ensure consistent dimensions across all samples"""
        sample_shapes = [s['data'].shape for s in self.samples.values()]

        # Check modality and position consistency
        unique_shapes = set((s[1], s[2]) for s in sample_shapes)
        if len(unique_shapes) > 1:
            raise ValueError(
                f"Inconsistent sample dimensions detected: {unique_shapes}"
            )

        # Check sample count consistency
        sample_counts = [s[0] for s in sample_shapes]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                f"Inconsistent sample counts: {min(sample_counts)}-{max(sample_counts)}"
            )

        return sample_counts[0]  # Consistent S value


    ### Sample Creation

    def _create_samples(self, df):
        """Create 3D sample tensors with comprehensive validation"""
        samples = {}
        has_features = 'feature' in df.columns

        for identifier, group in df.groupby('ID'):
            # Validate and extract feature consistently
            if has_features:
                # Drop NaNs and get unique non-null features
                unique_features = group['feature'].dropna().unique()
                if len(unique_features) == 0:
                    current_feature = None
                elif len(unique_features) == 1:
                    current_feature = unique_features[0]
                else:
                    raise ValueError(f"Multiple distinct features for ID {identifier}: {unique_features}")
            else:
                current_feature = None

            # Map actual sample IDs to 0-based indices (robust to non-consecutive or string-like sample IDs)
            unique_samples = sorted(group['sample'].unique())
            sample_to_index = {s: i for i, s in enumerate(unique_samples)}
            S_local = len(unique_samples)

            # Initialize tensor: (S_local, M, P)
            tensor = np.zeros((S_local, self.M, self.P), dtype=np.float32)

            # Fill tensor
            for _, row in group.iterrows():
                s_idx = sample_to_index[row['sample']]
                try:
                    m_idx = self.data_types.index(row['data'])
                except ValueError as e:
                    raise ValueError(f"Unknown modality '{row['data']}' in ID {identifier}") from e
                tensor[s_idx, m_idx] = row[self.pos_cols].values

            samples[identifier] = {
                'data': tensor,
                'feature': current_feature
            }

        return samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        identifier = list(self.samples.keys())[idx]
        sample_data = self.samples[identifier]['data']

        # Apply standardization
        normalized = self.scaler.transform(
            sample_data.reshape(-1, self.P)
        ).reshape(self.S, self.M, self.P)
        
        # Check self.labels non-NULL
        if self.is_train or (len(self.labels) > 0):
            label = self.labels[idx]
        else:
            label = -1

        return (
            torch.FloatTensor(normalized),
            torch.tensor(label, dtype=torch.long),
            identifier
        )


