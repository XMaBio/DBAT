import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict

from deepLoci.data.data import MultiModalDataset
from deepLoci.model.model import CNNFeaturePredictor


def train_predict(
        data_types,
        output_dir,
        patience,
        ini_lr,
        min_lr,
        workers
):
    # Convert string parameters to appropriate types
    ini_lr = float(ini_lr)
    min_lr = float(min_lr)
    patience = int(patience)
    workers = int(workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, num_classes = prepare_train(data_types, output_dir)

    model = CNNFeaturePredictor(
        S=train_ds.S,
        M=train_ds.M,
        P=train_ds.P,
        num_classes=num_classes
    ).to(device)

    # Enhanced optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=ini_lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=patience,
        factor=0.5,
        min_lr=min_lr,
    )


    # label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Data loaders with persistent workers
    train_loader = DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=workers,
        persistent_workers=workers > 0
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=32, 
        pin_memory=True,
        num_workers=workers,
        persistent_workers=workers > 0
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=32,
        num_workers=workers if workers > 0 else 0
    )

    best_accuracy = 0.0
    no_improve = 0
    early_stop = False

    for epoch in range(20):
        if early_stop: break

        model.train()
        epoch_loss = 0.0
        correct = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = (
                inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)  # Update learning rate


        # Early stopping check
        if val_acc > best_accuracy + 0.005:  # Require meaningful improvement
            best_accuracy = val_acc
            no_improve = 0
            model_path = os.path.join(output_dir, 'Loci_model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True


        # Metrics
        train_loss = epoch_loss / len(train_ds)
        train_acc = correct / len(train_ds)

        print(f"Epoch {epoch+1:03d} | "
              f"Train loss: {train_loss:.4f} (Train acc: {train_acc:.4f}) | "
              f"Val loss: {val_loss:.4f} (Val acc: {val_acc:.4f})  "
        )

    # Final evaluation
    model_path = os.path.join(output_dir, 'Loci_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Enhanced prediction aggregation
    id_confidences = defaultdict(list)
    with torch.no_grad():
        for inputs, _, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            for id, prob in zip(ids, probs):
                id_confidences[id].append(prob.cpu().numpy())

    # Aggregate predictions by mean confidence
    predictions = []
    for id, probs in id_confidences.items():
        avg_prob = np.mean(probs, axis=0)
        pred_class = np.argmax(avg_prob)
        confidence = avg_prob[pred_class]
        predictions.append((
            id,
            train_ds.label_encoder.inverse_transform([pred_class])[0],
            confidence
        ))

    # Save results
    res_path = os.path.join(output_dir, 'predictions.csv')
    pd.DataFrame({
        'ID': [id_ for id_, _, _ in predictions],
        'Predicted_Feature': [feat for _, feat, _ in predictions],
        'Confidence': [conf for _, _, conf in predictions]
    }).to_csv(res_path, index=False)

    # Remove temporary *.pkl files
    for pkl_file in glob.glob(os.path.join(output_dir, "*.pkl")):
        os.remove(pkl_file)
    
    print(f"\nOutput Files:")
    print(f"  Predictions: {res_path}")
    print(f"\nAnalysis complete!")
    print("=" * 70)



def prepare_train(data_types, output_dir):
    # Read train data
    train_nonN_file = os.path.join(output_dir, 'nonN_train_data.csv')
    train_df = pd.read_csv(train_nonN_file)

    # Create sample mapping
    sample_mapping = {
        v: i+1 for i, v in enumerate(sorted(train_df['sample'].unique()))
    }
    train_df['sample'] = train_df['sample'].map(sample_mapping).astype(int)

    # Get unique IDs with their corresponding features
    id_features = train_df.groupby('ID')['feature'].first()
    
    # Count feature frequencies and filter out rare classes
    feature_counts = id_features.value_counts()
    valid_features = feature_counts[feature_counts >= 2].index
    valid_ids = id_features[id_features.isin(valid_features)].index
    
    print(f"Original IDs: {len(id_features)}, After filtering rare classes: {len(valid_ids)}")
    print(f"Original classes: {len(feature_counts)}, After filtering: {len(valid_features)}")
    
    # Stratified split by feature labels on filtered data
    train_ids, val_ids = train_test_split(
        valid_ids, 
        test_size=0.2, 
        random_state=42,
        stratify=id_features[valid_ids]  # Use filtered features for stratification
    )

    # Filter the dataframe to only include valid IDs
    filtered_df = train_df[train_df['ID'].isin(valid_ids)]

    # Create datasets
    train_ds = MultiModalDataset(
        data_types, 
        output_dir, 
        filtered_df[filtered_df['ID'].isin(train_ids)], 
        is_train=True
    )

    val_ds = MultiModalDataset(
        data_types,
        output_dir, 
        filtered_df[filtered_df['ID'].isin(val_ids)],
        is_train=False
    )

    # Prepare test data (existing code)
    test_N_file = os.path.join(output_dir, 'N_test_data.csv')
    test_df = pd.read_csv(test_N_file)

    if 'feature' in test_df.columns:
        test_df = test_df.drop(columns=['feature'])
    test_df['sample'] = test_df['sample'].map(sample_mapping).fillna(-1).astype(int)

    test_ds = MultiModalDataset(
        data_types,
        output_dir,
        test_df,
        is_train=False
    )

    return train_ds, val_ds, test_ds, len(train_ds.label_encoder.classes_)




def validate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct    = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


if __name__ == '__main__':
    train_predict()
