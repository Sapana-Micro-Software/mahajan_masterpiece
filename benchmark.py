"""
Benchmark script comparing multiple ECG classification models:
1. Feedforward Neural Network (Lloyd et al., 2001)
2. Transformer-based Model (Ikram et al., 2025)
3. Three-Stage Hierarchical Transformer (Tang et al., 2025)
4. 1D Convolutional Neural Network (CNN)
5. Long Short-Term Memory (LSTM)
6. Hopfield Network (ETASR, 2013)
7. Variational Autoencoder (VAE) - FactorECG (van de Leur et al., 2022)
8. Liquid Time-Constant Network (LTC) - Hasani et al. (2020)
"""

import numpy as np                                                      # NumPy for array operations
import torch                                                            # PyTorch
from typing import Dict, Tuple                                          # Type hints
import time                                                             # Time measurement
import json                                                             # JSON for saving results
from neural_network import NeuralNetwork, create_sample_ecg_data      # Feedforward NN
from transformer_ecg import (                                          # Transformer model
    TransformerECGClassifier, ECGDataset, train_transformer,
    evaluate_model, create_synthetic_ecg_data
)
from three_stage_former import (                                       # Three-Stage Former model
    ThreeStageFormer, train_three_stage_former, evaluate_model as evaluate_3stage
)
from cnn_lstm_ecg import (                                             # CNN and LSTM models
    CNN1DECGClassifier, LSTMECGClassifier, train_model, evaluate_model as evaluate_cnn_lstm
)
from hopfield_ecg import (                                             # Hopfield Network model
    HopfieldECGClassifier, train_hopfield, evaluate_model as evaluate_hopfield
)
from vae_ecg import (                                                  # VAE model
    VAEEcgClassifier, train_vae, evaluate_model as evaluate_vae
)
from ltc_ecg import (                                                  # LTC model
    LTCEcgClassifier, train_ltc, evaluate_model as evaluate_ltc, ECGDataset as LTCECGDataset
)
from torch.utils.data import DataLoader                                 # Data loading
import matplotlib.pyplot as plt                                         # Plotting
from sklearn.metrics import (                                           # Evaluation metrics
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import seaborn as sns                                                   # Better plots


def prepare_data_for_feedforward(
    signals: np.ndarray,
    labels: np.ndarray,
    feature_extraction: str = 'statistical'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from ECG signals for feedforward neural network.
    
    Parameters:
    -----------
    signals : np.ndarray
        Raw ECG signals of shape (n_samples, seq_len)
    labels : np.ndarray
        Class labels
    feature_extraction : str
        Feature extraction method
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (features, labels)
    """
    n_samples = signals.shape[0]
    features_list = []
    
    if feature_extraction == 'statistical':
        # Extract statistical features
        for i in range(n_samples):
            signal = signals[i]
            feat = [
                np.mean(signal),                                        # Mean
                np.std(signal),                                         # Standard deviation
                np.median(signal),                                      # Median
                np.min(signal),                                         # Minimum
                np.max(signal),                                         # Maximum
                np.percentile(signal, 25),                              # 25th percentile
                np.percentile(signal, 75),                              # 75th percentile
                np.var(signal),                                         # Variance
                np.mean(np.abs(np.diff(signal))),                       # Mean absolute difference
                np.std(np.diff(signal)),                                # Std of differences
            ]
            # Add frequency domain features
            fft = np.fft.rfft(signal)                                   # FFT
            feat.append(np.mean(np.abs(fft)))                           # Mean frequency magnitude
            feat.append(np.std(np.abs(fft)))                            # Std frequency magnitude
            feat.append(np.argmax(np.abs(fft)))                        # Dominant frequency
            
            features_list.append(feat)
    
    features = np.array(features_list)                                  # Convert to array
    labels_binary = (labels > 0).astype(int)                            # Convert to binary for comparison
    
    return features, labels_binary.reshape(-1, 1)                        # Return features and binary labels


def benchmark_feedforward_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Benchmark feedforward neural network.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING FEEDFORWARD NEURAL NETWORK")
    print("="*60)
    
    # Normalize features
    mean = X_train.mean(axis=0)                                         # Compute mean
    std = X_train.std(axis=0)                                           # Compute std
    X_train_norm = (X_train - mean) / (std + 1e-8)                     # Normalize training
    X_val_norm = (X_val - mean) / (std + 1e-8)                          # Normalize validation
    X_test_norm = (X_test - mean) / (std + 1e-8)                         # Normalize test
    
    # Initialize model
    input_size = X_train.shape[1]                                      # Input dimension
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=[64, 32, 16],                                     # 3 hidden layers
        output_size=1,
        activation='sigmoid',
        learning_rate=0.01,
        random_seed=42
    )
    
    # Train model
    start_time = time.time()                                            # Start timer
    history = model.train(
        X_train_norm, y_train,
        X_val_norm, y_val,
        epochs=500,                                                     # 500 epochs
        batch_size=32,
        verbose=False,
        early_stopping=True,
        patience=20
    )
    train_time = time.time() - start_time                                # Training time
    
    # Evaluate on test set
    start_time = time.time()                                            # Start timer
    test_predictions = model.predict(X_test_norm)                        # Get predictions
    test_probabilities = model.predict_proba(X_test_norm)               # Get probabilities
    inference_time = time.time() - start_time                            # Inference time
    
    # Calculate metrics
    accuracy = model.compute_accuracy(y_test, test_predictions)          # Accuracy
    precision = precision_score(y_test.flatten(), test_predictions.flatten(), zero_division=0) # Precision
    recall = recall_score(y_test.flatten(), test_predictions.flatten(), zero_division=0) # Recall
    f1 = f1_score(y_test.flatten(), test_predictions.flatten(), zero_division=0) # F1 score
    
    # Count parameters (approximate)
    num_params = 0
    for weight in model.weights:                                        # Count weight parameters
        num_params += weight.size
    for bias in model.biases:                                            # Count bias parameters
        num_params += bias.size
    
    results = {
        'model_name': 'Feedforward Neural Network',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': history['loss_history'],
        'train_acc_history': history['accuracy_history'],
        'predictions': test_predictions.flatten().tolist(),
        'probabilities': test_probabilities.flatten().tolist(),
        'true_labels': y_test.flatten().tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return results                                                       # Return results


def benchmark_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark Transformer model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING TRANSFORMER-BASED MODEL")
    print("="*60)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)          # Training dataset
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)                 # Validation dataset
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)             # Test dataset
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Training loader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)   # Validation loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Test loader
    
    # Initialize model
    model = TransformerECGClassifier(
        input_dim=1,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=5,
        max_seq_len=1000
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Total parameters
    
    # Train model
    start_time = time.time()                                            # Start timer
    history = train_transformer(
        model, train_loader, val_loader,
        num_epochs=50,                                                  # 50 epochs
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time                                # Training time
    
    # Evaluate on test set
    start_time = time.time()                                            # Start timer
    results = evaluate_model(model, test_loader, device=device)          # Evaluate
    inference_time = time.time() - start_time                            # Inference time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)       # Convert to binary
    binary_labels = (results['labels'] > 0).astype(int)                 # Convert to binary
    
    # Calculate metrics
    accuracy = results['accuracy']                                       # Accuracy
    precision = precision_score(binary_labels, binary_predictions, zero_division=0) # Precision
    recall = recall_score(binary_labels, binary_predictions, zero_division=0) # Recall
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)    # F1 score
    
    benchmark_results = {
        'model_name': 'Transformer-based ECG Classifier',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results                                             # Return results


def benchmark_three_stage_former(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark Three-Stage Former model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING THREE-STAGE FORMER")
    print("="*60)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ThreeStageFormer(
        input_dim=1,
        d_model=128,
        nhead=8,
        num_layers_per_stage=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=5,
        max_seq_len=1000,
        pooling_stride=2
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_three_stage_former(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    results = evaluate_3stage(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)
    binary_labels = (results['labels'] > 0).astype(int)
    
    # Calculate metrics
    accuracy = results['accuracy']
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': 'Three-Stage Former',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def benchmark_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark 1D CNN model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING 1D CONVOLUTIONAL NEURAL NETWORK")
    print("="*60)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = CNN1DECGClassifier(
        input_channels=1,
        num_classes=5,
        seq_len=1000,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    results = evaluate_cnn_lstm(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)
    binary_labels = (results['labels'] > 0).astype(int)
    
    # Calculate metrics
    accuracy = results['accuracy']
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': '1D CNN',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def benchmark_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark LSTM model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING LSTM")
    print("="*60)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LSTMECGClassifier(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
        bidirectional=True
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    results = evaluate_cnn_lstm(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)
    binary_labels = (results['labels'] > 0).astype(int)
    
    # Calculate metrics
    accuracy = results['accuracy']
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': 'LSTM',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def benchmark_hopfield(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark Hopfield Network model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING HOPFIELD NETWORK")
    print("="*60)
    
    # Create datasets (using ECGDataset from transformer_ecg, compatible interface)
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = HopfieldECGClassifier(
        input_size=1000,
        feature_size=128,
        hidden_size=256,
        num_classes=5,
        num_iterations=10,
        beta=1.0
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_hopfield(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    results = evaluate_hopfield(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)
    binary_labels = (results['labels'] > 0).astype(int)
    
    # Calculate metrics
    accuracy = results['accuracy']
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': 'Hopfield Network',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def plot_comparison(all_results: Dict, save_path: str = 'comparison.png'):
    """Plot comparison of all models."""
    # Extract model names and results
    model_names = []
    results_list = []
    for key, value in all_results.items():
        if isinstance(value, dict) and 'model_name' in value:
            model_names.append(value['model_name'])
            results_list.append(value)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(metrics))
    width = 0.15
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    for i, results in enumerate(results_list):
        values = [results[m] for m in metrics]
        offset = (i - len(results_list) / 2) * width + width / 2
        axes[0, 0].bar(x + offset, values, width, label=model_names[i], alpha=0.8, color=colors[i % len(colors)])
    
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_labels)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    
    # Training time comparison
    times = [r['train_time'] for r in results_list]
    axes[0, 1].bar(model_names, times, alpha=0.8, color=colors[:len(model_names)])
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # Loss history
    for i, results in enumerate(results_list):
        axes[1, 0].plot(results['train_loss_history'], label=f"{model_names[i]} - Loss", alpha=0.7, linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss History')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy history
    for i, results in enumerate(results_list):
        axes[1, 1].plot(results['train_acc_history'], label=f"{model_names[i]} - Accuracy", alpha=0.7, linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training Accuracy History')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def benchmark_vae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark VAE model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING VARIATIONAL AUTOENCODER (VAE)")
    print("="*60)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = VAEEcgClassifier(
        input_size=1000,
        latent_dim=21,  # 21 factors as in FactorECG
        num_classes=5,
        hidden_dims=[256, 128, 64],
        beta=0.001
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_vae(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    results = evaluate_vae(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (results['predictions'] > 0).astype(int)
    binary_labels = (results['labels'] > 0).astype(int)
    
    # Calculate metrics
    accuracy = results['accuracy']
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': 'Variational Autoencoder (VAE)',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def benchmark_ltc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    Benchmark Liquid Time-Constant Network (LTC) model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    device : str
        Device to use
    
    Returns:
    --------
    Dict
        Benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING LIQUID TIME-CONSTANT NETWORK (LTC)")
    print("="*60)
    
    # Create datasets
    train_dataset = LTCECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = LTCECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = LTCECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LTCEcgClassifier(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
        dt=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Train model
    start_time = time.time()
    history = train_ltc(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        patience=10
    )
    train_time = time.time() - start_time
    
    # Evaluate on test set
    start_time = time.time()
    test_loss, test_acc, y_true, y_pred = evaluate_ltc(model, test_loader, device=device)
    inference_time = time.time() - start_time
    
    # Convert multi-class to binary for comparison
    binary_predictions = (y_pred > 0).astype(int)
    binary_labels = (y_true > 0).astype(int)
    
    # Calculate metrics
    accuracy = test_acc / 100.0  # Convert from percentage
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    
    benchmark_results = {
        'model_name': 'Liquid Time-Constant Network (LTC)',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'inference_time': inference_time,
        'num_parameters': int(num_params),
        'train_loss_history': [float(x) for x in history['train_loss']],
        'train_acc_history': [float(x) for x in history['train_acc']],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])],
        'val_acc_history': [float(x) for x in history.get('val_acc', [])],
        'predictions': binary_predictions.tolist(),
        'probabilities': binary_predictions.tolist(),
        'true_labels': binary_labels.tolist()
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Train Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  Parameters: {num_params:,}")
    
    return benchmark_results


def run_complete_benchmark():
    """Run complete benchmark comparison."""
    print("="*60)
    print("COMPREHENSIVE BENCHMARK: 8 ECG CLASSIFICATION MODELS")
    print("="*60)
    
    # Generate synthetic ECG data
    print("\nGenerating synthetic ECG dataset...")
    signals, labels = create_synthetic_ecg_data(
        n_samples=3000,                                                 # 3000 samples
        seq_len=1000,                                                   # 1000 timesteps
        num_classes=5,                                                    # 5 classes
        noise_level=0.1                                                  # 10% noise
    )
    
    # Split data
    split1 = int(0.7 * len(signals))                                     # 70% train
    split2 = int(0.85 * len(signals))                                   # 15% val, 15% test
    
    signals_train = signals[:split1]
    labels_train = labels[:split1]
    signals_val = signals[split1:split2]
    labels_val = labels[split1:split2]
    signals_test = signals[split2:]
    labels_test = labels[split2:]
    
    # Prepare features for feedforward NN
    print("\nExtracting features for Feedforward Neural Network...")
    X_train_ff, y_train_ff = prepare_data_for_feedforward(
        signals_train, labels_train, feature_extraction='statistical'
    )
    X_val_ff, y_val_ff = prepare_data_for_feedforward(
        signals_val, labels_val, feature_extraction='statistical'
    )
    X_test_ff, y_test_ff = prepare_data_for_feedforward(
        signals_test, labels_test, feature_extraction='statistical'
    )
    
    # Benchmark Feedforward NN
    results_ff = benchmark_feedforward_nn(
        X_train_ff, y_train_ff,
        X_val_ff, y_val_ff,
        X_test_ff, y_test_ff
    )
    
    # Benchmark Transformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    results_transformer = benchmark_transformer(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark Three-Stage Former
    results_3stage = benchmark_three_stage_former(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark 1D CNN
    results_cnn = benchmark_cnn(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark LSTM
    results_lstm = benchmark_lstm(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark Hopfield Network
    results_hopfield = benchmark_hopfield(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark VAE
    results_vae = benchmark_vae(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Benchmark LTC
    results_ltc = benchmark_ltc(
        signals_train, labels_train,
        signals_val, labels_val,
        signals_test, labels_test,
        device=device
    )
    
    # Save results to JSON
    all_results = {
        'feedforward_nn': results_ff,
        'transformer': results_transformer,
        'three_stage_former': results_3stage,
        'cnn_1d': results_cnn,
        'lstm': results_lstm,
        'hopfield': results_hopfield,
        'vae': results_vae,
        'ltc': results_ltc
    }
    
    # Create comparison plots
    plot_comparison(all_results, 'benchmark_comparison.png')
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<20} {'FFNN':<10} {'Trans.':<10} {'3stage':<10} {'CNN':<10} {'LSTM':<10} {'Hopfield':<10} {'VAE':<10} {'LTC':<10}")
    print("-" * 120)
    print(f"{'Accuracy':<20} {results_ff['accuracy']:<10.4f} {results_transformer['accuracy']:<10.4f} {results_3stage['accuracy']:<10.4f} {results_cnn['accuracy']:<10.4f} {results_lstm['accuracy']:<10.4f} {results_hopfield['accuracy']:<10.4f} {results_vae['accuracy']:<10.4f} {results_ltc['accuracy']:<10.4f}")
    print(f"{'Precision':<20} {results_ff['precision']:<10.4f} {results_transformer['precision']:<10.4f} {results_3stage['precision']:<10.4f} {results_cnn['precision']:<10.4f} {results_lstm['precision']:<10.4f} {results_hopfield['precision']:<10.4f} {results_vae['precision']:<10.4f} {results_ltc['precision']:<10.4f}")
    print(f"{'Recall':<20} {results_ff['recall']:<10.4f} {results_transformer['recall']:<10.4f} {results_3stage['recall']:<10.4f} {results_cnn['recall']:<10.4f} {results_lstm['recall']:<10.4f} {results_hopfield['recall']:<10.4f} {results_vae['recall']:<10.4f} {results_ltc['recall']:<10.4f}")
    print(f"{'F1 Score':<20} {results_ff['f1_score']:<10.4f} {results_transformer['f1_score']:<10.4f} {results_3stage['f1_score']:<10.4f} {results_cnn['f1_score']:<10.4f} {results_lstm['f1_score']:<10.4f} {results_hopfield['f1_score']:<10.4f} {results_vae['f1_score']:<10.4f} {results_ltc['f1_score']:<10.4f}")
    print(f"{'Train Time (s)':<20} {results_ff['train_time']:<10.2f} {results_transformer['train_time']:<10.2f} {results_3stage['train_time']:<10.2f} {results_cnn['train_time']:<10.2f} {results_lstm['train_time']:<10.2f} {results_hopfield['train_time']:<10.2f} {results_vae['train_time']:<10.2f} {results_ltc['train_time']:<10.2f}")
    print(f"{'Parameters':<20} {results_ff['num_parameters']:<10,} {results_transformer['num_parameters']:<10,} {results_3stage['num_parameters']:<10,} {results_cnn['num_parameters']:<10,} {results_lstm['num_parameters']:<10,} {results_hopfield['num_parameters']:<10,} {results_vae['num_parameters']:<10,} {results_ltc['num_parameters']:<10,}")
    
    print("\nResults saved to benchmark_results.json")
    print("Comparison plot saved to benchmark_comparison.png")
    
    return all_results


if __name__ == "__main__":
    results = run_complete_benchmark()

