"""
Liquid Time-Constant Network (LTC) for ECG Classification

This implementation is inspired by Liquid Time-Constant Networks (LTCs) described in:
Hasani, R., et al. (2020). "Liquid Time-Constant Networks." arXiv preprint arXiv:2006.04439.
Repository: https://github.com/raminmh/liquid_time_constant_networks

LTCs are continuous-time recurrent neural networks that use neural ODEs with
adaptive time constants, allowing them to model temporal dynamics more effectively
than traditional RNNs. They are particularly well-suited for time-series analysis
like ECG signals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import torch.nn.functional as F
from scipy.integrate import odeint


class ECGDataset(Dataset):
    """Dataset class for ECG signals."""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray, seq_len: int = 1000):
        """
        Initialize ECG dataset.
        
        Parameters:
        -----------
        signals : np.ndarray
            ECG signals of shape (n_samples, seq_len) or (n_samples, seq_len, features)
        labels : np.ndarray
            Class labels of shape (n_samples,)
        seq_len : int
            Sequence length to use (padding/truncation)
        """
        self.signals = signals
        self.labels = labels
        self.seq_len = seq_len
        
        # Ensure signals are 2D (n_samples, seq_len, features)
        if len(self.signals.shape) == 2:
            self.signals = self.signals[:, :, np.newaxis]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Parameters:
        -----------
        idx : int
            Sample index
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (signal, label)
        """
        signal = self.signals[idx]
        
        # Pad or truncate to seq_len
        if signal.shape[0] > self.seq_len:
            signal = signal[:self.seq_len]
        elif signal.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - signal.shape[0], signal.shape[1]))
            signal = np.vstack([signal, padding])
        
        return torch.FloatTensor(signal), torch.LongTensor([self.labels[idx]])[0]


class LiquidTimeConstantCell(nn.Module):
    """
    Liquid Time-Constant Cell for continuous-time dynamics.
    
    Implements a neural ODE-based recurrent cell where the time constant
    is learned and adapts to the input, allowing for adaptive temporal dynamics.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 0.1
    ):
        """
        Initialize LTC Cell.
        
        Parameters:
        -----------
        input_size : int
            Input feature dimension
        hidden_size : int
            Hidden state dimension
        dt : float
            Time step for ODE integration
        """
        super(LiquidTimeConstantCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Time constant network (learns adaptive time constants)
        self.tau_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # Time constants should be positive
        )
        
        # State update network (learns the dynamics)
        self.f_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Initialize time constants to reasonable values
        nn.init.constant_(self.tau_network[-2].bias, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through LTC cell.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)
        h : torch.Tensor
            Hidden state tensor of shape (batch_size, hidden_size)
        
        Returns:
        --------
        torch.Tensor
            Updated hidden state
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute adaptive time constants
        tau = self.tau_network(combined) + 0.1  # Ensure positive, minimum 0.1
        
        # Compute state derivative
        f = self.f_network(combined)
        
        # Euler step: h_new = h + dt * (f - h) / tau
        # This implements the continuous-time dynamics
        h_new = h + self.dt * (f - h) / (tau + 1e-6)
        
        return h_new


class LTCEcgClassifier(nn.Module):
    """
    Liquid Time-Constant Network for ECG Classification.
    
    Uses LTC cells to model continuous-time dynamics in ECG signals,
    allowing for adaptive temporal processing that can capture both
    fast and slow patterns in the signal.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        dt: float = 0.1
    ):
        """
        Initialize LTC ECG Classifier.
        
        Parameters:
        -----------
        input_size : int
            Input feature dimension (1 for single-lead ECG)
        hidden_size : int
            Hidden state dimension
        num_layers : int
            Number of LTC layers
        num_classes : int
            Number of classification classes
        dropout : float
            Dropout rate
        dt : float
            Time step for ODE integration
        """
        super(LTCEcgClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dt = dt
        
        # Stack multiple LTC cells
        self.ltc_cells = nn.ModuleList([
            LiquidTimeConstantCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt=dt
            )
            for i in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LTC network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Initialize hidden states for each layer
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        # Process sequence through LTC cells
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current time step
            
            # Project input if needed
            if isinstance(self.input_proj, nn.Linear):
                x_t = self.input_proj(x_t)
            
            # Process through each LTC layer
            for layer_idx, ltc_cell in enumerate(self.ltc_cells):
                if layer_idx == 0:
                    h[layer_idx] = ltc_cell(x_t, h[layer_idx])
                else:
                    h[layer_idx] = ltc_cell(h[layer_idx - 1], h[layer_idx])
        
        # Use final hidden state from last layer for classification
        final_hidden = h[-1]
        final_hidden = self.dropout(final_hidden)
        
        # Classification
        output = self.classifier(final_hidden)
        
        return output


def train_ltc(
    model: LTCEcgClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    patience: int = 10
) -> dict:
    """
    Train LTC model.
    
    Parameters:
    -----------
    model : LTCEcgClassifier
        LTC model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    device : str
        Device to use ('cpu' or 'cuda')
    patience : int
        Early stopping patience
    
    Returns:
    --------
    dict
        Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return history


def evaluate_model(
    model: LTCEcgClassifier,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate LTC model on test set.
    
    Parameters:
    -----------
    model : LTCEcgClassifier
        Trained LTC model
    test_loader : DataLoader
        Test data loader
    device : str
        Device to use
    
    Returns:
    --------
    Tuple[float, float, np.ndarray, np.ndarray]
        (test_loss, test_accuracy, y_true, y_pred)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    return avg_test_loss, test_accuracy, np.array(all_labels), np.array(all_preds)


def create_synthetic_ecg_data(
    n_samples: int = 1000,
    seq_len: int = 1000,
    num_classes: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic ECG data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    seq_len : int
        Sequence length
    num_classes : int
        Number of classes
    noise_level : float
        Noise level
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (signals, labels)
    """
    signals = []
    labels = []
    
    for i in range(n_samples):
        label = i % num_classes
        labels.append(label)
        
        # Generate synthetic ECG-like signal
        t = np.linspace(0, 2 * np.pi, seq_len)
        
        # Base signal with different patterns per class
        if label == 0:  # Normal
            signal = np.sin(t) + 0.5 * np.sin(2 * t)
        elif label == 1:  # Arrhythmia
            signal = np.sin(t) + 0.8 * np.sin(3 * t) + 0.3 * np.sin(5 * t)
        elif label == 2:  # Ischemia
            signal = 0.8 * np.sin(t) + 0.4 * np.sin(2 * t) + 0.2 * np.sin(4 * t)
        elif label == 3:  # Tachycardia
            signal = np.sin(2 * t) + 0.5 * np.sin(4 * t)
        else:  # Bradycardia
            signal = 0.6 * np.sin(0.5 * t) + 0.3 * np.sin(t)
        
        # Add noise
        signal += noise_level * np.random.randn(seq_len)
        
        signals.append(signal)
    
    return np.array(signals), np.array(labels)


if __name__ == '__main__':
    # Test the implementation
    print("Testing LTC ECG Classifier...")
    
    # Create synthetic data
    signals, labels = create_synthetic_ecg_data(n_samples=200, seq_len=1000, num_classes=5)
    
    # Create dataset
    dataset = ECGDataset(signals, labels, seq_len=1000)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = LTCEcgClassifier(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_classes=5,
        dt=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    history = train_ltc(
        model, train_loader, val_loader,
        num_epochs=20, learning_rate=0.001, device=device
    )
    
    # Evaluate
    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

