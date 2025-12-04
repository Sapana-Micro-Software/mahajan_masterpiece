"""
MAMBA (Selective State Space Model) for ECG Classification
Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

Mamba uses selective state space models with data-dependent state transitions,
achieving O(n) complexity while maintaining strong performance on long sequences.

Key features:
- Selective state space mechanism (data-dependent parameters)
- Linear-time complexity O(n)
- Hardware-aware parallel scan
- No attention mechanism (pure SSM)
- Efficient for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - core of Mamba.
    
    State space equation:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)
    
    In Mamba, B and C are data-dependent (selective).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Projects: x -> (B, C, Δ)
        self.x_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(d_model, self.d_inner, bias=True)
        
        # State space parameters
        # A: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        # D: (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Selective B and C projections
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        # Convolution (local processing)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seqlen, dim = x.shape
        
        # Project input
        x_inner = self.x_proj(x)  # (batch, seqlen, d_inner)
        
        # Apply 1D convolution (for local dependencies)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[..., :seqlen].transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # Selective B and C (data-dependent)
        B = self.B_proj(x)  # (batch, seqlen, d_state)
        C = self.C_proj(x)  # (batch, seqlen, d_state)
        
        # Delta (time step) - also data-dependent
        delta = F.softplus(self.dt_proj(x))  # (batch, seqlen, d_inner)
        
        # Discretize A (continuous to discrete)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Selective SSM scan
        y = self.selective_scan(x_conv, delta, A, B, C, self.D)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(
        self, 
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective SSM scan (simplified version).
        
        Args:
            x: (batch, seqlen, d_inner)
            delta: (batch, seqlen, d_inner)
            A: (d_inner, d_state)
            B: (batch, seqlen, d_state)
            C: (batch, seqlen, d_state)
            D: (d_inner,)
        Returns:
            y: (batch, seqlen, d_inner)
        """
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Sequential scan (can be parallelized with associative scan)
        for t in range(seqlen):
            # Discretize: A_discrete = exp(delta * A)
            # B_discrete = delta * B
            dt = delta[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            A_discrete = torch.exp(dt * A.unsqueeze(0))  # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            B_discrete = dt * B_t  # (batch, d_inner, d_state)
            
            # State update: h = A_discrete * h + B_discrete * x
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            h = A_discrete * h + B_discrete * x_t  # (batch, d_inner, d_state)
            
            # Output: y = C * h + D * x
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            y_t = torch.sum(C_t * h, dim=-1) + D.unsqueeze(0) * x[:, t, :]  # (batch, d_inner)
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seqlen, d_inner)
        
        return y


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual
        return x


class MambaECG(nn.Module):
    """
    Mamba model for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        d_model: Model dimension (default: 256)
        n_layers: Number of Mamba layers (default: 6)
        d_state: State space dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Conv1d(input_channels, d_model, kernel_size=7, padding=3)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head (pool over sequence)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) - ECG signal
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


def generate_synthetic_ecg(n_samples: int = 1000, seq_length: int = 1000, 
                          num_classes: int = 5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ECG data for testing."""
    X = []
    y = []
    
    for _ in range(n_samples):
        class_label = np.random.randint(0, num_classes)
        t = np.linspace(0, 4 * np.pi, seq_length)
        
        if class_label == 0:
            signal = np.sin(t) + 0.3 * np.sin(3 * t)
        elif class_label == 1:
            signal = np.sin(t * (1 + 0.3 * np.random.randn()))
        elif class_label == 2:
            signal = np.sin(t * 2) + 0.2 * np.sin(5 * t)
        elif class_label == 3:
            signal = np.sin(t * 0.5) + 0.3 * np.sin(2 * t)
        else:
            signal = np.sin(t) * np.exp(-t / 10)
        
        signal += noise_level * np.random.randn(seq_length)
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        X.append(signal)
        y.append(class_label)
    
    return np.array(X), np.array(y)


def train_mamba(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the Mamba model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model."""
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    print("=" * 80)
    print("MAMBA (Selective State Space Model) for ECG Classification")
    print("=" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nGenerating synthetic ECG data...")
    X, y = generate_synthetic_ecg(n_samples=1000, seq_length=1000, num_classes=5)
    
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\nInitializing Mamba model...")
    model = MambaECG(
        input_channels=1,
        seq_length=1000,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=5,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining Mamba...")
    start_time = time.time()
    history = train_mamba(model, train_loader, val_loader, epochs=50, device=device, learning_rate=0.001)
    training_time = time.time() - start_time
    
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("=" * 80)
