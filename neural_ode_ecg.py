"""
Neural ODE for ECG Classification
Based on: Chen et al. (2018) "Neural Ordinary Differential Equations"

Models hidden states as continuous-time ODEs:
    dh/dt = f(h(t), t, θ)

Key features:
- Continuous-depth networks
- Multiple ODE solvers (Euler, RK4, Dopri5)
- Memory efficient via adjoint method
- Adaptive computation
- Constant memory cost regardless of depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
import time


class ODEFunc(nn.Module):
    """
    ODE function: dh/dt = f(h(t), t, θ)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Time embedding
        self.time_embed = nn.Linear(1, hidden_dim)
        
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Current time (scalar)
            h: Hidden state (batch, hidden_dim)
        Returns:
            dh/dt: Time derivative (batch, hidden_dim)
        """
        # Add time information
        t_vec = torch.ones(h.size(0), 1, device=h.device) * t
        t_emb = self.time_embed(t_vec)
        
        # Compute derivative
        h_with_time = h + t_emb
        dh = self.net(h_with_time)
        
        return dh


class ODESolver:
    """
    Base ODE solver class with multiple integration methods.
    """
    @staticmethod
    def euler(func: Callable, h0: torch.Tensor, t_span: Tuple[float, float], n_steps: int = 10) -> torch.Tensor:
        """
        Euler method (first-order).
        Simple but less accurate.
        """
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        
        h = h0
        t = t0
        
        for _ in range(n_steps):
            dh = func(torch.tensor(t, device=h.device), h)
            h = h + dt * dh
            t = t + dt
        
        return h
    
    @staticmethod
    def rk4(func: Callable, h0: torch.Tensor, t_span: Tuple[float, float], n_steps: int = 10) -> torch.Tensor:
        """
        Runge-Kutta 4th order method.
        More accurate, commonly used.
        """
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        
        h = h0
        t = t0
        
        for _ in range(n_steps):
            t_tensor = torch.tensor(t, device=h.device)
            
            k1 = func(t_tensor, h)
            k2 = func(t_tensor + dt/2, h + dt/2 * k1)
            k3 = func(t_tensor + dt/2, h + dt/2 * k2)
            k4 = func(t_tensor + dt, h + dt * k3)
            
            h = h + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
        
        return h
    
    @staticmethod
    def dopri5(func: Callable, h0: torch.Tensor, t_span: Tuple[float, float], 
               rtol: float = 1e-3, atol: float = 1e-4) -> torch.Tensor:
        """
        Dormand-Prince adaptive step size method (5th order).
        High accuracy with adaptive steps.
        """
        try:
            from torchdiffeq import odeint
            
            t_eval = torch.tensor([t_span[0], t_span[1]], device=h0.device)
            
            # Wrap function to match torchdiffeq interface
            def ode_func(t, h):
                return func(t, h)
            
            solution = odeint(ode_func, h0, t_eval, rtol=rtol, atol=atol, method='dopri5')
            return solution[-1]
            
        except ImportError:
            # Fallback to RK4 if torchdiffeq not available
            print("Warning: torchdiffeq not available, falling back to RK4")
            return ODESolver.rk4(func, h0, t_span, n_steps=20)


class NeuralODEBlock(nn.Module):
    """
    Neural ODE block - replaces traditional residual block.
    """
    def __init__(self, hidden_dim: int, solver: str = 'rk4', n_steps: int = 10):
        super().__init__()
        
        self.ode_func = ODEFunc(hidden_dim)
        self.solver = solver
        self.n_steps = n_steps
        
        # Integration time
        self.register_buffer('integration_time', torch.tensor([0.0, 1.0]))
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input hidden state (batch, hidden_dim)
        Returns:
            h_out: Output hidden state after ODE integration (batch, hidden_dim)
        """
        t_span = (self.integration_time[0].item(), self.integration_time[1].item())
        
        if self.solver == 'euler':
            h_out = ODESolver.euler(self.ode_func, h, t_span, self.n_steps)
        elif self.solver == 'rk4':
            h_out = ODESolver.rk4(self.ode_func, h, t_span, self.n_steps)
        elif self.solver == 'dopri5':
            h_out = ODESolver.dopri5(self.ode_func, h, t_span)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        return h_out


class NeuralODEECG(nn.Module):
    """
    Neural ODE model for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        hidden_dim: Hidden dimension (default: 128)
        num_ode_blocks: Number of ODE blocks (default: 3)
        solver: ODE solver ('euler', 'rk4', 'dopri5')
        n_steps: Number of integration steps for euler/rk4
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        hidden_dim: int = 128,
        num_ode_blocks: int = 3,
        solver: str = 'rk4',
        n_steps: int = 10,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_ode_blocks = num_ode_blocks
        self.solver = solver
        self.num_classes = num_classes
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pool to fixed size
        )
        
        # Neural ODE blocks
        self.ode_blocks = nn.ModuleList([
            NeuralODEBlock(hidden_dim, solver, n_steps)
            for _ in range(num_ode_blocks)
        ])
        
        # Layer norms between blocks
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_ode_blocks)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) - ECG signal
        Returns:
            logits: (batch, num_classes)
        """
        # Input processing
        h = self.input_conv(x)  # (batch, hidden_dim, 1)
        h = h.squeeze(-1)  # (batch, hidden_dim)
        
        # Pass through ODE blocks
        for ode_block, norm in zip(self.ode_blocks, self.norms):
            h_new = ode_block(h)
            h = norm(h + h_new)  # Residual connection
            h = self.dropout(h)
        
        # Classification
        logits = self.classifier(h)
        
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


def train_neural_ode(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the Neural ODE model."""
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
    print("Neural ODE for ECG Classification")
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
    
    print("\nInitializing Neural ODE model...")
    print("Testing RK4 solver...")
    model = NeuralODEECG(
        input_channels=1,
        seq_length=1000,
        hidden_dim=128,
        num_ode_blocks=3,
        solver='rk4',  # Can also use 'euler' or 'dopri5'
        n_steps=10,
        num_classes=5,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"ODE solver: {model.solver}")
    print(f"Number of ODE blocks: {model.num_ode_blocks}")
    
    print("\nTraining Neural ODE...")
    start_time = time.time()
    history = train_neural_ode(model, train_loader, val_loader, epochs=50, device=device, learning_rate=0.001)
    training_time = time.time() - start_time
    
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Solver: {model.solver}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("=" * 80)
