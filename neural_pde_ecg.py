"""
Neural PDE for ECG Classification
Implements multiple PDE formulations as neural network layers:
1. Heat Equation (diffusion)
2. Wave Equation (propagation)
3. Reaction-Diffusion Equation (FitzHugh-Nagumo for cardiac modeling)

PDEs model spatial-temporal dynamics in ECG signals.
Particularly relevant for cardiac electrical propagation modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time


class HeatEquationLayer(nn.Module):
    """
    Heat/Diffusion Equation: ∂u/∂t = α ∇²u
    
    Models diffusion processes in signals.
    Discretized using finite differences.
    """
    def __init__(self, channels: int, diffusion_coef: float = 0.1, n_steps: int = 5):
        super().__init__()
        self.channels = channels
        self.diffusion_coef = nn.Parameter(torch.tensor(diffusion_coef))
        self.n_steps = n_steps
        
    def laplacian_1d(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D Laplacian using finite differences.
        Args:
            u: (batch, channels, length)
        Returns:
            laplacian: (batch, channels, length)
        """
        # Pad for boundary conditions
        u_pad = F.pad(u, (1, 1), mode='replicate')
        
        # Central difference: ∇²u ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
        laplacian = u_pad[:, :, 2:] - 2 * u_pad[:, :, 1:-1] + u_pad[:, :, :-2]
        
        return laplacian
    
    def forward(self, u: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Solve heat equation for n_steps.
        Args:
            u: (batch, channels, length)
            dt: Time step size
        Returns:
            u_final: (batch, channels, length)
        """
        for _ in range(self.n_steps):
            laplacian = self.laplacian_1d(u)
            u = u + dt * self.diffusion_coef * laplacian
        
        return u


class WaveEquationLayer(nn.Module):
    """
    Wave Equation: ∂²u/∂t² = c² ∇²u
    
    Models wave propagation (relevant for electrical signal propagation in heart).
    Second-order in time, requires velocity state.
    """
    def __init__(self, channels: int, wave_speed: float = 1.0, n_steps: int = 5):
        super().__init__()
        self.channels = channels
        self.wave_speed = nn.Parameter(torch.tensor(wave_speed))
        self.n_steps = n_steps
        
        # Velocity state
        self.register_buffer('velocity', None)
        
    def laplacian_1d(self, u: torch.Tensor) -> torch.Tensor:
        """Compute 1D Laplacian."""
        u_pad = F.pad(u, (1, 1), mode='replicate')
        laplacian = u_pad[:, :, 2:] - 2 * u_pad[:, :, 1:-1] + u_pad[:, :, :-2]
        return laplacian
    
    def forward(self, u: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Solve wave equation using leapfrog method.
        Args:
            u: (batch, channels, length)
            dt: Time step size
        Returns:
            u_final: (batch, channels, length)
        """
        # Initialize velocity if needed
        if self.velocity is None or self.velocity.shape != u.shape:
            self.velocity = torch.zeros_like(u)
        
        v = self.velocity
        
        for _ in range(self.n_steps):
            # Compute acceleration from spatial Laplacian
            laplacian = self.laplacian_1d(u)
            acceleration = self.wave_speed ** 2 * laplacian
            
            # Update velocity and position (leapfrog)
            v = v + dt * acceleration
            u = u + dt * v
        
        # Update stored velocity
        self.velocity = v.detach()
        
        return u


class ReactionDiffusionLayer(nn.Module):
    """
    Reaction-Diffusion Equation: ∂u/∂t = D∇²u + R(u)
    
    Uses FitzHugh-Nagumo model (simplified cardiac action potential model):
    ∂u/∂t = D∇²u + u(1-u)(u-a) - v
    ∂v/∂t = ε(u - γv)
    
    Where u is activation (fast), v is recovery (slow).
    Highly relevant for cardiac electrical propagation!
    """
    def __init__(self, channels: int, diffusion: float = 0.1, 
                 a: float = 0.1, epsilon: float = 0.01, gamma: float = 0.5, n_steps: int = 5):
        super().__init__()
        self.channels = channels
        
        # Learnable parameters
        self.diffusion = nn.Parameter(torch.tensor(diffusion))
        self.a = nn.Parameter(torch.tensor(a))
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
        self.n_steps = n_steps
        
        # Recovery variable
        self.register_buffer('v', None)
        
    def laplacian_1d(self, u: torch.Tensor) -> torch.Tensor:
        """Compute 1D Laplacian."""
        u_pad = F.pad(u, (1, 1), mode='replicate')
        laplacian = u_pad[:, :, 2:] - 2 * u_pad[:, :, 1:-1] + u_pad[:, :, :-2]
        return laplacian
    
    def reaction(self, u: torch.Tensor) -> torch.Tensor:
        """
        FitzHugh-Nagumo reaction term: u(1-u)(u-a)
        Cubic nonlinearity for excitation.
        """
        return u * (1 - u) * (u - self.a)
    
    def forward(self, u: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        """
        Solve reaction-diffusion equation.
        Args:
            u: (batch, channels, length) - activation variable
            dt: Time step size
        Returns:
            u_final: (batch, channels, length)
        """
        # Initialize recovery variable if needed
        if self.v is None or self.v.shape != u.shape:
            self.v = torch.zeros_like(u)
        
        v = self.v
        
        for _ in range(self.n_steps):
            # Diffusion term
            laplacian = self.laplacian_1d(u)
            diffusion_term = self.diffusion * laplacian
            
            # Reaction term
            reaction_term = self.reaction(u) - v
            
            # Update activation (u)
            du_dt = diffusion_term + reaction_term
            u = u + dt * du_dt
            
            # Update recovery (v)
            dv_dt = self.epsilon * (u - self.gamma * v)
            v = v + dt * dv_dt
        
        # Store recovery variable
        self.v = v.detach()
        
        return u


class NeuralPDEBlock(nn.Module):
    """
    Neural PDE block combining multiple PDE formulations.
    """
    def __init__(self, channels: int, pde_type: str = 'reaction_diffusion', n_steps: int = 5):
        super().__init__()
        
        self.pde_type = pde_type
        
        if pde_type == 'heat':
            self.pde_layer = HeatEquationLayer(channels, n_steps=n_steps)
        elif pde_type == 'wave':
            self.pde_layer = WaveEquationLayer(channels, n_steps=n_steps)
        elif pde_type == 'reaction_diffusion':
            self.pde_layer = ReactionDiffusionLayer(channels, n_steps=n_steps)
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")
        
        # Learnable gating
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.BatchNorm1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            output: (batch, channels, length)
        """
        # Apply PDE evolution
        pde_out = self.pde_layer(x)
        
        # Gated residual connection
        gate = self.gate(x)
        output = gate * pde_out + (1 - gate) * x
        
        # Normalize
        output = self.norm(output)
        
        return output


class NeuralPDEECG(nn.Module):
    """
    Neural PDE model for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        hidden_channels: Number of hidden channels (default: 64)
        num_pde_blocks: Number of PDE blocks (default: 3)
        pde_type: Type of PDE ('heat', 'wave', 'reaction_diffusion')
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        hidden_channels: int = 64,
        num_pde_blocks: int = 3,
        pde_type: str = 'reaction_diffusion',
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_channels = hidden_channels
        self.num_pde_blocks = num_pde_blocks
        self.pde_type = pde_type
        self.num_classes = num_classes
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # PDE blocks
        self.pde_blocks = nn.ModuleList([
            NeuralPDEBlock(hidden_channels, pde_type, n_steps=5)
            for _ in range(num_pde_blocks)
        ])
        
        # Additional processing
        self.process = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) - ECG signal
        Returns:
            logits: (batch, num_classes)
        """
        # Input processing
        x = self.input_conv(x)
        
        # Pass through PDE blocks
        for pde_block in self.pde_blocks:
            x = pde_block(x)
            x = self.dropout(x)
        
        # Process and pool
        x = self.process(x)
        x = x.squeeze(-1)
        
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


def train_neural_pde(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the Neural PDE model."""
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
    print("Neural PDE for ECG Classification")
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
    
    # Test all three PDE types
    for pde_type in ['heat', 'wave', 'reaction_diffusion']:
        print("\n" + "=" * 80)
        print(f"Testing {pde_type.upper()} PDE")
        print("=" * 80)
        
        model = NeuralPDEECG(
            input_channels=1,
            seq_length=1000,
            hidden_channels=64,
            num_pde_blocks=3,
            pde_type=pde_type,
            num_classes=5,
            dropout=0.1
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"PDE type: {pde_type}")
        
        print(f"\nTraining Neural PDE ({pde_type})...")
        start_time = time.time()
        history = train_neural_pde(model, train_loader, val_loader, epochs=30, device=device)
        training_time = time.time() - start_time
        
        test_accuracy = evaluate_model(model, test_loader, device=device)
        
        print(f"\nResults for {pde_type}:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Test accuracy: {test_accuracy:.2f}%")
        print(f"  Best val accuracy: {max(history['val_acc']):.2f}%")
