"""
Mixture of Experts (MoE) Transformer for ECG Classification
Based on: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" 
(Shazeer et al., 2017) and "Switch Transformers" (Fedus et al., 2021)

MoE uses multiple "expert" networks and a gating mechanism to route inputs,
allowing for model capacity scaling while maintaining computational efficiency.

Key features:
- Sparsely-gated mixture of experts
- Top-k routing (k=2 for balance)
- Load balancing loss to prevent expert collapse
- Conditional computation for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time


class Expert(nn.Module):
    """Single expert network - a simple FFN."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SparseGating(nn.Module):
    """
    Sparse gating mechanism for routing inputs to experts.
    Uses top-k routing to activate only k out of n experts.
    """
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Gating network
        self.gate = nn.Linear(embed_dim, num_experts)
        
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            train: Whether in training mode (adds noise)
        Returns:
            gates: (batch, seq_len, top_k) - gate values for selected experts
            indices: (batch, seq_len, top_k) - indices of selected experts
            load: (num_experts,) - load distribution across experts
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute gate logits
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Add noise during training for exploration
        if train and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Softmax to get probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        gates, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize top-k gates
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load for load balancing (how often each expert is selected)
        # Average over batch and sequence
        load = gate_probs.mean(dim=[0, 1])  # (num_experts,)
        
        return gates, indices, load


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    """
    def __init__(self, embed_dim: int, num_experts: int = 8, ffn_dim: int = 2048, 
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(embed_dim, ffn_dim, dropout) for _ in range(num_experts)
        ])
        
        # Gating mechanism
        self.gate = SparseGating(embed_dim, num_experts, top_k)
        
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            train: Whether in training mode
        Returns:
            output: (batch, seq_len, embed_dim)
            load: (num_experts,) - for load balancing loss
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Get gating decisions
        gates, indices, load = self.gate(x, train)  
        # gates: (batch, seq_len, top_k)
        # indices: (batch, seq_len, top_k)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Route to experts and combine
        for i in range(self.top_k):
            # Get gate weights for this position
            gate_weight = gates[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Get expert indices for this position
            expert_idx = indices[:, :, i]  # (batch, seq_len)
            
            # Process with each expert
            for expert_id in range(self.num_experts):
                # Find positions routed to this expert
                mask = (expert_idx == expert_id).unsqueeze(-1).float()  # (batch, seq_len, 1)
                
                if mask.sum() > 0:
                    # Apply expert
                    expert_output = self.experts[expert_id](x)
                    
                    # Add weighted contribution
                    output = output + gate_weight * mask * expert_output
        
        return output, load


class MoETransformerLayer(nn.Module):
    """Single MoE Transformer layer with multi-head attention and MoE FFN."""
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int = 8,
                 ffn_dim: int = 2048, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        
        # MoE layer replaces standard FFN
        self.moe = MoELayer(embed_dim, num_experts, ffn_dim, top_k, dropout)
        self.moe_layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            train: Whether in training mode
        Returns:
            output: (batch, seq_len, embed_dim)
            load: (num_experts,) - for load balancing
        """
        # Self attention with residual
        residual = x
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        x = self.attn_layer_norm(x)
        
        # MoE with residual
        residual = x
        x, load = self.moe(x, train)
        x = residual + x
        x = self.moe_layer_norm(x)
        
        return x, load


class MoETransformerECG(nn.Module):
    """
    Mixture of Experts Transformer for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        num_experts: Number of experts per MoE layer (default: 8)
        top_k: Number of experts to activate (default: 2)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
        load_balance_weight: Weight for load balancing loss (default: 0.01)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        
        # Input projection
        self.input_projection = nn.Conv1d(input_channels, embed_dim, kernel_size=7, padding=3)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # MoE Transformer layers
        self.layers = nn.ModuleList([
            MoETransformerLayer(embed_dim, num_heads, num_experts, embed_dim * 4, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor, return_load: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) - ECG signal
            return_load: Whether to return load distribution for load balancing loss
        Returns:
            logits: (batch, num_classes)
            load_loss: Scalar tensor (if return_load=True)
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through MoE Transformer layers
        total_load_loss = 0.0
        for layer in self.layers:
            x, load = layer(x, train=self.training)
            
            # Compute load balancing loss (encourage uniform distribution)
            # Variance of load distribution - lower is better (more balanced)
            load_loss = torch.var(load) * self.num_experts
            total_load_loss += load_loss
        
        x = self.layer_norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        # Classify
        logits = self.classifier(x)
        
        if return_load:
            return logits, total_load_loss
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


def train_moe(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the MoE Transformer model."""
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
            
            # Forward pass with load balancing loss
            logits, load_loss = model(batch_x, return_load=True)
            
            # Classification loss + load balancing loss
            cls_loss = criterion(logits, batch_y)
            loss = cls_loss + model.load_balance_weight * load_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += cls_loss.item()
            _, predicted = logits.max(1)
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
                
                logits = model(batch_x, return_load=False)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
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
            outputs = model(batch_x, return_load=False)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    print("=" * 80)
    print("Mixture of Experts (MoE) Transformer for ECG Classification")
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
    
    print("\nInitializing MoE Transformer model...")
    model = MoETransformerECG(
        input_channels=1,
        seq_length=1000,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        num_experts=8,
        top_k=2,
        num_classes=5,
        dropout=0.1,
        load_balance_weight=0.01
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of experts per layer: {model.num_experts}")
    print(f"Active experts per token: 2")
    
    print("\nTraining MoE Transformer...")
    start_time = time.time()
    history = train_moe(model, train_loader, val_loader, epochs=50, device=device, learning_rate=0.001)
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
