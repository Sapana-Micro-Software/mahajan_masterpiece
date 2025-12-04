"""
Longformer for ECG Classification
Based on: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)

Longformer uses sliding window attention with global attention for efficient processing
of long sequences, making it ideal for ECG signals with 1000+ timesteps.

Key features:
- Sliding window local attention (window size = 256)
- Global attention on special tokens
- O(n*w) complexity instead of O(n^2)
- Efficient for long sequences like ECG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import time


class LongformerSelfAttention(nn.Module):
    """
    Longformer self-attention with sliding window and global attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _sliding_window_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sliding window attention.
        Args:
            q, k, v: (batch, num_heads, seq_len, head_dim)
            mask: Optional attention mask
        Returns:
            output: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.size()
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # For each position, attend to window_size//2 on each side
        half_window = self.window_size // 2
        
        for i in range(seq_len):
            # Determine window boundaries
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            
            # Extract query for position i
            q_i = q[:, :, i:i+1, :]  # (batch, num_heads, 1, head_dim)
            
            # Extract keys and values for window
            k_window = k[:, :, start:end, :]  # (batch, num_heads, window_len, head_dim)
            v_window = v[:, :, start:end, :]  # (batch, num_heads, window_len, head_dim)
            
            # Compute attention scores
            attn_weights = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scaling
            
            # Apply mask if provided
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask[:, :, i:i+1, start:end] == 0, float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            output[:, :, i:i+1, :] = torch.matmul(attn_weights, v_window)
        
        return output
    
    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global attention for specified positions.
        Args:
            q, k, v: (batch, num_heads, seq_len, head_dim)
            global_mask: (batch, seq_len) - 1 for global positions, 0 otherwise
        Returns:
            output: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.size()
        
        # Get global token positions
        global_positions = global_mask.nonzero(as_tuple=False)
        
        if global_positions.size(0) == 0:
            return torch.zeros_like(q)
        
        output = torch.zeros_like(q)
        
        # For each global position, attend to all positions
        for batch_idx in range(batch_size):
            global_pos_batch = global_positions[global_positions[:, 0] == batch_idx, 1]
            
            if global_pos_batch.size(0) > 0:
                # Extract global queries
                q_global = q[batch_idx:batch_idx+1, :, global_pos_batch, :]
                
                # Attend to all keys/values
                attn_weights = torch.matmul(q_global, k[batch_idx:batch_idx+1].transpose(-2, -1)) * self.scaling
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Apply attention
                output[batch_idx:batch_idx+1, :, global_pos_batch, :] = torch.matmul(
                    attn_weights, v[batch_idx:batch_idx+1]
                )
        
        return output
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            attention_mask: Optional mask for attention
            global_attention_mask: (batch, seq_len) - 1 for global tokens
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute sliding window attention
        output = self._sliding_window_attention(q, k, v, attention_mask)
        
        # Add global attention if specified
        if global_attention_mask is not None:
            global_output = self._global_attention(q, k, v, global_attention_mask)
            output = output + global_output
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output


class LongformerLayer(nn.Module):
    """Single Longformer transformer layer."""
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 256, 
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = LongformerSelfAttention(embed_dim, num_heads, window_size, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Self attention
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states


class LongformerECG(nn.Module):
    """
    Longformer model for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of Longformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        window_size: Sliding window size (default: 256)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        window_size: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Conv1d(input_channels, embed_dim, kernel_size=7, padding=3)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # Global attention token (like CLS token)
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Longformer layers
        self.layers = nn.ModuleList([
            LongformerLayer(embed_dim, num_heads, window_size, embed_dim * 4, dropout)
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) - ECG signal
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Input projection: (batch, channels, seq_len) -> (batch, embed_dim, seq_len)
        x = self.input_projection(x)
        
        # Transpose: (batch, embed_dim, seq_len) -> (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Add global token at the beginning
        global_tokens = self.global_token.expand(batch_size, -1, -1)
        x = torch.cat([global_tokens, x], dim=1)
        
        # Create global attention mask (only first token is global)
        global_attention_mask = torch.zeros(batch_size, x.size(1), device=x.device)
        global_attention_mask[:, 0] = 1
        
        # Pass through Longformer layers
        for layer in self.layers:
            x = layer(x, global_attention_mask=global_attention_mask)
        
        x = self.layer_norm(x)
        
        # Use global token for classification
        x = x[:, 0]  # (batch, embed_dim)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


def generate_synthetic_ecg(n_samples: int = 1000, seq_length: int = 1000, 
                          num_classes: int = 5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ECG data for testing."""
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random class
        class_label = np.random.randint(0, num_classes)
        
        # Generate base signal
        t = np.linspace(0, 4 * np.pi, seq_length)
        
        # Different patterns for different classes
        if class_label == 0:  # Normal
            signal = np.sin(t) + 0.3 * np.sin(3 * t)
        elif class_label == 1:  # Arrhythmia
            signal = np.sin(t * (1 + 0.3 * np.random.randn()))
        elif class_label == 2:  # Tachycardia
            signal = np.sin(t * 2) + 0.2 * np.sin(5 * t)
        elif class_label == 3:  # Bradycardia
            signal = np.sin(t * 0.5) + 0.3 * np.sin(2 * t)
        else:  # Abnormal
            signal = np.sin(t) * np.exp(-t / 10)
        
        # Add noise
        signal += noise_level * np.random.randn(seq_length)
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        X.append(signal)
        y.append(class_label)
    
    return np.array(X), np.array(y)


def train_longformer(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the Longformer model."""
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
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
    print("Longformer for ECG Classification")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic ECG data...")
    X, y = generate_synthetic_ecg(n_samples=1000, seq_length=1000, num_classes=5)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert to PyTorch tensors and create data loaders
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
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
    
    # Initialize model
    print("\nInitializing Longformer model...")
    model = LongformerECG(
        input_channels=1,
        seq_length=1000,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        window_size=256,
        num_classes=5,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining Longformer...")
    start_time = time.time()
    history = train_longformer(model, train_loader, val_loader, epochs=50, device=device, learning_rate=0.001)
    training_time = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("=" * 80)
