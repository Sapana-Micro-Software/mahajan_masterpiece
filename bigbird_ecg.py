"""
Big Bird Transformer for ECG Classification
Based on: "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)

Big Bird uses a sparse attention mechanism combining:
1. Global attention (to special tokens)
2. Window attention (local neighborhood)
3. Random attention (random positions for long-range dependencies)

This reduces complexity from O(n^2) to O(n) while maintaining performance.

Key features:
- Sparse attention pattern (global + window + random)
- Efficient for long sequences
- O(n) complexity
- Maintains both local and global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import time
import math


class BigBirdAttention(nn.Module):
    """
    Big Bird sparse attention mechanism.
    Combines global, window (sliding), and random attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 3,
                 num_random_blocks: int = 3, block_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size  # Number of blocks to attend on each side
        self.num_random_blocks = num_random_blocks
        self.block_size = block_size
        
        assert self.head_dim * num_heads == embed_dim
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_bigbird_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create Big Bird attention mask.
        Returns a mask where 1 indicates positions to attend to.
        
        Pattern:
        - Global tokens (first few tokens) attend to all positions
        - Other tokens attend to: global tokens + local window + random blocks
        """
        num_blocks = seq_len // self.block_size
        if seq_len % self.block_size != 0:
            num_blocks += 1
        
        # Initialize mask (all zeros = no attention)
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # 1. Global attention: First block attends to everything and everything attends to first block
        global_block_size = min(self.block_size, seq_len)
        mask[:global_block_size, :] = 1
        mask[:, :global_block_size] = 1
        
        # 2. Window attention: Each position attends to local window
        for i in range(seq_len):
            block_idx = i // self.block_size
            
            # Attend to window_size blocks on each side
            start_block = max(0, block_idx - self.window_size)
            end_block = min(num_blocks, block_idx + self.window_size + 1)
            
            start_pos = start_block * self.block_size
            end_pos = min(seq_len, end_block * self.block_size)
            
            mask[i, start_pos:end_pos] = 1
        
        # 3. Random attention: Each block attends to random blocks
        for block_idx in range(1, num_blocks):  # Skip global block
            start_pos = block_idx * self.block_size
            end_pos = min(seq_len, (block_idx + 1) * self.block_size)
            
            # Select random blocks (excluding self and immediate neighbors)
            available_blocks = list(range(1, num_blocks))
            available_blocks = [b for b in available_blocks 
                              if abs(b - block_idx) > self.window_size]
            
            if len(available_blocks) > 0:
                num_random = min(self.num_random_blocks, len(available_blocks))
                random_blocks = np.random.choice(available_blocks, size=num_random, replace=False)
                
                for rand_block in random_blocks:
                    rand_start = rand_block * self.block_size
                    rand_end = min(seq_len, (rand_block + 1) * self.block_size)
                    mask[start_pos:end_pos, rand_start:rand_end] = 1
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            attention_mask: Optional mask
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
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Create Big Bird attention mask
        bigbird_mask = self._create_bigbird_mask(seq_len, hidden_states.device)
        bigbird_mask = bigbird_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Apply mask (set non-attended positions to -inf)
        attn_weights = attn_weights.masked_fill(bigbird_mask == 0, float('-inf'))
        
        # Apply additional mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output


class BigBirdLayer(nn.Module):
    """Single Big Bird transformer layer."""
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 3,
                 num_random_blocks: int = 3, block_size: int = 64,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = BigBirdAttention(
            embed_dim, num_heads, window_size, num_random_blocks, block_size, dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask)
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


class BigBirdECG(nn.Module):
    """
    Big Bird model for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of Big Bird layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        window_size: Number of blocks in sliding window (default: 3)
        num_random_blocks: Number of random blocks to attend (default: 3)
        block_size: Size of each attention block (default: 64)
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
        window_size: int = 3,
        num_random_blocks: int = 3,
        block_size: int = 64,
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
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Big Bird layers
        self.layers = nn.ModuleList([
            BigBirdLayer(embed_dim, num_heads, window_size, num_random_blocks, 
                        block_size, embed_dim * 4, dropout)
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
        
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through Big Bird layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.layer_norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
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


def train_bigbird(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the Big Bird model."""
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
    print("Big Bird Transformer for ECG Classification")
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
    
    print("\nInitializing Big Bird model...")
    model = BigBirdECG(
        input_channels=1,
        seq_length=1000,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        window_size=3,
        num_random_blocks=3,
        block_size=64,
        num_classes=5,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining Big Bird...")
    start_time = time.time()
    history = train_bigbird(model, train_loader, val_loader, epochs=50, device=device, learning_rate=0.001)
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
