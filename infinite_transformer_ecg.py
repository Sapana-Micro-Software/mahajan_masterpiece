"""
Infinite Transformer for ECG Classification
Implements three variants with infinite/extended memory:
1. Memorizing Transformers (kNN-augmented memory)
2. Infini-Transformer (compressive memory)
3. Transformer-XL (segment-level recurrence)

These architectures extend transformers with external memory mechanisms
to handle very long sequences efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import time


# ============================================================================
# 1. Memorizing Transformer (kNN-augmented memory)
# ============================================================================

class MemorizingTransformerLayer(nn.Module):
    """
    Memorizing Transformer layer with kNN-augmented memory.
    Stores past key-value pairs and retrieves nearest neighbors.
    """
    def __init__(self, embed_dim: int, num_heads: int, memory_size: int = 1000,
                 top_k: int = 32, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.top_k = top_k
        
        # Standard attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # Memory storage (non-parametric)
        self.register_buffer('memory_keys', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Memory attention
        self.memory_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.memory_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def update_memory(self, keys: torch.Tensor, values: torch.Tensor):
        """Update memory with new key-value pairs."""
        batch_size, seq_len, _ = keys.shape
        
        for i in range(batch_size):
            for j in range(seq_len):
                ptr = self.memory_ptr.item()
                self.memory_keys[ptr] = keys[i, j].detach()
                self.memory_values[ptr] = values[i, j].detach()
                self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def retrieve_from_memory(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k nearest neighbors from memory."""
        batch_size, seq_len, embed_dim = queries.shape
        
        # Compute distances to all memory keys
        # queries: (batch, seq_len, embed_dim)
        # memory_keys: (memory_size, embed_dim)
        
        # Normalize for cosine similarity
        queries_norm = F.normalize(queries, p=2, dim=-1)
        memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=-1)
        
        # Cosine similarity
        similarities = torch.matmul(queries_norm, memory_keys_norm.t())  # (batch, seq_len, memory_size)
        
        # Get top-k
        top_k_sims, top_k_indices = torch.topk(similarities, k=min(self.top_k, self.memory_size), dim=-1)
        
        # Retrieve values
        retrieved_values = self.memory_values[top_k_indices]  # (batch, seq_len, top_k, embed_dim)
        
        return retrieved_values, top_k_sims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Standard self-attention
        residual = x
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(x_attn)
        x = self.attn_norm(x)
        
        # Memory-augmented attention
        if self.training:
            # Update memory with current keys and values
            self.update_memory(x, x)
        
        # Retrieve from memory
        retrieved_values, sims = self.retrieve_from_memory(x)
        
        # Reshape for attention
        batch_size, seq_len, top_k, embed_dim = retrieved_values.shape
        retrieved_flat = retrieved_values.view(batch_size, seq_len * top_k, embed_dim)
        
        # Attend to retrieved values
        residual = x
        x_repeat = x.unsqueeze(2).repeat(1, 1, top_k, 1).view(batch_size, seq_len * top_k, embed_dim)
        mem_attn, _ = self.memory_attn(x, retrieved_flat, retrieved_flat)
        x = residual + self.dropout(mem_attn)
        x = self.memory_norm(x)
        
        # FFN
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.ffn_norm(x)
        
        return x


# ============================================================================
# 2. Infini-Transformer (Compressive Memory)
# ============================================================================

class InfiniAttention(nn.Module):
    """
    Infini-Attention with compressive memory.
    Maintains compressed memory state across segments.
    """
    def __init__(self, embed_dim: int, num_heads: int, memory_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.memory_dim = memory_dim
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Compressive memory (persistent across segments)
        self.register_buffer('memory_state', torch.zeros(1, memory_dim, embed_dim))
        self.register_buffer('memory_norm', torch.ones(1, memory_dim, 1))
        
        # Memory update
        self.memory_update = nn.Linear(embed_dim, memory_dim)
        self.memory_gate = nn.Linear(embed_dim, memory_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Memory attention
        # Expand memory for batch
        memory_state = self.memory_state.expand(batch_size, -1, -1)
        
        # Query memory
        mem_attn_weights = torch.matmul(x, memory_state.transpose(-2, -1)) * self.scale
        mem_attn_weights = F.softmax(mem_attn_weights, dim=-1)
        mem_output = torch.matmul(mem_attn_weights, memory_state)
        
        # Combine standard and memory attention
        output = attn_output + 0.5 * mem_output
        output = self.out_proj(output)
        
        # Update memory (compress current segment)
        if self.training:
            # Aggregate segment information
            segment_summary = x.mean(dim=1, keepdim=True)  # (batch, 1, embed_dim)
            
            # Update gate
            gate = torch.sigmoid(self.memory_gate(segment_summary))  # (batch, 1, memory_dim)
            gate = gate.transpose(-2, -1)  # (batch, memory_dim, 1)
            
            # New memory values
            new_mem = self.memory_update(segment_summary)  # (batch, 1, memory_dim)
            new_mem = new_mem.transpose(-2, -1).unsqueeze(-1)  # (batch, memory_dim, 1, 1)
            
            # Update memory with gating (exponential moving average)
            self.memory_state = self.memory_state * (1 - gate.mean(0, keepdim=True)) + \
                               segment_summary.mean(0, keepdim=True).unsqueeze(1) * gate.mean(0, keepdim=True)
        
        return output


class InfiniTransformerLayer(nn.Module):
    """Infini-Transformer layer with compressive memory."""
    def __init__(self, embed_dim: int, num_heads: int, memory_dim: int = 256,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.infini_attn = InfiniAttention(embed_dim, num_heads, memory_dim, dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Infini-attention
        residual = x
        x = self.infini_attn(x)
        x = self.dropout(x)
        x = residual + x
        x = self.attn_norm(x)
        
        # FFN
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.ffn_norm(x)
        
        return x


# ============================================================================
# 3. Transformer-XL (Segment-level Recurrence)
# ============================================================================

class TransformerXLLayer(nn.Module):
    """
    Transformer-XL layer with segment-level recurrence.
    Caches previous segment's hidden states.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Memory for previous segment
        self.register_buffer('prev_segment', None)
        
    def forward(self, x: torch.Tensor, use_memory: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            use_memory: Whether to use cached previous segment
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Concatenate with previous segment if available
        if use_memory and self.prev_segment is not None and self.prev_segment.size(0) == x.size(0):
            # Concatenate along sequence dimension
            x_with_mem = torch.cat([self.prev_segment, x], dim=1)
        else:
            x_with_mem = x
        
        # Self-attention (attend to current + previous segment)
        residual = x
        if x_with_mem.size(1) > x.size(1):
            # Query only current segment, but key/value include previous
            attn_out, _ = self.self_attn(x, x_with_mem, x_with_mem)
        else:
            attn_out, _ = self.self_attn(x, x, x)
        
        x = residual + self.dropout(attn_out)
        x = self.attn_norm(x)
        
        # Update memory (cache current segment)
        if self.training:
            self.prev_segment = x.detach()
        
        # FFN
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.ffn_norm(x)
        
        return x


# ============================================================================
# Unified Infinite Transformer Model
# ============================================================================

class InfiniteTransformerECG(nn.Module):
    """
    Infinite Transformer for ECG classification.
    Supports three variants: 'memorizing', 'infini', 'xl'
    
    Args:
        variant: 'memorizing', 'infini', or 'xl'
        input_channels: Number of input channels
        seq_length: ECG sequence length
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    def __init__(
        self,
        variant: str = 'infini',  # 'memorizing', 'infini', or 'xl'
        input_channels: int = 1,
        seq_length: int = 1000,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert variant in ['memorizing', 'infini', 'xl'], \
            f"variant must be 'memorizing', 'infini', or 'xl', got {variant}"
        
        self.variant = variant
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Conv1d(input_channels, embed_dim, kernel_size=7, padding=3)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # Transformer layers based on variant
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if variant == 'memorizing':
                self.layers.append(MemorizingTransformerLayer(embed_dim, num_heads, dropout=dropout))
            elif variant == 'infini':
                self.layers.append(InfiniTransformerLayer(embed_dim, num_heads, dropout=dropout))
            elif variant == 'xl':
                self.layers.append(TransformerXLLayer(embed_dim, num_heads, dropout=dropout))
        
        self.norm = nn.LayerNorm(embed_dim)
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
            x: (batch, channels, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


def generate_synthetic_ecg(n_samples: int = 1000, seq_length: int = 1000, 
                          num_classes: int = 5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ECG data."""
    X, y = [], []
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


def train_model(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.001):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
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
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
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
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


if __name__ == "__main__":
    print("=" * 80)
    print("Infinite Transformer for ECG Classification")
    print("Testing all three variants: Memorizing, Infini, Transformer-XL")
    print("=" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    X, y = generate_synthetic_ecg(n_samples=1000, seq_length=1000, num_classes=5)
    
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(1),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(1),
        torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test Infini-Transformer variant
    print("\n" + "=" * 80)
    print("Training Infini-Transformer variant...")
    print("=" * 80)
    
    model = InfiniteTransformerECG(
        variant='infini',
        input_channels=1,
        seq_length=1000,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        num_classes=5,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    history = train_model(model, train_loader, val_loader, epochs=30, device=device)
    training_time = time.time() - start_time
    
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("=" * 80)
