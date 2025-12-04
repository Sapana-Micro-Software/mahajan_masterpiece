"""
Stacked Transformer (Deep Architecture) for ECG Classification
Implements a very deep transformer with up to 24+ layers inspired by models like:
- GPT-2/GPT-3 (up to 96 layers)
- BERT-Large (24 layers)
- PaLM (118 layers)

Features:
- Deep architecture (12-24 layers)
- Pre-layer normalization for stability
- Gradient checkpointing for memory efficiency
- Residual connections with layer scaling
- Enhanced training stability for deep networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time
from torch.utils.checkpoint import checkpoint


class LayerScale(nn.Module):
    """
    Layer scaling for deep transformers (from CaiT paper).
    Helps training very deep networks by scaling residual connections.
    """
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DeepTransformerLayer(nn.Module):
    """
    Deep Transformer layer with pre-norm and layer scaling.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, 
                 dropout: float = 0.1, use_layer_scale: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_layer_scale = use_layer_scale
        
        # Pre-normalization (more stable for deep networks)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        # Layer scaling for stability
        if use_layer_scale:
            self.attn_scale = LayerScale(embed_dim)
            self.ffn_scale = LayerScale(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Pre-norm attention
        normed = self.attn_norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=attention_mask)
        attn_out = self.dropout(attn_out)
        
        # Layer scaling and residual
        if self.use_layer_scale:
            x = x + self.attn_scale(attn_out)
        else:
            x = x + attn_out
        
        # Pre-norm FFN
        normed = self.ffn_norm(x)
        ffn_out = self.fc1(normed)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.fc2(ffn_out)
        ffn_out = self.dropout(ffn_out)
        
        # Layer scaling and residual
        if self.use_layer_scale:
            x = x + self.ffn_scale(ffn_out)
        else:
            x = x + ffn_out
        
        return x


class StackedTransformerECG(nn.Module):
    """
    Stacked (Deep) Transformer for ECG classification.
    
    Args:
        input_channels: Number of input channels (1 for single-lead ECG)
        seq_length: Length of ECG sequence (default: 1000)
        embed_dim: Embedding dimension (default: 256)
        num_layers: Number of transformer layers (default: 24 for "deep")
        num_heads: Number of attention heads (default: 8)
        ffn_multiplier: FFN dimension multiplier (default: 4)
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate (default: 0.1)
        use_layer_scale: Use layer scaling for stability (default: True)
        use_gradient_checkpointing: Use gradient checkpointing to save memory (default: False)
    """
    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        embed_dim: int = 256,
        num_layers: int = 24,  # Deep network!
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1,
        use_layer_scale: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        ffn_dim = embed_dim * ffn_multiplier
        
        # Input projection
        self.input_projection = nn.Conv1d(input_channels, embed_dim, kernel_size=7, padding=3)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Deep transformer layers
        self.layers = nn.ModuleList([
            DeepTransformerLayer(
                embed_dim, num_heads, ffn_dim, dropout, use_layer_scale
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights (important for deep networks)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
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
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through deep transformer layers
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory for very deep networks
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Use CLS token for classification
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


def train_stacked_transformer(model, train_loader, val_loader, epochs=50, device='cpu', learning_rate=0.0001):
    """Train the stacked transformer model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer (better for deep transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler (common for deep transformers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
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
            
            # Gradient clipping (important for deep networks)
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
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step()
        
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
    print("Stacked Transformer (Deep Architecture) for ECG Classification")
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for deep model
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("\nInitializing Stacked Transformer model (24 layers)...")
    model = StackedTransformerECG(
        input_channels=1,
        seq_length=1000,
        embed_dim=128,  # Smaller for demonstration
        num_layers=12,  # 12 layers (can go up to 24+)
        num_heads=8,
        ffn_multiplier=4,
        num_classes=5,
        dropout=0.1,
        use_layer_scale=True,
        use_gradient_checkpointing=False  # Set to True for very deep networks
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of layers: {model.num_layers}")
    
    print("\nTraining Stacked Transformer...")
    start_time = time.time()
    history = train_stacked_transformer(
        model, train_loader, val_loader, 
        epochs=50, device=device, learning_rate=0.0001
    )
    training_time = time.time() - start_time
    
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Architecture: {model.num_layers} layers, {model.embed_dim} dimensions")
    print("=" * 80)
