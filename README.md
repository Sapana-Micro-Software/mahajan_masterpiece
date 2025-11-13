# ECG Classification Models: A Comparative Study

This repository implements and compares eight deep learning architectures for ECG classification:

1. **Feedforward Neural Network** (Lloyd et al., 2001) - A feedforward neural network from scratch using NumPy
2. **Transformer-based Model** (Ikram et al., 2025) - Transformer architecture for ECG classification
3. **Three-Stage Hierarchical Transformer (3stageFormer)** (Tang et al., 2025) - Multi-scale hierarchical transformer
4. **1D Convolutional Neural Network (CNN)** - Local pattern extraction using convolution
5. **Long Short-Term Memory (LSTM)** - Sequential modeling with recurrent connections
6. **Hopfield Network** (ETASR, 2013) - Energy-based associative memory for pattern recognition
7. **Variational Autoencoder (VAE)** (van de Leur et al., 2022) - Explainable ECG classification using latent factors
8. **Liquid Time-Constant Network (LTC)** (Hasani et al., 2020) - Continuous-time neural ODE with adaptive time constants

The feedforward neural network implementation is designed for ECG analysis and heart disease prediction tasks, based on research published in *Circulation* (2001) on detecting ischemia in electrocardiograms using artificial neural networks.

## Features

### Feedforward Neural Network
- **Multi-layer perceptron**: Configurable architecture
- **Backpropagation**: Gradient descent with mini-batch support
- **Multiple Activation Functions**: Sigmoid, tanh, and ReLU
- **Training Features**: 
  - Early stopping
  - Validation monitoring
  - Training history tracking
  - Loss and accuracy visualization

### Transformer-based Model
- **Multi-head self-attention**: Captures temporal dependencies
- **Positional encoding**: Preserves temporal information
- **End-to-end learning**: Directly from raw ECG signals

### Three-Stage Hierarchical Transformer (3stageFormer)
- **Multi-scale processing**: Three stages at different temporal resolutions
- **Hierarchical feature extraction**: Captures both local and global patterns
- **Feature fusion**: Combines multi-scale representations for classification

### 1D Convolutional Neural Network (CNN)
- **Local pattern extraction**: Convolutional kernels detect morphological features
- **Translation invariance**: Recognizes patterns regardless of position
- **Efficiency**: Fast training and inference with good accuracy

### Long Short-Term Memory (LSTM)
- **Sequential modeling**: Processes signals step-by-step with memory
- **Bidirectional context**: Considers both past and future information
- **Gating mechanisms**: Selectively remembers important information

### Hopfield Network
- **Associative memory**: Stores and recalls patterns through energy minimization
- **Noise robustness**: Effective at retrieving patterns from noisy or incomplete inputs
- **Pattern completion**: Can reconstruct missing or corrupted signal segments
- **Energy-based learning**: Uses energy function to converge to stable states

### Variational Autoencoder (VAE)
- **Explainable latent factors**: Compresses ECG signals into 21 interpretable factors (FactorECG approach)
- **Dual purpose**: Can be used for both reconstruction and classification
- **Generative capability**: Can generate new ECG signals by sampling from latent space
- **Clinical interpretability**: Latent factors can be associated with physiologically meaningful processes

### Liquid Time-Constant Network (LTC)
- **Continuous-time dynamics**: Models ECG signals as continuous-time processes using neural ODEs
- **Adaptive time constants**: Learns time constants that adapt to input patterns
- **Temporal flexibility**: Captures both fast and slow temporal patterns
- **Neural ODE integration**: Uses differential equations for state evolution

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from neural_network import NeuralNetwork
import numpy as np

# Prepare your data
# X should be shape (n_samples, n_features)
# y should be shape (n_samples, 1) with binary labels (0 or 1)

# Normalize features
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

# Initialize network
nn = NeuralNetwork(
    input_size=10,           # Number of input features
    hidden_layers=[16, 8],   # Two hidden layers with 16 and 8 neurons
    output_size=1,           # Binary classification
    activation='sigmoid',    # or 'tanh', 'relu'
    learning_rate=0.01
)

# Train the network
history = nn.train(
    X_train, y_train,
    X_val, y_val,
    epochs=1000,
    batch_size=32,
    early_stopping=True,
    patience=20
)

# Make predictions
predictions = nn.predict(X_test)
probabilities = nn.predict_proba(X_test)

# Evaluate accuracy
accuracy = nn.compute_accuracy(y_test, predictions)
```

### Running Individual Models

#### Feedforward Neural Network
```bash
python neural_network.py
```

#### Transformer Model
```bash
python transformer_ecg.py
```

#### Three-Stage Former
```bash
python three_stage_former.py
```

#### 1D CNN and LSTM
```bash
python cnn_lstm_ecg.py
```

#### Hopfield Network
```bash
python hopfield_ecg.py
```

#### Variational Autoencoder (VAE)
```bash
python vae_ecg.py
```

#### Liquid Time-Constant Network (LTC)
```bash
python ltc_ecg.py
```

### Running Complete Benchmark

To compare all eight models:

```bash
python benchmark.py
```

This will:
1. Generate a synthetic ECG dataset
2. Train all eight models
3. Evaluate performance
4. Generate comparison plots
5. Save results to `benchmark_results.json`

See [BENCHMARK_README.md](BENCHMARK_README.md) for detailed benchmarking instructions.

## Network Architecture

The default architecture follows a typical pattern for medical classification:
- **Input Layer**: Number of features (e.g., ECG features, patient demographics)
- **Hidden Layers**: Configurable (default: 16 → 8 neurons)
- **Output Layer**: Single neuron with sigmoid activation for binary classification

## Customization

### Adjust Network Architecture

```python
nn = NeuralNetwork(
    input_size=20,              # Your feature count
    hidden_layers=[32, 16, 8], # Add more layers
    output_size=1,
    activation='relu',          # Try different activations
    learning_rate=0.001         # Adjust learning rate
)
```

### Training Parameters

- `epochs`: Number of training iterations
- `batch_size`: Mini-batch size (None for full batch)
- `early_stopping`: Stop training if validation loss doesn't improve
- `patience`: Number of epochs to wait before early stopping

## Data Format

- **Input (X)**: NumPy array of shape `(n_samples, n_features)`
  - Features should be normalized (zero mean, unit variance)
  - Example: ECG features, heart rate variability, patient demographics
  
- **Labels (y)**: NumPy array of shape `(n_samples, 1)`
  - Binary labels: 0 (negative) or 1 (positive)
  - Example: 0 = no ischemia, 1 = ischemia detected

## Implementation Details

- **Weight Initialization**: Xavier/Glorot initialization for better convergence
- **Loss Function**: Binary cross-entropy
- **Optimization**: Gradient descent with backpropagation
- **Activation**: Sigmoid for output layer, configurable for hidden layers

## Notes

This implementation is educational and demonstrates neural network fundamentals. For production use with real ECG data, consider:

1. Proper feature engineering from raw ECG signals
2. Data augmentation techniques
3. Cross-validation for robust evaluation
4. Hyperparameter tuning
5. Integration with medical imaging/ECG processing libraries

## Model Comparison

| Model | Architecture | Input | Parameters | Training Speed | Best For |
|-------|-------------|-------|------------|----------------|----------|
| Feedforward NN | Feature-based MLP | Statistical features | Few (100s-1000s) | Fastest | Real-time, edge devices |
| Transformer | Single-scale Transformer | Raw signals | Many (100Ks) | Moderate | High-accuracy, research |
| Three-Stage Former | Multi-scale Hierarchical | Raw signals (3 resolutions) | Many (100Ks+) | Slowest | High-accuracy, multi-scale patterns |
| 1D CNN | Convolutional | Raw signals | Moderate (10Ks-100Ks) | Fast | Local patterns, efficiency |
| LSTM | Recurrent | Raw signals | Moderate (10Ks-100Ks) | Moderate | Sequential patterns, rhythm analysis |
| Hopfield Network | Energy-based Associative Memory | Raw signals | Moderate (10Ks-100Ks) | Moderate | Pattern completion, noise robustness |
| VAE | Variational Autoencoder | Raw signals | Moderate (10Ks-100Ks) | Moderate | Explainable AI, interpretable factors |
| LTC | Continuous-time Neural ODE | Raw signals | Moderate (10Ks-100Ks) | Moderate | Adaptive temporal dynamics, continuous-time modeling |

## Detailed Comparison and Contrast

### Architectural Similarities

All eight models share common deep learning foundations:
- **End-to-end learning**: All except FFNN process raw ECG signals directly
- **Multi-layer architectures**: All use multiple layers of non-linear transformations
- **Gradient-based optimization**: All trained with backpropagation
- **Regularization**: All employ dropout or similar techniques
- **Classification capability**: All can perform multi-class ECG classification

### Key Architectural Differences

#### 1. **Temporal Modeling Approaches**
- **FFNN**: No temporal modeling (feature-based)
- **Transformer**: Global attention across entire sequence
- **3stageFormer**: Multi-scale attention at three resolutions
- **CNN**: Local convolutional filters with translation invariance
- **LSTM**: Sequential processing with explicit memory gates
- **Hopfield**: Energy-based associative memory
- **VAE**: Latent factor representation with reconstruction
- **LTC**: Continuous-time dynamics with adaptive time constants (neural ODE)

#### 2. **Input Processing**
- **FFNN**: Requires hand-crafted statistical features (mean, std, FFT, etc.)
- **All Others**: Process raw ECG signals directly (1000 timesteps)

#### 3. **Feature Engineering**
- **FFNN**: Manual feature extraction required
- **All Others**: Automatic feature learning from raw signals

#### 4. **Scale Processing**
- **Single-scale**: FFNN, Transformer, CNN, LSTM, Hopfield, VAE, LTC
- **Multi-scale**: Only 3stageFormer processes at multiple temporal resolutions

#### 5. **Model Type**
- **Discriminative**: FFNN, Transformer, 3stageFormer, CNN, LSTM, Hopfield
- **Generative**: VAE (can reconstruct and generate signals)

### Performance Comparison

#### Accuracy Ranking (Expected)
1. **3stageFormer**: Highest accuracy (multi-scale hierarchical processing)
2. **Transformer**: Excellent accuracy (global attention)
3. **LTC, CNN, LSTM, VAE, Hopfield**: Competitive accuracy with different strengths
4. **FFNN**: Good accuracy (limited by feature engineering)

#### Efficiency Ranking
1. **FFNN**: Fastest training and inference
2. **CNN**: Fast with good accuracy-efficiency balance
3. **LSTM, Hopfield, VAE, LTC**: Moderate speed
4. **Transformer**: Moderate speed, higher accuracy
5. **3stageFormer**: Slowest but highest accuracy

### Strengths and Weaknesses Summary

| Model | Key Strengths | Key Weaknesses |
|-------|--------------|----------------|
| **FFNN** | Fastest, simplest, low memory | Requires features, no temporal modeling |
| **Transformer** | High accuracy, global attention | Many parameters, slower training |
| **3stageFormer** | Best accuracy, multi-scale | Most parameters, slowest |
| **CNN** | Good balance, local patterns | Limited long-range dependencies |
| **LSTM** | Sequential modeling, interpretable | Sequential processing, moderate speed |
| **Hopfield** | Noise robust, pattern completion | Limited capacity, iterative updates |
| **VAE** | Explainable, generative, dual purpose | Blurry reconstructions, training complexity |
| **LTC** | Adaptive time constants, continuous-time dynamics | ODE solver overhead, moderate complexity |

### When to Choose Which Model?

- **Choose FFNN** if: Real-time constraints, edge devices, well-understood features
- **Choose Transformer** if: High accuracy needed, single-scale patterns sufficient, research setting
- **Choose 3stageFormer** if: Highest accuracy needed, multi-scale patterns, abundant resources
- **Choose CNN** if: Balance of accuracy and efficiency, local morphological features important
- **Choose LSTM** if: Sequential patterns critical, rhythm analysis, interpretable processing
- **Choose Hopfield** if: Noisy data, pattern completion needed, associative memory beneficial
- **Choose VAE** if: Explainability required, clinical interpretability, generative capabilities needed
- **Choose LTC** if: Continuous-time modeling needed, adaptive temporal dynamics, neural ODE benefits

### Trade-offs Summary

1. **Accuracy vs. Speed**: Higher accuracy models (3stageFormer, Transformer) are slower
2. **Complexity vs. Simplicity**: More powerful models are more complex to implement and train
3. **Feature Engineering vs. End-to-end**: FFNN requires features, others learn automatically
4. **Single-scale vs. Multi-scale**: 3stageFormer unique in multi-scale processing
5. **Discriminative vs. Generative**: VAE only model with generative capabilities
6. **Explainability vs. Performance**: VAE offers highest explainability, 3stageFormer offers best performance
7. **Noise Robustness**: Hopfield excels, others rely on learned representations

## References

- Lloyd, M. D., et al. (2001). "Detection of Ischemia in the Electrocardiogram Using Artificial Neural Networks." *Circulation*, 103(22), 2711-2716.

- Ikram, Sunnia, et al. (2025). "Transformer-based ECG classification for early detection of cardiac arrhythmias." *Frontiers in Medicine*, 12, 1600855.

- Tang, Xiaoya, et al. (2024). "Hierarchical Transformer for Electrocardiogram Diagnosis." *arXiv preprint arXiv:2411.00755*.

- "Electrocardiogram (ECG) Signal Modeling and Noise Reduction Using Hopfield Neural Networks." *Engineering, Technology & Applied Science Research (ETASR)*, Vol. 3, No. 1, 2013.

- van de Leur, Rutger R., et al. (2022). "Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders." *European Heart Journal - Digital Health*, 3(3), 2022. DOI: 10.1093/ehjdh/ztac038.

- Hasani, Ramin, et al. (2020). "Liquid Time-Constant Networks." *arXiv preprint arXiv:2006.04439*. [GitHub](https://github.com/raminmh/liquid_time_constant_networks)

## License

This implementation is provided for educational and research purposes.

