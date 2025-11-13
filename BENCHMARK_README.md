# Benchmark and Comparison Guide

This guide explains how to run the comprehensive benchmark comparing eight deep learning architectures for ECG classification.

## Overview

This project implements and compares eight deep learning architectures for ECG classification:

1. **Feedforward Neural Network** (based on Lloyd et al., 2001)
2. **Transformer-based Model** (based on Ikram et al., 2025)
3. **Three-Stage Hierarchical Transformer (3stageFormer)** (based on Tang et al., 2025)
4. **1D Convolutional Neural Network (CNN)** - Standard baseline for ECG analysis
5. **Long Short-Term Memory (LSTM)** - Sequential modeling with recurrent connections
6. **Hopfield Network** (based on ETASR, 2013) - Energy-based associative memory
7. **Variational Autoencoder (VAE)** (based on van de Leur et al., 2022) - Explainable ECG classification
8. **Liquid Time-Constant Network (LTC)** (based on Hasani et al., 2020) - Continuous-time neural ODE

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.4.0` - Plotting
- `torch>=1.12.0` - PyTorch for Transformer model
- `scikit-learn>=1.0.0` - Evaluation metrics
- `seaborn>=0.11.0` - Enhanced plotting

### Optional: GPU Support

For faster training of the Transformer model, install PyTorch with CUDA support:

```bash
# For CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running the Benchmark

### Quick Start

To run the complete benchmark comparison:

```bash
python benchmark.py
```

This will:
1. Generate synthetic ECG dataset (3000 samples)
2. Train all eight models
3. Evaluate performance
4. Generate comparison plots
5. Save results to `benchmark_results.json`

### Expected Output

The script will display:
- Training progress for all eight models
- Performance metrics (accuracy, precision, recall, F1 score)
- Computational metrics (training time, inference time, parameters)
- Summary comparison table
- Saved plots: `benchmark_comparison.png`
- Saved results: `benchmark_results.json`

### Individual Model Testing

#### Test Feedforward Neural Network Only

```python
python neural_network.py
```

#### Test Transformer Model Only

```python
python transformer_ecg.py
```

#### Test Three-Stage Former Only

```python
python three_stage_former.py
```

#### Test 1D CNN and LSTM

```python
python cnn_lstm_ecg.py
```

#### Test Hopfield Network

```python
python hopfield_ecg.py
```

#### Test Variational Autoencoder (VAE)

```python
python vae_ecg.py
```

#### Test Liquid Time-Constant Network (LTC)

```python
python ltc_ecg.py
```

## Understanding the Results

### Performance Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Computational Metrics

- **Training Time**: Time to train the model (in seconds)
- **Inference Time**: Time to predict on test set (in milliseconds)
- **Parameters**: Number of trainable model parameters

### Output Files

1. **benchmark_comparison.png**: Visual comparison plots
   - Metrics bar chart
   - Training time comparison
   - Loss curves
   - Accuracy curves

2. **benchmark_results.json**: Detailed results in JSON format
   - All metrics for both models
   - Training history
   - Predictions and labels

## LaTeX Documents

### Compiling the Presentation

To compile the Beamer presentation:

```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references
```

Or use:

```bash
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

### Compiling the Paper

To compile the unabridged paper:

```bash
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
bibtex paper        # If using bibliography
pdflatex paper.tex  # Final compilation
```

### Updating LaTeX with Results

After running the benchmark, you should update the LaTeX documents with actual results:

1. Open `benchmark_results.json`
2. Extract the metrics
3. Update tables in `presentation.tex` and `paper.tex`
4. Recompile the LaTeX documents

### Quick Update Script

You can manually update the LaTeX files or create a script to automatically populate results.

## Model Details

### Feedforward Neural Network

- **Input**: 13 statistical features extracted from ECG
- **Architecture**: [64, 32, 16] hidden neurons
- **Activation**: Sigmoid
- **Loss**: Binary cross-entropy
- **Optimizer**: Gradient descent

### Comparison Summary

| Aspect | FFNN | Transformer | 3stageFormer | 1D CNN | LSTM | Hopfield | VAE | LTC |
|--------|------|------------|--------------|--------|------|----------|-----|-----|
| **Architecture** | Feature MLP | Single-scale Transformer | Multi-scale Transformer | Convolutional | Recurrent | Energy-based | Variational Autoencoder | Continuous-time Neural ODE |
| **Input Processing** | Statistical features | Raw signals | Raw signals (3 resolutions) | Raw signals | Raw signals | Raw signals | Raw signals | Raw signals |
| **Temporal Modeling** | None | Excellent (global) | Excellent (multi-scale) | Good (local) | Good (sequential) | Good (associative) | Good (latent factors) | Excellent (adaptive continuous-time) |
| **Parameters** | Few (100s-1Ks) | Many (100Ks) | Many (100Ks+) | Moderate (10Ks-100Ks) | Moderate (10Ks-100Ks) | Moderate (10Ks-100Ks) | Moderate (10Ks-100Ks) | Moderate (10Ks-100Ks) |
| **Training Speed** | Fastest | Moderate | Slowest | Fast | Moderate | Moderate | Moderate | Moderate |
| **Inference Speed** | Fastest | Moderate | Moderate-Slow | Fast | Moderate | Moderate | Moderate | Moderate |
| **Accuracy** | Good | Excellent | Excellent+ | Good-Excellent | Good-Excellent | Good-Excellent | Good-Excellent | Good-Excellent |
| **Best For** | Real-time, edge | High-accuracy, research | Multi-scale patterns | Local patterns, efficiency | Sequential patterns | Pattern completion, noise robustness | Explainable AI, interpretable factors | Adaptive temporal dynamics, continuous-time modeling |

### Transformer Model

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - 6 Transformer encoder layers
  - 8 attention heads
  - Model dimension: 128
  - Feedforward dimension: 512
- **Loss**: Cross-entropy
- **Optimizer**: AdamW with learning rate scheduling

### Three-Stage Hierarchical Transformer (3stageFormer)

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - Three hierarchical stages processing at different temporal resolutions:
    - **Stage 1**: Fine-grained local patterns (full resolution: 1000 timesteps)
    - **Stage 2**: Medium-scale patterns (medium resolution: 500 timesteps)
    - **Stage 3**: Coarse-grained global patterns (low resolution: 250 timesteps)
  - 2 Transformer encoder layers per stage
  - 8 attention heads per stage
  - Model dimension: 128
  - Feedforward dimension: 512
  - Feature fusion layer combining multi-scale representations
- **Loss**: Cross-entropy
- **Optimizer**: AdamW with learning rate scheduling
- **Key Innovation**: Multi-scale hierarchical processing captures both local and global ECG patterns simultaneously

### 1D Convolutional Neural Network

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - 4 convolutional blocks with increasing filters (32→64→128→256)
  - Batch normalization and max pooling after each block
  - Global average pooling
  - Classification head with dropout
- **Loss**: Cross-entropy
- **Optimizer**: AdamW with learning rate scheduling
- **Key Innovation**: Convolutional operations extract local morphological patterns (QRS complexes, P waves, T waves) efficiently

### Long Short-Term Memory (LSTM)

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - 2-layer bidirectional LSTM
  - Hidden size: 128 per direction (256 total)
  - Gating mechanisms (forget, input, output gates)
  - Classification head with dropout
- **Loss**: Cross-entropy
- **Optimizer**: AdamW with learning rate scheduling
- **Key Innovation**: Sequential processing with explicit memory captures temporal dependencies and rhythm patterns

### Hopfield Network

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - Feature extraction layer (128 dimensions)
  - Symmetric weight matrix (256×256) for associative memory
  - Iterative energy-based updates (10 iterations)
  - Energy function: $E = -\frac{1}{2}\sum_{i,j} w_{ij} x_i x_j - \sum_i b_i x_i$
  - Classification head with dropout
- **Loss**: Cross-entropy
- **Optimizer**: AdamW with learning rate scheduling
- **Key Innovation**: Energy-based associative memory enables pattern completion and noise-robust classification

### Variational Autoencoder (VAE)

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - Encoder: Three fully connected layers (1000→256→128→64) with ReLU and dropout
  - Latent space: 21 dimensions (as in FactorECG)
  - Reparameterization: $z = \mu + \epsilon \cdot \sigma$ where $\epsilon \sim \mathcal{N}(0,1)$
  - Decoder: Three fully connected layers (64→128→256→1000) with ReLU and dropout
  - Classification head: Uses latent mean for classification (64→32→5) with ReLU and dropout
  - Beta parameter: 0.001 (controls disentanglement)
- **Loss**: Combined reconstruction (MSE), KL divergence, and cross-entropy
- **Optimizer**: AdamW with learning rate scheduling
- **Key Innovation**: Explainable latent factors enable both reconstruction and classification, providing clinical interpretability

### Liquid Time-Constant Network (LTC)

- **Input**: Raw ECG signals (1000 timesteps)
- **Architecture**: 
  - Liquid Time-Constant cells with adaptive time constants
  - 2-layer LTC network
  - Hidden size: 128
  - Time step (dt): 0.1
  - Time constant network: Learns adaptive time constants based on input
  - State update network: Implements continuous-time dynamics
  - Classification head with dropout
- **Loss**: Cross-entropy
- **Optimizer**: Adam with learning rate scheduling
- **Key Innovation**: Continuous-time neural ODE with adaptive time constants allows the network to model both fast and slow temporal patterns dynamically

## Comprehensive Comparison and Contrast

### Architectural Similarities

All eight models share common deep learning foundations:

- **End-to-end learning**: All except FFNN process raw ECG signals directly (1000 timesteps)
- **Multi-layer architectures**: All use multiple layers of non-linear transformations
- **Gradient-based optimization**: All trained using backpropagation and gradient descent variants
- **Regularization**: All employ dropout or similar regularization techniques
- **Classification capability**: All models can perform multi-class ECG classification
- **PyTorch implementation**: All deep learning models use PyTorch framework

### Key Architectural Differences

#### 1. Temporal Modeling Approaches
- **LTC**: Continuous-time dynamics with adaptive time constants (neural ODE)

| Model | Temporal Approach | Mechanism |
|-------|------------------|-----------|
| **FFNN** | None | Operates on statistical features, no temporal modeling |
| **Transformer** | Global attention | Self-attention across entire sequence simultaneously |
| **3stageFormer** | Multi-scale attention | Attention at three temporal resolutions (1000, 500, 250) |
| **CNN** | Local convolution | Convolutional filters with local receptive fields |
| **LSTM** | Sequential recurrence | Bidirectional LSTM with memory gates |
| **Hopfield** | Energy-based | Iterative energy minimization for pattern convergence |
| **VAE** | Latent factors | Compressed representation in 21-dimensional latent space |

#### 2. Input Processing

- **FFNN**: Requires hand-crafted features (13 statistical features: mean, std, FFT, etc.)
- **All Others**: Process raw ECG signals directly (1000 timesteps, no feature engineering)

#### 3. Feature Engineering Requirements

- **FFNN**: Manual feature extraction required (domain knowledge needed)
- **All Others**: Automatic feature learning from raw signals (no domain knowledge needed)

#### 4. Scale Processing

- **Single-scale**: FFNN, Transformer, CNN, LSTM, Hopfield, VAE (process at one resolution)
- **Multi-scale**: Only 3stageFormer processes at multiple temporal resolutions simultaneously

#### 5. Model Type

- **Discriminative**: FFNN, Transformer, 3stageFormer, CNN, LSTM, Hopfield (classification only)
- **Generative**: VAE (can reconstruct and generate ECG signals)

### Performance Comparison

#### Accuracy Ranking (Expected)

1. **3stageFormer**: Highest accuracy (multi-scale hierarchical processing)
2. **Transformer**: Excellent accuracy (global attention mechanisms)
3. **CNN, LSTM, VAE, Hopfield**: Competitive accuracy with different architectural strengths
4. **FFNN**: Good accuracy (limited by feature engineering)

#### Efficiency Ranking

1. **FFNN**: Fastest training and inference (simple architecture, few parameters)
2. **CNN**: Fast with excellent accuracy-efficiency balance
3. **LSTM, Hopfield, VAE**: Moderate speed with good accuracy
4. **Transformer**: Moderate speed with higher accuracy
5. **3stageFormer**: Slowest but highest accuracy (most parameters, multi-scale processing)

### Detailed Strengths and Weaknesses

#### Feedforward Neural Network
- **Strengths**: Fastest, simplest, low memory, interpretable features, easy deployment
- **Weaknesses**: Requires feature engineering, no temporal modeling, limited accuracy

#### Transformer
- **Strengths**: High accuracy, global attention, end-to-end learning, no feature engineering
- **Weaknesses**: Many parameters, slower training, high memory requirements

#### Three-Stage Hierarchical Transformer
- **Strengths**: Best accuracy, multi-scale processing, hierarchical features, captures local and global patterns
- **Weaknesses**: Most parameters, slowest training, highest memory, most complex

#### 1D Convolutional Neural Network
- **Strengths**: Good balance, local patterns, efficient, translation invariant, fast inference
- **Weaknesses**: Limited long-range dependencies, local focus, may miss global patterns

#### Long Short-Term Memory
- **Strengths**: Sequential modeling, bidirectional context, interpretable, explicit memory
- **Weaknesses**: Sequential processing limits parallelism, moderate speed, vanishing gradients in very long sequences

#### Hopfield Network
- **Strengths**: Noise robust, pattern completion, associative memory, energy-based learning
- **Weaknesses**: Limited capacity (~0.15N patterns for N neurons), iterative updates, quadratic memory growth

#### Variational Autoencoder
- **Strengths**: Explainable latent factors, generative capability, dual purpose, clinical interpretability
- **Weaknesses**: Blurry reconstructions, training complexity (balancing reconstruction, KL, and classification losses), moderate speed

### Trade-offs Summary

1. **Accuracy vs. Speed**: Higher accuracy models (3stageFormer, Transformer) are slower
2. **Complexity vs. Simplicity**: More powerful models are more complex to implement and train
3. **Feature Engineering vs. End-to-end**: FFNN requires features, others learn automatically
4. **Single-scale vs. Multi-scale**: 3stageFormer unique in multi-scale processing
5. **Discriminative vs. Generative**: VAE only model with generative capabilities
6. **Explainability vs. Performance**: VAE offers highest explainability, 3stageFormer offers best performance
7. **Noise Robustness**: Hopfield excels at noise robustness, others rely on learned representations
8. **Memory vs. Capacity**: Hopfield has quadratic memory growth, others scale more efficiently
9. **Local vs. Global**: CNN focuses on local patterns, Transformer/3stageFormer capture global patterns
10. **Sequential vs. Parallel**: LSTM processes sequentially, others can parallelize better

### When to Choose Which Model?

- **Choose FFNN** if: Real-time constraints, edge devices, well-understood features, minimal computational resources
- **Choose Transformer** if: High accuracy needed, single-scale patterns sufficient, research setting, abundant resources
- **Choose 3stageFormer** if: Highest accuracy needed, multi-scale patterns, abundant computational resources, complex temporal patterns
- **Choose CNN** if: Balance of accuracy and efficiency, local morphological features important, fast inference needed
- **Choose LSTM** if: Sequential patterns critical, rhythm analysis, interpretable temporal processing, moderate resources
- **Choose Hopfield** if: Noisy data, pattern completion needed, associative memory beneficial, energy-based learning preferred
- **Choose VAE** if: Explainability required, clinical interpretability, generative capabilities needed, latent factor analysis
- **Choose LTC** if: Continuous-time modeling needed, adaptive temporal dynamics, neural ODE benefits, varying time scales

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size in `benchmark.py`
   - Use CPU instead of GPU
   - Reduce sequence length

2. **Slow Training**
   - Ensure GPU is being used (check device output)
   - Reduce number of epochs
   - Use smaller model dimensions

3. **Inconsistent Results**
   - Set random seeds (already done in code)
   - Ensure same train/test split

### Performance Tips

- Use GPU for Transformer model (much faster)
- Reduce dataset size for quick testing
- Adjust early stopping patience for faster completion

## Extending the Benchmark

### Adding New Models

1. Create model class (following existing patterns)
2. Add benchmark function in `benchmark.py`
3. Update comparison plots
4. Add to LaTeX documents

### Using Real ECG Data

To use real ECG data (e.g., MIT-BIH):

1. Download dataset from PhysioNet
2. Create data loader in `transformer_ecg.py`
3. Update feature extraction in `benchmark.py`
4. Adjust preprocessing as needed

### Custom Metrics

Add custom evaluation metrics:

```python
from sklearn.metrics import roc_auc_score, cohen_kappa_score

# Add to evaluate_model function
auc = roc_auc_score(y_true, y_pred_proba)
kappa = cohen_kappa_score(y_true, y_pred)
```

## Citation

If you use this code, please cite:

```bibtex
@article{lloyd2001,
  title={Detection of Ischemia in the Electrocardiogram Using Artificial Neural Networks},
  author={Lloyd, M. D. and others},
  journal={Circulation},
  volume={103},
  number={22},
  pages={2711--2716},
  year={2001}
}

@article{ikram2025,
  title={Transformer-based ECG classification for early detection of cardiac arrhythmias},
  author={Ikram, Sunnia and others},
  journal={Frontiers in Medicine},
  volume={12},
  pages={1600855},
  year={2025}
}

@article{tang2024hierarchical,
  title={Hierarchical Transformer for Electrocardiogram Diagnosis},
  author={Tang, Xiaoya and Berquist, Jake and Steinberg, Benjamin A and Tasdizen, Tolga},
  journal={arXiv preprint arXiv:2411.00755},
  year={2024}
}

@article{hopfield2013,
  title={Electrocardiogram (ECG) Signal Modeling and Noise Reduction Using Hopfield Neural Networks},
  journal={Engineering, Technology \& Applied Science Research (ETASR)},
  volume={3},
  number={1},
  year={2013}
}

@article{vandeleur2022,
  title={Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders},
  author={van de Leur, Rutger R and Bos, Max N and Taha, Karim and Sammani, Arjan and Yeung, Ming Wai and van Duijvenboden, Stefan and Lambiase, Pier D and Hassink, Rutger J and van der Harst, Pim and Doevendans, Pieter A and Gupta, Deepak K and van Es, Ren{\'e}},
  journal={European Heart Journal - Digital Health},
  volume={3},
  number={3},
  year={2022},
  doi={10.1093/ehjdh/ztac038}
}

@article{hasani2020ltc,
  title={Liquid Time-Constant Networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  journal={arXiv preprint arXiv:2006.04439},
  year={2020}
}
```

## License

This implementation is provided for educational and research purposes.

## Contact

For questions or issues, please refer to the project repository.

