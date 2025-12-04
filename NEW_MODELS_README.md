# New Models Added - Comprehensive Guide

This document describes the 11 new advanced models added to the ECG classification project, bringing the total from 15 to **26+ models**.

## 🚀 New Models Overview

### 1. **Longformer** (`longformer_ecg.py`)
**Based on**: Beltagy et al. (2020) "Longformer: The Long-Document Transformer"

**Key Features**:
- Sliding window attention with global attention
- O(n) complexity instead of O(n²)
- Window size: 256
- Efficient for long ECG sequences

**Use Cases**: Long ECG recordings, Holter monitoring data

**Architecture**:
- 4 Longformer layers
- 8 attention heads
- 128 embedding dimensions

---

### 2. **Mixture of Experts (MoE) Transformer** (`moe_transformer_ecg.py`)
**Based on**: Shazeer et al. (2017) + Fedus et al. (2021) "Switch Transformers"

**Key Features**:
- 8 expert networks per layer
- Top-2 routing for efficiency
- Load balancing loss
- Conditional computation

**Use Cases**: High-capacity models, multi-task scenarios

**Architecture**:
- 4 MoE transformer layers
- 8 experts per layer
- Sparse gating mechanism

---

### 3. **Big Bird** (`bigbird_ecg.py`)
**Based on**: Zaheer et al. (2020) "Big Bird: Transformers for Longer Sequences"

**Key Features**:
- Sparse attention (global + window + random)
- O(n) complexity
- Block size: 64
- Random attention blocks: 3

**Use Cases**: Very long sequences, memory-constrained deployment

**Architecture**:
- Global tokens + sliding window + random blocks
- 4 Big Bird layers

---

### 4. **MAMBA** (`mamba_ecg.py`)
**Based on**: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling"

**Key Features**:
- Selective state space model
- Data-dependent parameters
- O(n) complexity
- No attention mechanism

**Use Cases**: Efficient long-sequence modeling

**Architecture**:
- 4 Mamba blocks
- State dimension: 16
- Convolution kernel: 4

---

### 5. **BAMBA (Bidirectional MAMBA)** (`bamba_ecg.py`)
**Extension of MAMBA**

**Key Features**:
- Bidirectional selective SSM
- Forward and backward processing
- Enhanced temporal context
- Linear complexity per direction

**Use Cases**: Tasks requiring bidirectional context

**Architecture**:
- Dual SSM (forward + backward)
- 4 BAMBA blocks

---

### 6. **Infinite Transformer** (`infinite_transformer_ecg.py`)
**Three variants implemented**:

#### a) **Memorizing Transformers**
- kNN-augmented memory
- Memory size: 1000
- Top-k retrieval: 32

#### b) **Infini-Transformer**
- Compressive memory
- Persistent memory across segments
- Memory dimension: 256

#### c) **Transformer-XL**
- Segment-level recurrence
- Cached previous segments
- Continuous context

**Use Cases**: Very long sequences with memory requirements

---

### 7. **Stacked Transformer (Deep Architecture)** (`stacked_transformer_ecg.py`)
**Inspired by**: GPT-2/3, BERT-Large, PaLM

**Key Features**:
- Up to 12-24 layers
- Pre-layer normalization
- Layer scaling for stability
- Gradient checkpointing option

**Use Cases**: Maximum accuracy, research applications

**Architecture**:
- 12 deep transformer layers
- Layer scaling (CaiT-style)
- 128 embedding dimensions

---

### 8. **HyperNEAT** (`hyperneat_ecg.py`)
**Based on**: Stanley et al. (2009)

**Key Features**:
- CPPN-based weight generation
- Geometric encoding
- Indirect encoding via patterns
- Evolutionary optimization

**Use Cases**: Structured network discovery, research

**Parameters**:
- Population size: 30-50
- Generations: 50-100
- Substrate layers: [16, 8]

---

### 9. **Super-NEAT** (`superneat_ecg.py`)
**Enhanced NEAT with**:

**Key Features**:
- Speciation for diversity
- Adaptive mutation rates
- Novelty search
- Multi-objective optimization

**Use Cases**: Complex topology search, adaptive systems

**Parameters**:
- Population: 100
- Compatibility threshold: 3.0
- 40+ generations

---

### 10. **Neural ODE** (`neural_ode_ecg.py`)
**Based on**: Chen et al. (2018)

**Key Features**:
- Continuous-depth networks
- Multiple ODE solvers: Euler, RK4, Dopri5
- Constant memory cost
- Adjoint method support

**Use Cases**: Continuous-time modeling, memory-efficient training

**Architecture**:
- 3 ODE blocks
- RK4 solver (10 steps)
- 128 hidden dimensions

---

### 11. **Neural PDE** (`neural_pde_ecg.py`)
**Three PDE formulations**:

#### a) **Heat Equation**
- ∂u/∂t = α∇²u
- Diffusion modeling

#### b) **Wave Equation**
- ∂²u/∂t² = c²∇²u
- Propagation modeling

#### c) **Reaction-Diffusion (FitzHugh-Nagumo)**
- Models cardiac action potentials
- ∂u/∂t = D∇²u + R(u)
- Clinically relevant for cardiac modeling

**Use Cases**: Physically-informed models, cardiac electrophysiology

---

## 📊 Performance Comparison

| Model | Complexity | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-----------|-------|----------|----------|
| Longformer | O(n) | ~500K | Fast | High | Long sequences |
| MoE Transformer | O(n²) | ~1M | Moderate | Highest | Multi-task |
| Big Bird | O(n) | ~400K | Fast | High | Memory efficiency |
| MAMBA | O(n) | ~300K | Very Fast | High | Efficiency |
| BAMBA | O(n) | ~600K | Fast | Higher | Bidirectional |
| Infinite Transformer | O(n) | ~500K | Moderate | High | Long context |
| Stacked Transformer | O(n²) | ~2M | Slow | Highest | Max accuracy |
| HyperNEAT | - | Variable | Slow | Moderate | Topology search |
| Super-NEAT | - | Variable | Slow | Moderate | Evolution |
| Neural ODE | O(n) | ~400K | Moderate | High | Continuous-time |
| Neural PDE | O(n) | ~300K | Moderate | High | Physical modeling |

## 🔧 Usage Examples

### Longformer
```python
from longformer_ecg import LongformerECG

model = LongformerECG(
    input_channels=1,
    seq_length=1000,
    embed_dim=128,
    num_layers=4,
    window_size=256,
    num_classes=5
)
```

### MoE Transformer
```python
from moe_transformer_ecg import MoETransformerECG

model = MoETransformerECG(
    input_channels=1,
    num_experts=8,
    top_k=2,
    num_classes=5
)
```

### Neural PDE
```python
from neural_pde_ecg import NeuralPDEECG

model = NeuralPDEECG(
    input_channels=1,
    hidden_channels=64,
    pde_type='reaction_diffusion',  # or 'heat', 'wave'
    num_classes=5
)
```

## 📈 Evaluation

All models support comprehensive evaluation via `evaluation_metrics.py`:

```python
from evaluation_metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(num_classes=5)
metrics = evaluator.evaluate_model(model, test_loader)

# Generate reports
evaluator.plot_confusion_matrix(save_path='confusion.png')
evaluator.plot_roc_curves(save_path='roc.png')
report = evaluator.generate_report(metrics, 'Longformer')
```

## 🚢 Deployment

### Model Export
```python
from model_export import ModelExporter

exporter = ModelExporter(model, 'longformer', (1, 1, 1000))
exporter.export_all('./exports', formats=['onnx', 'torchscript', 'quantized'])
```

### API Deployment
```bash
# Start FastAPI server
python api_server.py

# Or with Docker
docker-compose up -d
```

### Access API
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'signal': ecg_data.tolist()}
)
print(response.json())
```

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: For Neural ODE with adaptive solvers
pip install torchdiffeq

# For API deployment
pip install fastapi uvicorn pydantic

# For model export
pip install onnx onnxruntime
```

## 🏃 Running Models

Each model can be run standalone:

```bash
# Run individual models
python longformer_ecg.py
python moe_transformer_ecg.py
python bigbird_ecg.py
python mamba_ecg.py
python bamba_ecg.py
python infinite_transformer_ecg.py
python stacked_transformer_ecg.py
python hyperneat_ecg.py
python superneat_ecg.py
python neural_ode_ecg.py
python neural_pde_ecg.py
```

## 🔬 Research Applications

1. **Longformer, Big Bird**: Long-term ECG monitoring (24h+ Holter data)
2. **MoE**: Multi-task learning (classification + anomaly detection)
3. **MAMBA/BAMBA**: Real-time streaming ECG analysis
4. **Infinite Transformers**: Patient history integration
5. **Stacked Transformer**: State-of-the-art accuracy research
6. **HyperNEAT/Super-NEAT**: Architecture search, AutoML
7. **Neural ODE**: Continuous monitoring, variable sampling rates
8. **Neural PDE**: Cardiac electrophysiology simulation

## 📚 References

1. **Longformer**: Beltagy et al. (2020)
2. **MoE**: Shazeer et al. (2017), Fedus et al. (2021)
3. **Big Bird**: Zaheer et al. (2020)
4. **MAMBA**: Gu & Dao (2023)
5. **Infinite Transformers**: Wu et al. (2022), Munkhdalai et al. (2024)
6. **HyperNEAT**: Stanley et al. (2009)
7. **NEAT**: Stanley & Miikkulainen (2002)
8. **Neural ODE**: Chen et al. (2018)
9. **Neural PDE**: Brandstetter et al. (2022)

## 🎯 Model Selection Guide

| Your Need | Recommended Model |
|-----------|------------------|
| **Maximum Accuracy** | Stacked Transformer, MoE |
| **Fastest Inference** | MAMBA, Longformer |
| **Long Sequences** | Longformer, Big Bird, Infinite |
| **Low Memory** | Big Bird, MAMBA |
| **Bidirectional Context** | BAMBA, Transformer-XL |
| **Physical Modeling** | Neural PDE |
| **Continuous Time** | Neural ODE, LTC |
| **Architecture Search** | HyperNEAT, Super-NEAT |
| **Real-time Streaming** | MAMBA, Neural ODE |
| **Research/SOTA** | Stacked Transformer, MoE |

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use `gradient_checkpointing=True` in Stacked Transformer
- Use Big Bird or Longformer for long sequences

### Slow Training
- Use MAMBA or Longformer
- Reduce number of layers
- Enable mixed precision training

### Poor Accuracy
- Try Stacked Transformer or MoE
- Increase model capacity
- Tune hyperparameters

## 🤝 Contributing

To add new models:
1. Follow the existing model structure
2. Include comprehensive docstrings
3. Add to `benchmark.py`
4. Update this README

---

**Total Models**: 26+ architectures for ECG classification
**New Models**: 11 advanced architectures added
**Deployment Ready**: ✅ API, Docker, Export utilities included
