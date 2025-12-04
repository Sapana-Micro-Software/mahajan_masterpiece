# Quick Start Guide - ECG Classification Models

## 🚀 Get Started in 5 Minutes

### 1. Installation

```bash
# Clone or navigate to the repository
cd Mahajan_Masterpiece

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Model

```python
# Example: Run Longformer
python longformer_ecg.py
```

**Or try any of the 26+ models:**
```bash
python mamba_ecg.py           # Fastest model
python stacked_transformer_ecg.py  # Most accurate
python neural_pde_ecg.py      # Physics-informed
```

---

## 🎯 Choose Your Model

### For Maximum Speed ⚡
```bash
python mamba_ecg.py        # MAMBA - Linear complexity
python longformer_ecg.py   # Longformer - Efficient attention
```

### For Maximum Accuracy 🎯
```bash
python stacked_transformer_ecg.py  # Deep transformer (12 layers)
python moe_transformer_ecg.py      # Mixture of Experts
```

### For Long Sequences 📏
```bash
python longformer_ecg.py          # Sliding window attention
python bigbird_ecg.py             # Sparse attention
python infinite_transformer_ecg.py # Infinite memory
```

### For Research 🔬
```bash
python hyperneat_ecg.py     # Architecture evolution
python neural_ode_ecg.py    # Continuous-depth
python neural_pde_ecg.py    # PDE-based modeling
```

---

## 📊 Evaluate Models

```python
from evaluation_metrics import ComprehensiveEvaluator
import torch

# Load your model
model = ...  # Your trained model

# Create evaluator
evaluator = ComprehensiveEvaluator(num_classes=5)

# Evaluate
metrics = evaluator.evaluate_model(model, test_loader, device='cpu')

# Generate visualizations
evaluator.plot_confusion_matrix(save_path='confusion.png')
evaluator.plot_roc_curves(save_path='roc.png')
evaluator.plot_metrics_summary(metrics, save_path='summary.png')

# Generate report
report = evaluator.generate_report(metrics, 'MyModel')
print(report)
```

---

## 🚢 Deploy to Production

### Option 1: Docker (Recommended)

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ecg-api

# Access API
curl http://localhost:8000/health
```

### Option 2: Direct Python

```bash
# Start API server
python api_server.py

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 3: Export Models

```python
from model_export import ModelExporter

# Export to multiple formats
exporter = ModelExporter(model, 'my_model', (1, 1, 1000))
results = exporter.export_all('./exports')

# Creates:
# - my_model.onnx (cross-platform)
# - my_model_torchscript.pt (C++ deployment)
# - my_model_quantized.pth (mobile/edge)
```

---

## 🔧 API Usage

### Python Client

```python
import requests
import numpy as np

# Your ECG signal
ecg_signal = np.random.randn(1000).tolist()

# Predict
response = requests.post(
    'http://localhost:8000/predict',
    json={'signal': ecg_signal, 'sampling_rate': 250}
)

result = response.json()
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, 0.3, ...],
    "sampling_rate": 250
  }'

# Model info
curl http://localhost:8000/model/info
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function classifyECG(signal) {
    const response = await axios.post('http://localhost:8000/predict', {
        signal: signal,
        sampling_rate: 250
    });
    
    console.log('Prediction:', response.data.class_name);
    console.log('Confidence:', response.data.confidence);
    return response.data;
}
```

---

## 📚 Available Models (26+)

### Original Models (15)
1. Feedforward Neural Network
2. Transformer
3. Three-Stage Hierarchical Transformer
4. 1D CNN
5. LSTM
6. Hopfield Network
7. VAE
8. Liquid Time-Constant Network (LTC)
9. Hidden Markov Model (HMM)
10. Hierarchical HMM
11. Dynamic Bayesian Network (DBN)
12. Markov Decision Process (MDP)
13. Partially Observable MDP (PO-MDP)
14. Markov Random Field (MRF)
15. Granger Causality

### New Models (11)
16. **Longformer** - Efficient long-sequence processing
17. **MoE Transformer** - Mixture of Experts
18. **Big Bird** - Sparse attention
19. **MAMBA** - State Space Model
20. **BAMBA** - Bidirectional MAMBA
21. **Infinite Transformer** - Memorizing/Infini/XL variants
22. **Stacked Transformer** - Deep architecture (12-24 layers)
23. **HyperNEAT** - Evolutionary architecture
24. **Super-NEAT** - Advanced neuroevolution
25. **Neural ODE** - Continuous-depth networks
26. **Neural PDE** - Heat/Wave/Reaction-Diffusion

---

## 📖 Key Files

| File | Description |
|------|-------------|
| `longformer_ecg.py` | Longformer implementation |
| `mamba_ecg.py` | MAMBA state space model |
| `neural_ode_ecg.py` | Neural ODE with multiple solvers |
| `evaluation_metrics.py` | Comprehensive evaluation tools |
| `model_export.py` | Export to ONNX/TorchScript |
| `api_server.py` | FastAPI production server |
| `benchmark.py` | Compare all models |

---

## 🎓 Tutorials

### Train a Model

```python
from longformer_ecg import LongformerECG, train_longformer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Create model
model = LongformerECG(
    input_channels=1,
    seq_length=1000,
    embed_dim=128,
    num_layers=4,
    num_classes=5
)

# Prepare data
X_train = torch.randn(1000, 1, 1000)  # Your ECG data
y_train = torch.randint(0, 5, (1000,))  # Labels

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# Train
history = train_longformer(
    model, train_loader, val_loader,
    epochs=50, device='cuda'
)
```

### Export for Deployment

```python
from model_export import export_model_wrapper

# Export trained model
results = export_model_wrapper(
    model=trained_model,
    model_name='longformer_ecg',
    input_shape=(1, 1, 1000),
    output_dir='./exports'
)

# Files created:
# - longformer_ecg.onnx
# - longformer_ecg_torchscript.pt
# - longformer_ecg_quantized.pth
# - longformer_ecg_info.json
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Use gradient checkpointing
model = StackedTransformerECG(..., use_gradient_checkpointing=True)
```

### Slow Training
```python
# Use faster model
from mamba_ecg import MambaECG
model = MambaECG(...)  # Linear complexity

# Or enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### API Not Starting
```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn api_server:app --port 8080
```

---

## 📞 Next Steps

1. **Explore Models**: Try different architectures
2. **Evaluate**: Use `evaluation_metrics.py`
3. **Deploy**: Use Docker or API
4. **Customize**: Modify hyperparameters
5. **Integrate**: Use real ECG data

---

## 📚 Documentation

- **NEW_MODELS_README.md** - Detailed model guide
- **DEPLOYMENT_GUIDE.md** - Production deployment
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **API Docs** - http://localhost:8000/docs (when running)

---

## 🎯 Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Test a model
python mamba_ecg.py

# 3. Start API
docker-compose up -d

# 4. Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [...]}'

# 5. Check results
curl http://localhost:8000/health
```

---

**Ready to start? Pick a model and run it!** 🚀

```bash
python mamba_ecg.py  # Try the fastest model
```
