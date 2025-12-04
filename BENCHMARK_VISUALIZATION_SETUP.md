# Benchmark Visualization Setup for GitHub Pages

## ✅ What Was Implemented

An automated system to update benchmark visualizations on GitHub Pages using GitHub Actions.

## 🎯 Components

### 1. **Visualization Generator** (`generate_visualizations.py`)
- Reads `benchmark_results.json` from benchmark runs
- Generates interactive Chart.js visualizations
- Creates updated SVG charts with actual data
- Falls back to placeholder data if no results available

### 2. **Index Updater** (`update_index_with_charts.py`)
- Updates `index.html` to include generated visualizations inline
- Ensures charts work without Jekyll includes

### 3. **Benchmark Workflow** (`.github/workflows/update-benchmarks.yml`)
- Runs benchmarks automatically (daily at 2 AM UTC)
- Can be triggered manually via `workflow_dispatch`
- Runs on push to main when benchmark files change
- Generates visualizations from results
- Commits and pushes updates

### 4. **GitHub Pages Workflow** (`.github/workflows/jekyll.yml`)
- Updated to generate visualizations before deployment
- Includes benchmark charts in the site
- Automatically redeploys when benchmarks update

## 📊 Visualizations Generated

### Interactive Charts (Chart.js)
1. **Performance Metrics Comparison**
   - Bar chart showing Accuracy, Precision, Recall, F1 Score
   - Interactive tooltips
   - Color-coded by metric

2. **Training Time Comparison**
   - Bar chart of training times
   - Shows efficiency differences

3. **Accuracy vs Complexity Scatter Plot**
   - Shows trade-offs between accuracy and computational cost
   - Interactive tooltips with model names

### Updated SVG Charts
- Updated scatter plot with actual benchmark positions
- Models positioned based on real performance data

## 🔄 How It Works

### Automatic Updates

1. **Scheduled Runs**: Daily at 2 AM UTC
2. **On Code Changes**: When benchmark.py or model files change
3. **Manual Trigger**: Via GitHub Actions UI

### Process Flow

```
Benchmark Run → Generate Results → Create Visualizations → Update index.html → Deploy to GitHub Pages
```

1. Workflow runs benchmarks (subset of fast models)
2. Results saved to `benchmark_results.json`
3. `generate_visualizations.py` creates HTML/JS charts
4. `update_index_with_charts.py` embeds charts in index.html
5. Changes committed and pushed
6. GitHub Pages automatically redeploys

## 🚀 Usage

### Manual Benchmark Run

1. Go to **Actions** tab in GitHub
2. Select **"Update Benchmark Visualizations"**
3. Click **"Run workflow"**
4. Wait for completion (~5-10 minutes)
5. Visualizations update automatically

### View Results

- Visit: https://sapana-micro-software.github.io/mahajan_masterpiece/
- Scroll to "Comprehensive Comparison" section
- See interactive charts with latest benchmark data

## 📁 Files Created

- `generate_visualizations.py` - Visualization generator
- `update_index_with_charts.py` - Index updater
- `.github/workflows/update-benchmarks.yml` - Benchmark workflow
- `_includes/benchmark_charts.html` - Generated Chart.js charts
- `_includes/updated_svg_charts.html` - Updated SVG charts

## 🔧 Configuration

### Workflow Triggers

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:      # Manual trigger
  push:
    branches: [main]
    paths:
      - 'benchmark.py'
      - '*.py'
```

### Models Benchmarked

Currently runs a subset of fast models:
- FFNN (Feedforward Neural Network)
- CNN (1D Convolutional)
- MAMBA (State Space Model)

Can be extended to include all 26+ models.

## 📊 Data Format

### benchmark_results.json

```json
{
  "model_key": {
    "model_name": "Model Name",
    "accuracy": 0.90,
    "precision": 0.89,
    "recall": 0.88,
    "f1_score": 0.88,
    "train_time": 18.0,
    "train_loss_history": [0.5, 0.4, 0.3],
    "train_acc_history": [0.7, 0.8, 0.9]
  }
}
```

## 🎨 Chart Features

### Interactive Elements
- Hover tooltips with detailed metrics
- Zoom and pan (Chart.js)
- Responsive design
- Color-coded by model type

### Visualizations
- Bar charts for metrics comparison
- Scatter plots for trade-off analysis
- Time series for training progress
- SVG charts for static reference

## 🔍 Monitoring

### Check Workflow Status
1. Go to **Actions** tab
2. View **"Update Benchmark Visualizations"** runs
3. Check logs for any errors

### Verify Deployment
1. Visit GitHub Pages site
2. Check "Comprehensive Comparison" section
3. Verify charts are interactive and show data

## 🛠️ Troubleshooting

### Charts Not Showing
- Check if `benchmark_results.json` exists
- Verify workflow completed successfully
- Check browser console for errors
- Ensure Chart.js CDN loads correctly

### Workflow Fails
- Check Python dependencies in workflow
- Verify model imports work
- Review workflow logs for errors
- Ensure sufficient timeout (60 minutes)

### No Updates
- Verify workflow has write permissions
- Check if changes were committed
- Ensure GitHub Pages is enabled
- Wait for Pages deployment to complete

## 📈 Future Enhancements

- [ ] Add more models to benchmark subset
- [ ] Include inference time metrics
- [ ] Add memory usage tracking
- [ ] Create comparison tables
- [ ] Add export functionality for charts
- [ ] Include confidence intervals
- [ ] Add historical trend tracking

## ✅ Status

- [x] Visualization generator created
- [x] Workflow configured
- [x] GitHub Pages integration
- [x] Automatic updates enabled
- [x] Charts embedded in index.html
- [x] Tested and working

---

**Last Updated**: December 4, 2025  
**Status**: ✅ Fully Operational
