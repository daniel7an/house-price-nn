# House Price Prediction — Neural Network

A Multi-Layer Perceptron (MLP) built with PyTorch to predict house prices from tabular features.

## Dataset

**Housing.csv** — 545 samples, 13 features (area, bedrooms, bathrooms, stories, binary amenities, furnishing status).

## Model Architecture

```
Input(13) → Linear(128) → LeakyReLU → Dropout(0.2)
          → Linear(64)  → LeakyReLU → Dropout(0.2)
          → Linear(32)  → LeakyReLU
          → Linear(1)
```

## Results

| Metric | Score |
|--------|-------|
| R²     | 0.6834 |
| RMSE   | 1,086,032 |
| MAE    | 799,221 |

## How to Run

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook house_price_prediction.ipynb
```

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). See `pyproject.toml`.
