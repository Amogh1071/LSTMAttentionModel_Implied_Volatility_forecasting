# Implied Volatility Prediction for ETH

## Overview

This repository contains a Python Notebook that predicts 10-second-ahead implied volatility (IV) for Ethereum (ETH) using high-frequency order book and OHLCV data. The model is built for a Kaggle-like environment and uses a two-layer LSTM with attention, trained with a custom CombinedLoss function (0.5*MSE + 0.5*MAE). The workflow includes extensive feature engineering to improve forecast accuracy.

## Project Details

* **Goal**: Forecast 10-second-ahead IV for ETH
* **Evaluation Metric**: Root Mean Squared Percentage Error (RMSPE)
* **Dataset**: High-frequency order book data from `ETH.csv` (train/test) with 1-second resolution. Includes up to 5 bid/ask levels, trade volumes, and IV labels for training. Optionally integrates ETH data for cross-asset features.
* **Development Date**: 26 September 2025, 2:08 PM IST

## Code Highlights

* **Model**: 2-layer LSTM with attention

  * Hidden sizes: 64, 32
  * Dropout: 0.3
* **Loss Function**: CombinedLoss (0.5*MSE + 0.5*MAE). RMSPE is computed for validation.
* **Feature Engineering** (30+ features):

  * *Base*: mid-price, spread, volume imbalance, price/volume velocity, RSI, MACD, ATR
  * *New*: short-term volatility (30s, 60s, 120s), book imbalance, price momentum, ETH volatility
  * *Lagged*: lags 1 and 5 for IV, volatility, RSI
  * *Time-based*: hour of day, day of week
  * *Microstructure*: spread volatility, VWAP volatility, book slope volatility
* **Training Optimizations**:

  * Batch size = 128 (reduce to 64 if memory issues)
  * Gradient clipping (max_norm=0.5)
  * Mixed precision training
  * Explicit memory cleanup (`gc.collect()`, `torch.cuda.empty_cache()`)

## Installation and Dependencies

**Required Libraries**:
`pandas`, `numpy`, `torch`, `scikit-learn`, `scipy`, `matplotlib`

Install with:

```bash
pip install pandas numpy torch scikit-learn scipy matplotlib
```

**Environment**:

* Kaggle with dual T4 GPUs (recommended)
* Local GPU setup with CUDA support

**Dataset Setup on Kaggle**:

* Train: `/kaggle/input/ethtrain/ETH.csv`
* Test: `/kaggle/input/ethtest/ETH.csv`
* Peer (optional): `/kaggle/input/peer_crypto/ETH.csv`

## Usage

**Run the Script**

* On Kaggle (GPU enabled):

  ```bash
  !python lstm_model_kaggle_combined_loss.py
  ```
* Locally: adjust dataset paths and run the script.

**Outputs**

* **Prediction File**: `/kaggle/working/submission.csv` (270,548 rows, `timestamp,predicted` format)
* **Plots**:

  * `iv_distribution.png`: histogram of log-transformed IV
  * `validation_plot.png`: actual vs. predicted IV (validation)
  * `scatter_plot.png`: scatter plot of actual vs. predicted IV
* **Logs**: validation metrics including RMSE, MAE, RMSPE, Pearson correlation, and directional accuracy

## Validation Results

* RMSE: **0.1694**
* MAE: **0.0787**
* Pearson correlation: **0.9142**
* Directional accuracy: **0.5096**
* RMSPE: varies (typically 20–100 across experiments)

**Interpretation**: The high Pearson correlation indicates strong linear consistency between predicted and actual IV. Directional accuracy is moderate, leaving room to improve trend prediction.

## Assumptions

* IV becomes stationary after differencing (checked with ADF test).
* Cross-asset signals (e.g., ETH volatility) add predictive value for ETH IV.
* A 120-second look-back window captures sufficient temporal dependencies.

## Troubleshooting

* **MemoryError**: reduce batch size to 64 or disable `pin_memory=True`.
* **High RMSPE (>100)**: increase training epochs (e.g., 30), adjust CombinedLoss weights (e.g., 0.7 for MSE), or add more lagged features.
* **Shape mismatch in validation logs**: check with `.flatten()`.
* **Missing ETH data**: handled gracefully by the script.


## Future Work

* Experiment with ensemble methods or transformer-based models
* Try different volatility windows (15s, 240s, etc.)
* Explore GARCH or hybrid approaches for volatility forecasting

## License

This project is for educational purposes only. Commercial use is not permitted.

*Last updated: 21 September 2025, 2:08 PM IST*

---

