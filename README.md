# [cite_start]Earthquake Prediction Project - M2 IM-IMDS [cite: 3, 5]

Deep Learning project for automated earthquake detection using time-series data. Compares MLP, CNN, and RNN (LSTM) architectures to classify 512-hour seismic windows. Features complexity analysis, performance visualization, and optimized training on PyTorch with MPS support. High-accuracy models designed for low-resource environments.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run experiments: `python main.py`

## Architectures
- **MLP**: Simple dense layers for baseline.
- **CNN**: 1D Convolution for spatial patterns and "ripples".
- [cite_start]**RNN**: LSTM for temporal dependencies in 512-hour sequences[cite: 15, 28].

## Results & Complexity
The experiments were conducted on a MacBook Pro M4 (Metal acceleration). The RNN model demonstrated the highest predictive power for seismic signals.

| Model | Parameters | Inference (s) | Best Acc. |
|-------|------------|---------------|-----------|
| MLP   | 74,050     | 0.0014s       | 74.82%    |
| CNN   | 43,842     | 0.0092s       | 74.82%    |
| RNN   | 50,562     | 0.0047s       | **76.98%**|

## Project Structure
- `models/`: Implementation of the three architectures in PyTorch.
- `utils/`: Data loading pipelines and metric calculation tools.
- `assets/`: Training curves (Loss/Accuracy) and best model weights (`.pth`).
- `data/`: Contains `Earthquakes_TRAIN.tsv` and `Earthquakes_TEST.tsv`.
- `rapport.pdf`: Detailed technical report of findings.

## Author
**Oreste MUHIRWA GABO** - *Master 2 Informatique et Mobilité*