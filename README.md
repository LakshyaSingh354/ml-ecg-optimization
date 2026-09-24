# Edge AI Bench

Research harness for measuring the tradeoffs between model accuracy, artifact size,
latency, and memory when deploying medical AI models on CPU devices.

The first study targets multilabel diagnostic classification from 12-lead ECGs using
the public PTB-XL dataset and ONNX Runtime. No patient data or model artifacts belong
in this repository.

## Current milestone

The initial scaffold provides a deterministic synthetic ECG fixture and a benchmark
result schema. This makes the experiment plumbing testable before downloading PTB-XL
or selecting a pretrained checkpoint. Synthetic results must not be used as medical
or deployment conclusions.

The current package also includes PTB-XL metadata parsing and per-lead normalization
primitives. These operate on metadata or arrays supplied by the caller; the repository
does not download or store patient data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,benchmark]'
```

Run the smoke benchmark:

```bash
edge-ai-bench benchmark --config configs/synthetic.toml
```

Run tests:

```bash
python -m pytest
```

## Research protocol

The planned real-data comparison is FP32 versus ONNX Runtime graph optimization,
dynamic INT8, and static INT8 with deterministic calibration data. The benchmark will
record the CPU and software environment, model size, initialization time, warm p50/p95
latency, throughput, memory where available, and macro AUROC/F1. The model's source
checkpoint, preprocessing, PTB-XL patient-wise split, and calibration subset must be
recorded in each experiment manifest.

This is an engineering study, not clinical validation. Accuracy changes must be
interpreted with the dataset's limitations and the model's intended-use context.
