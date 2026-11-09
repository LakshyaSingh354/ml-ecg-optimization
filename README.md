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

The real baseline runner is available as a Python API through
`edge_ai_bench.real_baseline.run_baseline`. It downloads metadata and requested
waveforms at runtime, trains a bounded scikit-learn multilabel baseline on official
PTB-XL folds 1-8/9/10, exports ONNX, checks output agreement, and records CPU latency.

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

Run the representative real PTB-XL baseline:

```bash
edge-ai-bench real-baseline
```

This downloads metadata and 1,400 selected 100 Hz waveform records into ignored local
directories, trains a feature-based multilabel logistic baseline on folds 1-8, saves
the source checkpoint as `artifacts/ptbxl-logistic.joblib`, exports the FP32 deployment
artifact as `artifacts/ptbxl-logistic-fp32.onnx`, validates ONNX outputs against the
source model, and writes `results/ptbxl-baseline.json`. Smaller limits can be passed
explicitly for a network smoke check.

Run tests:

```bash
python -m pytest
```

The checkpoint benchmark evaluates the FP32 artifact through both PyTorch (reference)
and ONNX Runtime (deployment) paths. The ONNX Runtime result records initialization
time, provider, graph optimization level, thread count, peak RSS measurements, and
latency/throughput for batch sizes 1, 4, 8, and 16. Override runtime settings when
comparing configurations:

```bash
edge-ai-bench checkpoint-baseline --onnx-threads 4 --onnx-optimization all
```

Run the FP32 runtime sweep across graph optimization levels and thread counts:

```bash
edge-ai-bench checkpoint-sweep --threads 1 2 4 8
```

The sweep writes `results/ecg-hybrid-net-fp32-sweep.json` and uses the same fixed
PTB-XL train/test selection and preprocessing as the checkpoint baseline.

Generate compiler and quantization variants:

```bash
edge-ai-bench optimize-checkpoint --threads 2 --calibration-limit 100
```

The default is the fast 1,000-train/200-test engineering subset. For the complete
PTB-XL labeled split, use approximately 4,200 training records and 677 test
records; missing waveform files are downloaded into the ignored data directory:

```bash
edge-ai-bench optimize-checkpoint \
  --threads 2 \
  --train-limit 4200 \
  --test-limit 677 \
  --calibration-limit 500 \
  --isolated-repeats 3
```

This writes `results/ecg-hybrid-net-optimizations.json` and evaluates four explicit
conditions:

1. **Baseline**: original FP32 graph with ONNX Runtime graph optimization disabled.
2. **Compiler only**: serialized `ORT_ENABLE_EXTENDED` FP32 graph with runtime
   optimization disabled at load time.
3. **Quantization only**: dynamic and static INT8 artifacts generated from the
   original FP32 graph, loaded without runtime graph optimization.
4. **Compiler plus quantization**: original graph is quantized first, then each
   INT8 artifact is serialized through `ORT_ENABLE_EXTENDED`, and loaded without
   runtime graph optimization.

Every variant is evaluated on the fixed PTB-XL test selection with artifact size,
macro AUROC/F1, output error, initialization time, RSS, latency, and throughput.
The optimization result also records per-class AUROC/F1, the exact calibration
record IDs and their SHA-256 identity, three repeated timing runs, and the
non-dominated Pareto variants for artifact size, batch-1 latency, and macro F1.
The static quantization result is a systems measurement, not clinical validation;
calibration identity and the fixed test split must remain unchanged when comparing
future variants.

The reported current RSS and peak RSS values are process-level measurements. The
initialization RSS delta is sampled around ONNX Runtime session creation, while
`resource.getrusage` peak RSS is cumulative for the process; compare runs on the
same host and do not interpret either value as isolated tensor allocation size.
Optimization runs now additionally launch a fresh subprocess for every timing
repeat, using memory-mapped test signals. Each variant records per-process current
RSS, peak RSS, initialization time, and raw latency/throughput measurements; the
memory summary reports median and maximum peak RSS across those processes.

Quantization sensitivity variants can be selected without changing the dataset or
benchmark protocol:

```bash
# Per-tensor weights with percentile calibration
edge-ai-bench optimize-checkpoint \
  --per-tensor \
  --calibration-method percentile
```

The default is per-channel weight quantization with MinMax calibration. The result
manifest records both settings so variants remain directly comparable.
Use `--result-path` when running exploratory variants to preserve earlier manifests.

## Research protocol

The planned real-data comparison is FP32 versus ONNX Runtime graph optimization,
dynamic INT8, and static INT8 with deterministic calibration data. The benchmark will
record the CPU and software environment, model size, initialization time, warm p50/p95
latency, throughput, memory where available, and macro AUROC/F1. The model's source
checkpoint, preprocessing, PTB-XL patient-wise split, and calibration subset must be
recorded in each experiment manifest.

This is an engineering study, not clinical validation. Accuracy changes must be
interpreted with the dataset's limitations and the model's intended-use context.
