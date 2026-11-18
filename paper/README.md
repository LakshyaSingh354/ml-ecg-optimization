# Workshop paper assets

These assets are generated from the authoritative statistical five-repeat
manifest:

`results/ecg-hybrid-net-final-repeat5-pertensor-minmax-statistical.json`

The manifest is frozen by [manifest-lock.json](manifest-lock.json). The
generator verifies its SHA-256 hash and refuses to generate assets if the
source result changes.

Regenerate them after changing the manifest:

```bash
python paper/generate_assets.py
```

After a new authoritative manifest is produced:

```bash
python paper/update_lock.py results/ecg-hybrid-net-final-repeat5-pertensor-minmax-statistical.json
python paper/generate_assets.py
```

Outputs:

- `tables/final_results.csv`: machine-readable comparison table (init, RSS, bootstrap-ready fields).
- `tables/final_results.tex`: LaTeX table for the paper.
- `tables/runtime_identity.json` / `runtime_identity.txt`: CPU model and ONNX Runtime version.
- `tables/pareto_frontiers.json`: non-dominated variant sets from the manifest.
- `tables/batch_scaling.csv` and `tables/thread_sweep.csv` when the manifest includes them.
- `figures/accuracy_latency.png`: accuracy/latency trade-off with Pareto edges highlighted.
- `figures/size_throughput.png`: artifact-size/throughput trade-off.
- `figures/pareto_size_latency.png`: size–latency Pareto frontier.
- `figures/batch_scaling.png` and `figures/thread_sweep.png` when present in the manifest.
- `tables/static_int8_interaction.json`, `tables/node_mechanism.tex`, and
  `tables/calibration_comparison.tex` when schema v2 fields and comparison results exist.
- `figures/optimization_pipeline.png`: experiment design with shared evaluation block.

Figures use batch-1 median latency and throughput, with metrics and artifact sizes
taken directly from the persisted manifest. Regenerate the manifest with schema v2
(`edge-ai-bench optimize-checkpoint`) to populate bootstrap CIs, default-runtime arms,
thread sweeps, and ORT node profiles before updating `manifest-lock.json`.

The paper’s research questions and bounded claims are in
[research-framing.md](research-framing.md).

The initial paper draft is [main.tex](main.tex), with references in
[references.bib](references.bib).

The paper now includes a related-work section covering PTB-XL/ECG deep
learning, residual architectures, model compression and quantization, and
deep-learning compilers/inference runtimes. Citation entries are maintained in
`references.bib`.

The operator-level evidence for the Static INT8 versus Compiled Static INT8
interaction is documented in [operator-analysis.md](operator-analysis.md), with
the machine-readable comparison in [operator-analysis.json](operator-analysis.json).
