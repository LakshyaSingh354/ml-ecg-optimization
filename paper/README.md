# Workshop paper assets

These assets are generated from the authoritative five-repeat manifest:

`results/ecg-hybrid-net-final-repeat5-pertensor-minmax.json`

The manifest is frozen by [manifest-lock.json](manifest-lock.json). The
generator verifies its SHA-256 hash and refuses to generate assets if the
source result changes.

Regenerate them after changing the manifest:

```bash
python paper/generate_assets.py
```

Outputs:

- `tables/final_results.csv`: machine-readable comparison table.
- `tables/final_results.tex`: LaTeX table for the paper.
- `figures/accuracy_latency.png`: accuracy/latency trade-off.
- `figures/size_throughput.png`: artifact-size/throughput trade-off.
- `figures/optimization_pipeline.png`: four-condition experiment design.

The figures use batch-1 median latency and throughput, with metrics and
artifact sizes taken directly from the persisted manifest. Memory values remain
in the CSV for later memory-specific plots.

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
