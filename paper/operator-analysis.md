# Static INT8 operator interaction analysis

This analysis compares the exact artifacts used in the paper:

- `artifacts/ecg_hybrid_net_exp12_resnet_medium-int8-static.onnx`
- `artifacts/ecg_hybrid_net_exp12_resnet_medium-ort-optimized-int8-static.onnx`

The machine-readable result is [operator-analysis.json](operator-analysis.json).
It records artifact hashes, complete operator histograms, graph domains, and
the definitions used for derived counts.

## Observable graph changes

| Metric | Static INT8 | Compiled Static INT8 | Change |
|---|---:|---:|---:|
| Total graph nodes | 801 | 310 | -491 |
| Initializers | 288 | 267 | -21 |
| QuantizeLinear nodes | 135 | 13 | -122 |
| DequantizeLinear nodes | 187 | 43 | -144 |
| Total Q/DQ nodes | 322 | 56 | -266 |
| Explicit quantized operators | 0 | 53 | +53 |
| Q/DQ-backed compute operators | 59 | 6 | -53 |
| `com.microsoft` domain nodes | 0 | 31 | +31 |
| Declared FP32 fallback compute operators | 83 | 24 | -59 |
| Quantization boundary transitions | 358 | 136 | -222 |

## What this demonstrates

The Static INT8 graph is a QDQ representation: its compute nodes remain named
`Conv`, `MatMul`, `Gemm`, `Add`, and so on, with quantization represented by
separate `QuantizeLinear`/`DequantizeLinear` nodes. Consequently, its explicit
quantized-operator count is zero even though 59 compute operators are
Q/DQ-backed.

The compiled graph changes representation. It replaces 53 compute nodes with
explicit `QLinear*`/`QGemm` operators and introduces 31
`com.microsoft`-domain nodes. It also removes 266 Q/DQ nodes and 222
quantization-boundary transitions. This is direct graph evidence that the
compiled artifact is not merely the same static INT8 graph with a different
filename: quantized regions have been consolidated and the number of
conversion boundaries has been reduced.

The compiled graph still contains 24 operators from the declared compute
whitelist that are not represented as explicit quantized operators. These are
reported as fallback operators, not as proof that they execute in FP32 at
runtime. Runtime kernel dispatch would require profiling or provider-specific
execution traces; the analysis deliberately reports the weaker, observable
graph-level claim.

## Definitions

- **Explicit quantized operator:** operator type beginning with `QLinear` or
  equal to `QGemm`.
- **Q/DQ-backed compute operator:** a declared compute operator with at least
  one input produced by `DequantizeLinear`.
- **ORT-domain node:** a node with domain `com.microsoft`; this is an
  observable ORT fused/custom-domain signal, not a universal proof that every
  node is a fusion.
- **Fallback compute operator:** a non-quantized operator from the declared
  compute whitelist.
- **Quantization boundary:** a graph edge crossing between `fp32`, `qdq`, and
  `quantized` node categories.

These definitions are intentionally conservative. They show graph
transformation evidence without claiming that ONNX node labels alone reveal
the exact machine kernel selected at runtime.
