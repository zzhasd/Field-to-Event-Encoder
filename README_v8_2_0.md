# lab_ontology_encoder v8.2.0

This release adds `lab_comprehensive_comparison.py`, a three-layer comparison runner for proving the value of the Field-to-Event (F2E) encoder and the F2E+LLM system.

The core no-leak encoder stays in `University_Field_to_Event_Encoder.py`. The new comprehensive script is evaluation-only and calls the core encoder through `encoder.update(t, current_fields)`.

## Files

- `University_Field_to_Event_Encoder.py` — no-leak core encoder.
- `lab_ontologyEvaluation_and_ablation.py` — previous ontology evaluation and ablation runner with progress bar.
- `lab_comprehensive_comparison.py` — new three-layer comparison runner.
- `outputs/comprehensive_quick_offline/` — smoke-test outputs generated without API calls.

## API key setup

Do not hardcode API keys in code. Set it in the shell:

PowerShell:

```powershell
$env:DASHSCOPE_API_KEY = "your_key_here"
```

Optional region override:

```powershell
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Other documented region endpoints include Singapore and Virginia. Use the endpoint that matches your key region.

## Thinking mode policy

For the main paper comparison, keep thinking mode **off**:

```bash
--enable-thinking  # omit this for main experiments
```

Reason: the paper compares diagnosis accuracy, token consumption, and response speed. Thinking mode changes both latency and token/cost behavior, so enabling it only for some methods would be unfair. If you want, run a supplementary sensitivity test with `--enable-thinking` for every LLM/VLM method.

## Quick offline smoke test

This runs all three layers without calling DashScope:

```bash
python lab_comprehensive_comparison.py --profile quick --api-mode offline --out-dir outputs/comprehensive_quick_offline
```

## Main paper comparison with DashScope API

This uses `qwen3-vl-flash` for both text/event/matrix LLM comparisons and raw-image VLM comparison, with thinking disabled:

```bash
python lab_comprehensive_comparison.py \
  --profile paper \
  --api-mode api \
  --seeds 10 \
  --steps 170 \
  --llm-model qwen3-vl-flash \
  --vlm-model qwen3-vl-flash \
  --out-dir outputs/comprehensive_paper_qwen3vl_flash_v8_2_0
```

## Supplementary thinking-mode sensitivity test

Run this only as an appendix/sensitivity experiment:

```bash
python lab_comprehensive_comparison.py \
  --profile paper \
  --api-mode api \
  --seeds 10 \
  --steps 170 \
  --llm-model qwen3-vl-flash \
  --vlm-model qwen3-vl-flash \
  --enable-thinking \
  --out-dir outputs/comprehensive_paper_qwen3vl_flash_thinking_v8_2_0
```

## Layer-specific commands

Layer 1 only, local detection comparison:

```bash
python lab_comprehensive_comparison.py --profile paper --layers 1 --seeds 10 --steps 170 --api-mode offline --out-dir outputs/layer1_detection_v8_2_0
```

Layers 2 and 3 with API:

```bash
python lab_comprehensive_comparison.py --profile paper --layers 2,3 --api-mode api --seeds 10 --steps 170 --out-dir outputs/layer23_api_v8_2_0
```

## Outputs

- `layer1_detection_records.csv`
- `layer1_detection_summary.csv`
- `layer2_diagnosis_records.csv`
- `layer2_diagnosis_summary.csv`
- `layer3_action_records.csv`
- `layer3_action_summary.csv`
- `comprehensive_comparison_results.json`
- `run_config.json`
- `dashscope_cache.json` if API calls are made
- `field_images/` for raw image + VLM runs

## Three-layer evidence design

### Layer 1: low-level detection

Compares:

- threshold connected components
- CUSUM/EWMA
- vision-style heatmap detector
- F2E encoder

Metrics:

- detection F1
- IoU
- centroid error
- normal false positive rate
- extra event false positive rate
- latency

### Layer 2: high-level diagnosis

Compares:

- threshold events + rules
- threshold events + LLM
- raw matrix summary + LLM
- raw field image + VLM
- F2E events + rules
- F2E events + LLM

Metrics:

- accident classification accuracy
- macro-F1
- ambiguous-case accuracy
- unseen-combination accuracy
- explanation correctness
- token consumption
- API latency

### Layer 3: action decision

Compares the same high-level methods on:

- actionable precision
- actionable recall
- false positives per active step
- review precision
- resampling success rate
- diagnosis latency/cost

## Designed confusing scenarios

- fire vs electrical overheating
- water leak vs steam leak
- CO2 accumulation vs dust pollution
- composite anomaly
- low-SNR anomaly
- missing/low-confidence region
- unseen pressure/AQI/humidity combination
- normal no-incident case
