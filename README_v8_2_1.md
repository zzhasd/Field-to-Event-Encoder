# Lab Ontology Comprehensive Comparison v8.2.1

This package keeps the core no-leak encoder in `University_Field_to_Event_Encoder.py` unchanged and modifies only the experimental comparison layer in `lab_comprehensive_comparison.py`.

## What changed from v8.2.0

- `raw_field_image + VLM` now defaults to a temporal contact sheet instead of a single last frame.
- New VLM controls:
  - `--vlm-frame-mode last|sampled|all`
  - `--vlm-num-frames 8`
- LLM/VLM prompts no longer receive evaluator-side `scenario_notes`; accident labels and scenario design notes remain only in the evaluator.
- Added `--profile hard` for ambiguous/composite/low-SNR/missing/unseen cases.
- Added hard-case summary columns: `hard_case_accuracy`, `composite_accuracy`, `low_snr_accuracy`, and `expected_review_success`.
- Token accounting now supports DashScope/OpenAI-style usage aliases and fills a reproducible proxy estimate when the API response omits usage.
- Default text LLM model is now `qwen3.6-flash`; default VLM model remains `qwen3-vl-flash`.

## Recommended paper command

PowerShell:

```powershell
$env:DASHSCOPE_API_KEY = "YOUR_KEY"
python lab_comprehensive_comparison.py `
  --profile paper `
  --api-mode api `
  --seeds 1 `
  --steps 170 `
  --llm-model qwen3.6-flash `
  --vlm-model qwen3-vl-flash `
  --vlm-frame-mode sampled `
  --vlm-num-frames 8 `
  --out-dir outputs/comprehensive_paper_qwen_flash_v8_2_1
```

Linux/macOS:

```bash
export DASHSCOPE_API_KEY="YOUR_KEY"
python lab_comprehensive_comparison.py \
  --profile paper \
  --api-mode api \
  --seeds 1 \
  --steps 170 \
  --llm-model qwen3.6-flash \
  --vlm-model qwen3-vl-flash \
  --vlm-frame-mode sampled \
  --vlm-num-frames 8 \
  --out-dir outputs/comprehensive_paper_qwen_flash_v8_2_1
```

## Strong hard-case check

```bash
python lab_comprehensive_comparison.py \
  --profile hard \
  --api-mode api \
  --seeds 1 \
  --steps 170 \
  --llm-model qwen3.6-flash \
  --vlm-model qwen3-vl-flash \
  --vlm-frame-mode sampled \
  --vlm-num-frames 8 \
  --out-dir outputs/comprehensive_hard_qwen_flash_v8_2_1
```

## No-leak rule

The F2E encoder is still called only as:

```python
encoder.update(t, current_fields)
```

Clean backgrounds, GT masks, injected incident labels, and accident labels are used only by the simulator/evaluator, never by the encoder.
