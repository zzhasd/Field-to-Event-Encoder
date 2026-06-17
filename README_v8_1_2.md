# lab_ontology_encoder_v8_1_2

This is a small runner-only update on top of v8.1.1.

## What changed

- `University_Field_to_Event_Encoder.py` is unchanged from v8.1.1.
- `lab_ontologyEvaluation_and_ablation.py` is updated to v8.1.2.
- Added a dependency-free progress bar with elapsed time, ETA, rate, and current seed/stress/variant/case status.
- Added progress-related fields to `run_config.json`.

## Quick check

```bash
python lab_ontologyEvaluation_and_ablation.py --profile quick --out-dir outputs/quick_v8_1_2
```

## Recommended paper experiment

```bash
python lab_ontologyEvaluation_and_ablation.py --profile paper --seeds 10 --steps 170 --stress-set core --variants all --out-dir outputs/paper10_core_all_v8_1_2
```

## Full appendix-scale stress experiment

```bash
python lab_ontologyEvaluation_and_ablation.py --profile paper --seeds 10 --steps 170 --stress-set all --variants all --out-dir outputs/paper10_allstress_all_v8_1_2
```

## Progress controls

Progress is on by default. Disable it with:

```bash
python lab_ontologyEvaluation_and_ablation.py --profile paper --progress off --out-dir outputs/no_progress
```

Change update frequency or bar width:

```bash
python lab_ontologyEvaluation_and_ablation.py --profile paper --progress-interval 5 --progress-width 40 --out-dir outputs/paper_progress
```
