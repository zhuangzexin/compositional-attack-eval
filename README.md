# Compositional Attack Transferability Evaluation

Code and data for "Are Compositional Attacks Compositional? Measuring Structure/Content Separability in LLM Safety Bypasses" (ICML 2026 CompLearn Workshop).

## Contents

- `attack_templates.py` — Attack structure templates (decomposition, role-based, escalation) with placeholder content slots. Specific harmful scenarios are withheld.
- - `evaluate.py` — Inference and dual-metric evaluation pipeline (keyword heuristic + LLM-as-judge consensus).
  - - `results_summary.csv` — Aggregated bypass rates with bootstrap 95% confidence intervals for all model/attack/domain combinations.
   
    - ## Usage
   
    - ```bash
      pip install torch transformers numpy
      python evaluate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --attack decomposition
      ```

      Requires a CUDA-capable GPU (tested on T4 16GB). Models are downloaded from Hugging Face on first run.

      ## Results

      See `results_summary.csv` for full results. Each row contains: model, domain type (core/transfer), attack type, sample size, keyword bypass rate, consensus bypass rate, and bootstrap 95% CIs.

      ## Ethical Note

      We release attack structure templates and evaluation code to support reproducibility. We do not release specific harmful-action scenarios or raw model outputs.
