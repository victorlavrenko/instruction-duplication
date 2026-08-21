# Paper-to-code implementation map

This map supports AAAI reproducibility-checklist item **4.6 = yes**. The primary implementation modules already contain concise module-level paper-section references; this table identifies the exact source entry points implementing each major method/result.

| Paper method/result | Exact source file | Primary function(s) / object(s) |
|---|---|---|
| Frozen eight-stage procedural instruction and all 2×2×2 placement conditions | `src/instruction_duplication/protocol.py` | `PROTOCOL`, `PROTOCOL_HASH`, `CONDITIONS`, `render_messages()` |
| Dataset loading, normalization, frozen question selection | `src/instruction_duplication/datasets_loader.py` | dataset loader/selection entry points in that module; pinned provenance is also frozen under `paper-run/config/` and `paper-run/data/` |
| Request construction, per-cell seed, fixed output ceiling and generation persistence | `src/instruction_duplication/generate.py`; `src/instruction_duplication/provider.py` | generation request/execution entry points; provider payload construction uses the scheduled cell seed and model ceiling |
| Visible trajectory/role recovery independent of heading surface form | `src/instruction_duplication/trajectory.py` | `recover_protocol()` → `RecoveredProtocol` |
| Deterministic judge: section completion, substantive/validated roles, contrastive discussion, premature commitment, final-answer accuracy | `src/instruction_duplication/judge.py` | `judge()`; `extract_protocol_final_answer()`; supporting deterministic role checks in the same module |
| PubMed-reference TF-IDF lexical exposure / pre-provisional stem recall | `src/instruction_duplication/lexical.py`; `src/instruction_duplication/pubmed_idf.py` | `build_reference()`, `compile_reference()`, `score_preanswer()`, `measurement_stem()`, TF-IDF token/IDF helpers |
| Paired one-copy vs two-copy inference, clustered bootstrap CIs, sign-flip tests, Holm correction and factorial terms | `src/instruction_duplication/stats.py` | `bootstrap_ci()`, `sign_flip_p_value()`, `holm_adjust()`, `factorial_contrast()`, `build_analysis()` |
| Generic blinded human-audit extraction infrastructure | `src/instruction_duplication/audit.py` | `export_blinded_matched_pairs()`, `human_audit_schema()`, blinding/highlighting helpers |
| AAAI effect-focused human validation: prospective exact-binomial design, stable-hash sampling, A/B blinding, freeze, score, package | `tools/aaai27_human_validation.py` | `exact_design()`, `choose_hash_sample()`, `allocate_primary_by_pattern()`, `build_task()`, `assign_treatment_sides()`, `export_command()`, `freeze_command()`, `score_command()`, `package_command()` |
| AAAI robustness analysis on frozen generations/judgments | `tools/aaai27_robustness_analysis.py` | `paired_contrast()`, `holm()`, `additive_residuals()`, `fixed_effect_length_adjustment()`, `matched_pair_length_adjustment()`, `status_summary()`, `accuracy_counts()`, `main()` |
| Reproducibility/supplement construction | `scripts/build_aaai_supplement.py` | supplement-building entry point; packaging only, not scientific inference |

## Paper-section references already present in core implementation

The core files above contain concise module docstrings tying implementation to the relevant paper sections, including: `protocol.py` (factorial conditions/procedural instruction), `trajectory.py` and `judge.py` (Measurements and Eligibility), `lexical.py` (Pre-provisional TF-IDF recall), `stats.py` (Inference and Results), `audit.py` and `aaai27_human_validation.py` (Inference and Audit Design / Evaluation Validity), and `aaai27_robustness_analysis.py` (Results / Limitations and Reproducibility).

No executable behavior was changed to create this map.
