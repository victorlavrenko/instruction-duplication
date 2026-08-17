# Regeneration estimate

## Full default run

The default confirmatory run contains 300 questions × 7 models × 8 conditions = 16,800 cells.

The preceding 16,800-cell run used:

- 13,107,669 generation input tokens;
- 14,168,445 generation output tokens;
- $12.0773 in accounted generation spend;
- $0.0165 in preflight spend.

The simple 3.0.5 prompt is unchanged from 3.0.4 and is shorter than the XML protocol. Across the
same 300 questions, its rendered input character count is lower by approximately:

- 21.2% for zero-copy cells;
- 29.0% for one-copy cells;
- 30.7% for two-copy cells;
- 31.5% for three-copy cells.

Weighting the eight conditions equally implies approximately 9.2 million input tokens if the
same routes tokenize the new prompts in the same character-to-token ratio. Output length is less
predictable because the model may change its prose even though the requested reasoning roles are
the same.

Using the preceding run's realized per-route prices and retry pattern gives:

| Assumed output-token change | Estimated generation spend |
|---:|---:|
| −5% | $10.48 |
| 0% | $10.91 |
| +5% | $11.34 |

The practical planning estimate is therefore **$10.5–$11.4**, with **$15** as a conservative manual
cumulative cap for ordinary retry variation. Live preflight now includes two functional requests
plus up to eight simultaneous capacity requests per model. At routes comparable to the preceding
run, this is expected to add roughly **$0.06–$0.10** rather than the preceding $0.0165 preflight
cost. It may select different providers or calibrate different prices, so the program's
route-aware automatic cap remains the authoritative safety calculation at run time.

For a 10-question-per-dataset smoke run, the generation component scales to roughly **$1.0–$1.2**
under comparable routing.

## Commands

For a clean 3.0.5 regeneration, use a fresh workspace. A partially generated 3.0.4 workspace may instead be resumed directly by 3.0.5; 3.0.1–3.0.3 remain incompatible because the 3.0.4 Mistral Large ceiling changed the frozen model/smoke identity:

```bash
instruction-duplication 10 --workspace run-simple-headings-smoke --max-cost 2
```

After checking the smoke generations, run the independent full experiment while excluding prior
question ledgers:

```bash
instruction-duplication 100 \
  --workspace run-simple-headings-2026-08 \
  --exclude-workspace ../instruction-duplication-2.2.3/run-2026-08-07 \
  --exclude-workspace ../instruction-duplication-2.4.0/run-2026-08-09 \
  --exclude-workspace ../instruction-duplication-2.8.2/run-2026-08-10 \
  --max-cost 15
```

Adjust the example paths to the actual locations of the earlier workspaces. Omitting `--max-cost`
uses the program's live route-aware automatic cap.
