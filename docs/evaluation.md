# Evaluation

TRX-AI includes a labelled code-review benchmark (`evaluation.py`, run with `trx eval`).
For end-to-end runtime validation (routing, fallback, cache, observability), use `smoke_e2e.py`.

## Dataset

`benchmarks/cases.jsonl` holds 66 labelled cases, one JSON object per line:

| Field | Meaning |
|---|---|
| `id` | Stable case id (`case-001` ...) |
| `language` | python (37), javascript (10), java (7), c (6), go (3), sql (3) |
| `category` | bug (26), security (18), edge-case (10), performance (9), resource (3) |
| `code` | Code under review |
| `expected_debug` | Short label for the issue a correct review must name |
| `expected_fix` | Short label for the fix a correct review must suggest |

To add cases, append lines with a new unique `id`. The test suite rejects duplicate ids. It also
fails if a generic review can "pass" too many labels (see the chance floor below).

## Metrics

A review counts as a **hit** for a case when the semantic similarity between the expected label
and the review text reaches the threshold (default `0.5`, set with `--threshold`).

- **Debug detection**: share of cases where the expected issue is named.
- **Fix suggestion**: share of cases where the expected fix is suggested.
- **Chance floor**: the score that a fixed, code-agnostic review gets at the same threshold. Treat any
  result close to this floor as noise.
- **Mean similarity**: raw average similarity, shown for trend tracking only.
- **Completeness**: share of required review sections that were populated.
- **Latency**: average and p95 time per case. The review cache is cleared before each case, so
  cached answers never inflate the numbers.
- **Served by LLM vs rule fallback**: if the LLM is unreachable, the analyzer falls back to rules.
  The report flags such runs, because their numbers say nothing about model quality.

Every rate is printed with its sample size and a 95% Wilson confidence interval, for example
`72.7% (48/66, 95% CI 61.0-82.0%)`. With 66 cases, the interval is about ±11 points. Do not
quote a single percentage without its interval.

### Threshold calibration

The scorer is lexical: token overlap, synonyms and fuzzy phrase matching. It is not
embedding-based. A generic review scored against all 66 labels gave these false hits:

| Threshold | False debug hits | False fix hits |
|---|---|---|
| 0.25 | 11 / 66 | 31 / 66 |
| 0.40 | 0 / 66 | 6 / 66 |
| 0.50 | 0 / 66 | 2 / 66 |
| 0.60 | 0 / 66 | 0 / 66 |

`0.5` is the default because correct paraphrases often score between 0.3 and 0.6. Strict thresholds
under-count good reviews. Loose thresholds reward generic ones.

## Baseline comparison

Unless you pass `--no-baseline`, each case also runs through one plain prompt on the same model. The
report shows the baseline's own hit rates and latency, plus win / tie / loss counts. A win means the
TRX combined similarity is more than 0.05 above the baseline's.

## Running

```bash
trx eval                              # all cases, with baseline
trx eval --no-baseline --limit 10     # quick check
trx eval --category security --language python
trx eval --threshold 0.6 --out reports/eval/strict.txt
```

Outputs go to `reports/eval/eval_<timestamp>.txt` and `.json`. The JSON contains per-case rows plus
provenance: timestamp, git commit (with `-dirty` if uncommitted changes exist), model, dataset
fingerprint and threshold. Quote numbers together with that provenance.

## Smoke E2E

Run:

```bash
python smoke_e2e.py --disable-llm --min-score 10
```

This emits a deterministic JSON report containing:

- overall score out of 10
- category summaries
- per-test pass/fail matrix

`--min-score` makes the command exit non-zero below the given score. CI uses this check.
