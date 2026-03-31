# Evaluation

TRX-AI includes an evaluation and benchmarking module in `evaluation.py`.

## Dataset

The benchmark dataset contains 10 representative Python code-review cases spanning:

- logic bugs
- index errors
- recursion inefficiency
- security patterns
- input-validation edge cases

Each case defines:

- input_text
- expected_debug
- expected_fix
- category

## Metrics

The evaluator computes:

- **Accuracy**: expected debug issue matched in output
- **Fix Quality**: expected fix suggestion matched
- **Avg Response Time**: mean per-case processing time
- **Completeness**: percentage of required sections populated

## Baseline Comparison

TRX-AI output is compared against a simple baseline LLM prompt.

Comparison highlights:

- match quality difference (debug + fix)
- response-time trends
- structured completeness advantage

## Example Results

```text
Accuracy: 80%
Fix Quality: 75%
Avg Response Time: 5.8s
Completeness: 90%
```

## Artifacts

- `evaluation_report.txt`
- benchmark terminal output
- accuracy graph image
- evaluation screenshot
