# Reality Debugger Sample Test Cases

These are sample inputs and expected output characteristics. Actual AI text can vary, but the strict section format should remain identical.

Required sections in every output:
- `[INPUT ANALYSIS]`
- `[BUGS DETECTED]`
- `[FIX SUGGESTIONS]`
- `[PSEUDO CODE]`
- `[CONFIDENCE SCORE]`
- `[CONFIDENCE EXPLANATION]`

## Test Case 1
Input:
`I failed exam twice and feel tired every day.`

Expected key signals:
- `[BUGS DETECTED]` should include items about `lack of practice` and `poor sleep cycle`.
- `[FIX SUGGESTIONS]` should include a practice plan and sleep improvement action.
- `[CONFIDENCE SCORE]` should be a number between `0%` and `100%`.

## Test Case 2
Input:
`I keep procrastinating on my final year project and miss deadlines.`

Expected key signals:
- `[BUGS DETECTED]` should mention task avoidance or procrastination loop.
- `[FIX SUGGESTIONS]` should suggest breaking work into smaller focused blocks.
- `[PSEUDO CODE]` should contain at least one simple `if ... then ... else ...` line.

## Test Case 3
Input:
`My startup is growing, but our team communication is chaotic and outputs are inconsistent.`

Expected key signals:
- `[INPUT ANALYSIS]` should identify coordination and process gaps.
- `[FIX SUGGESTIONS]` should provide practical workflow/process actions.
- Output must preserve all required sections in strict order.

## Test Case 4 (Predict Mode)
Mode:
`mode predict`

Input:
`I keep delaying my project tasks and my stress is increasing every week.`

Expected key signals:
- `[INPUT ANALYSIS]` should mention prediction mode or trajectory estimation.
- `[BUGS DETECTED]` should include delivery or stress risk over time.
- `[PSEUDO CODE]` should include a future-outcome condition (for example, persistence over weeks).
- `[CONFIDENCE EXPLANATION]` should explain why the score was assigned.
