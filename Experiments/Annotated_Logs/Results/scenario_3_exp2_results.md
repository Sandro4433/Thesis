# Scenario 3 — Experiment 2 (Kitting)

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Turn-Type Breakdown

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T2 | R + O | 12 | 6 | 6 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 0 | 4 | Y |
| T3 | I + S | 12 | 6 | 6 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | 0 | 4 | Y |
| T4 | O + S | 16 | 8 | 8 | 3 | 2 | 2 | 1 | 3 | 2 | 3 | 0 | 6 | N — incomplete first session |
| T5 | R + I + O | 12 | 6 | 6 | 1 | 1 | 3 | 1 | 2 | 1 | 3 | 0 | 4 | Y |
| T6 | R + I + O + S | 12 | 6 | 6 | 1 | 1 | 3 | 1 | 2 | 1 | 3 | 0 | 6 | Y |
| T-Viol | Constraint Violation Handling (CV) | 7 | 3 | 4 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | N/A |


