# Scenario 3 — Experiment 1 (Sorting)

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Turn-Type Breakdown

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 10 | 5 | 5 | 2 | 2 | 0 | 1 | 3 | 2 | 0 | 0 | 0 | Y |
| T2 | R + O | 12 | 6 | 6 | 1 | 3 | 1 | 1 | 2 | 3 | 1 | 0 | 4 | Y |
| T3 | I + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T4 | O + S | 10 | 5 | 5 | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | Y |
| T5 | R + I + O | 8 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T6 | R + I + O + S | 13 | 6 | 7 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 1 | 4 | Y |
| T-Viol | Constraint Violation Handling (CV) | 13 | 5 | 8 | 5 | 0 | 0 | 0 | 2 | 2 | 0 | 4 | 0 | N — CV-2 not detected |


