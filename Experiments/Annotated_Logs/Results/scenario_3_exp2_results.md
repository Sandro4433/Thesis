# Scenario 3 — Experiment 2 (Kitting)

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Ambiguity Handling Trials

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T2 | R + O | 14 | 7 | 7 | 2 | 2 | 2 | 1 | 3 | 3 | 2 | 0 | 4 | Y |
| T3 | I + S | 12 | 6 | 6 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 0 | 4 | Y |
| T4 | O + S | 18 | 9 | 9 | 3 | 2 | 3 | 1 | 2 | 4 | 3 | 0 | 6 | N |
| T5 | R + I + O | 12 | 6 | 6 | 1 | 1 | 3 | 1 | 2 | 2 | 3 | 0 | 4 | Y |
| T6 | R+I+O+S | 14 | 7 | 7 | 1 | 1 | 4 | 1 | 2 | 2 | 4 | 0 | 6 | Y |

## Constraint Violation Trial

> Each CV sub-trial tests one violation type in isolation.

| CV | Description | Total | USR | AST | USR-INST | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Detected |
|----|-------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|
| CV-1 | Kit recipe exceeds kit capacity | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | Y |
| CV-2 | Batch size exceeds kit count    | 3 | 1 | 2 | 1 | 1 | 0 | 0 | 1 | Y |
| CV-3 | Insufficient parts for recipe   | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | Y |

