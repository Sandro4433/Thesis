# Scenario 3 — Experiment 1 (Sorting)

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Ambiguity Handling Trials

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 10 | 5 | 5 | 2 | 2 | 0 | 1 | 3 | 2 | 0 | 0 | 0 | Y |
| T2 | R + O | 14 | 7 | 7 | 2 | 3 | 1 | 1 | 3 | 3 | 1 | 0 | 4 | Y |
| T3 | I + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T4 | O + S | 10 | 5 | 5 | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | Y |
| T5 | R + I + O | 9 | 4 | 5 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | Y |
| T6 | R+I+O+S | 14 | 7 | 7 | 1 | 2 | 3 | 1 | 2 | 3 | 2 | 1 | 4 | Y |

## Constraint Violation Trial

> Each CV sub-trial tests one violation type in isolation.

| CV | Description | Total | USR | AST | USR-INST | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Detected |
|----|-------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|
| CV-1 | Non-existent part color (pink) | 3 | 1 | 2 | 1 | 1 | 0 | 0 | 1 | Y |
| CV-2 | Conflicting color+fragility rules | 2 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | **N** |
| CV-3 | Capacity exceeded | 4 | 1 | 3 | 1 | 1 | 1 | 0 | 1 | Y |
| CV-4 | Non-existent container | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | Y |
| CV-5 | Conflicting priority | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | Y |


