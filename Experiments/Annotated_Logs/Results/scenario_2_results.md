# Scenario 2 — Workspace State-Description Ambiguity

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Ambiguity Handling Trials

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 5 | 3 | 2 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | Y |
| T2 | R + O | 9 | 4 | 5 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T3 | I + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | Y |
| T4 | O + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | Y |
| T5 | R + I + O | 9 | 4 | 5 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T6 | R + I + O + S | 9 | 4 | 5 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| Volunteer | R | 8 | 4 | 4 | 2 | 1 | 0 | 1 | 1 | 2 | 0 | 0 | 2 | Y |

## Constraint Violation Trial

> Each CV sub-trial tests one violation type in isolation.

| CV | Description | Total | USR | AST | USR-INST | USR-CLAR | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Detected |
|----|-------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|
| CV-1 | Wrong part count removed | 3 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 1 | Y |
| CV-2 | Wrong part count added   | 3 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 1 | Y |
| CV-3 | Wrong part moved         | 3 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 1 | Y |
| CV-4 | Wrong part removed       | 3 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 1 | Y |
| CV-5 | Contradictory scene claim| 9 | 4 | 5 | 1 | 2 | 1 | 1 | 2 | 0 | N |
| CV-6 | Non-existent container + wrong colors + off-topic | 11 | 5 | 6 | 3 | 0 | 3 | 0 | 0 | 2 | Y |
