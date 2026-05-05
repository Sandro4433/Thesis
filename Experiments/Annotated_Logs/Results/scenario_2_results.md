# Scenario 2 — Workspace State-Description Ambiguity

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Turn-Type Breakdown

| Trial | Ambiguity Profile | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | AST-VIOL | Extra Turns | Config Correct |
|-------|-------------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | None | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T2 | R + O | 8 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T3 | I + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T4 | O + S | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | Y |
| T5 | R + I + O | 8 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T6 | R + I + O + S | 8 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | Y |
| T-Viol | Constraint Violation Handling (CV) | 32 | 13 | 19 | 11 | 0 | 2 | 0 | 8 | 2 | 2 | 6 | 0 | N/A |
| Volunteer | R (typo "zes" instead of "yes" causes silent config loss; "exchanged" causes swap misinterpretation) | 8 | 4 | 4 | 2 | 1 | 0 | 1 | 2 | 2 | 0 | 0 | 2 | Y |


