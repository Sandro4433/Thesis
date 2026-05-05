# Scenario 1 — Baseline Reconfiguration

**Turn-type key:** USR-INST = instruction | USR-CONF = confirmation | USR-CLAR = clarification response | USR-DONE = session close | AST-OPEN = open/continuation prompt | AST-PROP = change proposal | AST-CLAR = clarification question | AST-VIOL = constraint violation response | AST-IMG = image-check prompt

## Turn-Type Breakdown

| Trial | Session | Attribute Type | Total | USR | AST | USR-INST | USR-CONF | USR-CLAR | USR-DONE | AST-OPEN | AST-PROP | AST-CLAR | Extra Turns | Config Correct |
|-------|---------|----------------|-------|-----|-----|----------|----------|----------|----------|----------|----------|----------|-------------|----------------|
| T1 | S1 | Operational Mode | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Operational Mode | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T2 | S1 | Object Role | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Object Role | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S3 | Object Role | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T3 | S1 | Kit Recipe | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Kit Recipe | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S3 | Kit Recipe | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T4 | S1 | Priority | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Priority | 8 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 2 | Y |
|  | S3 | Priority | 8 | 4 | 4 | 1 | 2 | 0 | 1 | 2 | 2 | 0 | 0 | Y |
|  | S4 | Priority | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S5 | Priority | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T5 | S1 | Part Compatibility | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Part Compatibility | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S3 | Sorting / Compatibility | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T6 | S1 | Batch Size | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Batch Size | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
| T7 | S1 | Part Fragility | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |
|  | S2 | Part Fragility | 6 | 3 | 3 | 1 | 1 | 0 | 1 | 2 | 1 | 0 | 0 | Y |


