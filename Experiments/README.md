# Experiments

Session logs and before/after photographs from the thesis evaluation.

## Structure

```
Experiments/
├── Scenario_1/          Natural-language workspace reconfiguration
│   ├── Scenario_1.png   Workspace photo
│   └── Trial_1…7        Session transcripts (plain text)
├── Scenario_2/          Constraint violation detection
│   ├── Before.png / After.png
│   ├── Trial_1…6 + Trial_violation
│   └── Philipp/         Independent replication by a second evaluator
├── Scenario_3/          Multi-step reconfiguration with memory
│   ├── Experiment_1/ + Experiment_2/
│   └── Philipp/         Independent replication
└── Scenario_4/          PDDL-based motion sequence planning
    ├── Trial_1…3.png
    └── Whole_test_new   Full end-to-end session transcript
```

## Trial file format

Each `Trial_N` file is a plain-text session transcript captured from the CLI,
showing the exact operator–system dialogue, LLM responses, confirmed changes,
and execution output.
