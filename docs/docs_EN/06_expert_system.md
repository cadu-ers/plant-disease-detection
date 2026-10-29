[← back to README](../../README.en.md)

# Expert system

<!-- When editing this file, also update ../06_sistema_especialista.md -->

[🇧🇷 Português](../06_sistema_especialista.md) | **🇺🇸 English**

The project also includes an expert system in [`src/ia/expert_system/`](../../src/ia/expert_system/).
It receives facts about the plant and applies **IF... THEN...** rules to
rank diseases, compute certainty factors and explain the result.

The knowledge base is in [`src/ia/expert_system/knowledge_base.yaml`](../../src/ia/expert_system/knowledge_base.yaml) and
contains the symptoms, the 13 diseases, the rules, the certainty factors and
general recommendations. The engine can also receive probabilities produced
by the CNN through the `cnn_prior` field.

## Running an example diagnosis

From the project root, with `src/` on the `PYTHONPATH` (see
[manual execution](07_how_to_reproduce.md#manual-execution-script-by-script)), run:

```bash
export PYTHONPATH=src            # in PowerShell: $env:PYTHONPATH = "src"
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/late_blight_tomate.json
```

To run without using the CNN evidence:

```bash
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/late_blight_tomate.json --sem-cnn
```

To use interactive mode:

```bash
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/ambiguo_early_target.json --interativo
```

The output shows the main hypothesis, the evidence from the rules, the
alternative hypotheses, the pending symptoms and general recommendations.
The value displayed as **CF** is a certainty factor of the system, not a
calibrated statistical probability.

Currently, the expert system can be run independently with facts provided
in JSON. Automatic integration between image, segmentation, measurement
extraction, CNN and expert system is a future stage of the project.

## Running the tests

```bash
python -m pytest -q
```

The tests cover loading the knowledge base, class mapping, certainty factor
computation, the diagnosis rules, the explanations and interactive mode.

The module's own documentation: [`src/ia/expert_system/README.md`](../../src/ia/expert_system/README.md) (in Portuguese).
