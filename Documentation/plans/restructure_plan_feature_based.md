# Feature-Based Restructure Plan (Preview Only)

Status: No files moved yet. This is a proposed plan for approval.

## Objectives

1. Move preprocessing, inference, and explainability code into separate feature folders under a src-style layout.
2. Reorganize documentation and image assets into a clean Documentation structure.
3. Update imports, config paths, markdown links, and command examples to match new paths.
4. Validate that old path references are removed from Python, Markdown, TXT, and JSON files.

## Proposed Target Layout

```text
src/
  qids/
    preprocessing/
      make_all_classes_input.py
      size.py
    inference/
      run_pipeline.py
      live_inference.py
      make_sample_excel.py
      hybrid_assembly.py
      pipeline_utils.py
      plotting.py
    explainability/
      explainability_dashboard.py
      shap_pipeline.py
      generate_classwise_shap_json.py
      app.py
      lime_shared_helper.py

scripts/
  run_pipeline.py
  live_inference.py
  generate_classwise_shap_json.py
  explainability_dashboard.py
  dev/
    _tmp_schema_inspect.py
    _tmp_schema_inspect2.py

configs/
  pipeline/
    config.json

artifacts/
  inference/
    all_classes_input.csv
    sample_input.csv
    live_predictions.json
    lime_local_explanations.json
    sample_predictions.xlsx
  plots/

Documentation/
  commands/
    command.txt
  guides/
    HACKATHON_PPT_GUIDE.md
    PreProcessingGuide.md
  design/
    activity_diagrams.md
    class_diagram_design.md
    sequence_diagrams.md
    use_case_actor_explanation.md
  reports/
    Stage1_Assessment_Report.txt
    Stage1_Design_Specification_Preprocessing_Part_1.txt
    Stage2_Evaluation_Report.txt
    Stage2_Evaluation_Report_v2.txt
    Stage3_VAE_Design.txt
  references/
    Stage1_Stage2_TechRef.txt
    Stage2_Technical_Reference.txt
    problem_statement.txt
  diagrams/
    architecture/
      Arc_diag.png
      Arc_diag.pdf
    class/
      class_diag.png
    use_case/
      use_case_diag.png
    activity/
      (from Images/Activity_diag/*)
    sequence/
      (from Images/Seq_diag/*)
```

## Planned Moves (Old -> New)

### Code Moves

- pipeline/run_pipeline.py -> src/qids/inference/run_pipeline.py
- pipeline/live_inference.py -> src/qids/inference/live_inference.py
- pipeline/make_sample_excel.py -> src/qids/inference/make_sample_excel.py
- pipeline/pipeline_utils.py -> src/qids/inference/pipeline_utils.py
- pipeline/plotting.py -> src/qids/inference/plotting.py
- hybrid_assembly.py -> src/qids/inference/hybrid_assembly.py
- pipeline/make_all_classes_input.py -> src/qids/preprocessing/make_all_classes_input.py
- PreProcessing/size.py -> src/qids/preprocessing/size.py
- qids_shap/shap_pipeline.py -> src/qids/explainability/shap_pipeline.py
- qids_shap/generate_classwise_shap_json.py -> src/qids/explainability/generate_classwise_shap_json.py
- qids_shap/app.py -> src/qids/explainability/app.py
- pipeline/explainability_dashboard.py -> src/qids/explainability/explainability_dashboard.py
- lime_shared_helper.py -> src/qids/explainability/lime_shared_helper.py
- _tmp_schema_inspect.py -> scripts/dev/_tmp_schema_inspect.py
- _tmp_schema_inspect2.py -> scripts/dev/_tmp_schema_inspect2.py

### Config and Runtime Files

- pipeline/config.json -> configs/pipeline/config.json
- pipeline/all_classes_input.csv -> artifacts/inference/all_classes_input.csv
- pipeline/sample_input.csv -> artifacts/inference/sample_input.csv
- pipeline/live_predictions.json -> artifacts/inference/live_predictions.json
- pipeline/lime_local_explanations.json -> artifacts/inference/lime_local_explanations.json
- pipeline/sample_predictions.xlsx -> artifacts/inference/sample_predictions.xlsx
- pipeline/plots/ -> artifacts/plots/

### Documentation and Image Moves

- Documentation/command.txt -> Documentation/commands/command.txt
- Documentation/HACKATHON_PPT_GUIDE.md -> Documentation/guides/HACKATHON_PPT_GUIDE.md
- Documentation/PreProcessingGuide.md -> Documentation/guides/PreProcessingGuide.md
- Documentation/activity_diagrams.md -> Documentation/design/activity_diagrams.md
- Documentation/class_diagram_design.md -> Documentation/design/class_diagram_design.md
- Documentation/sequence_diagrams.md -> Documentation/design/sequence_diagrams.md
- Documentation/use_case_actor_explanation.md -> Documentation/design/use_case_actor_explanation.md
- Documentation/Stage1_Assessment_Report.txt -> Documentation/reports/Stage1_Assessment_Report.txt
- Documentation/Stage1_Design_Specification_Preprocessing_Part_1.txt -> Documentation/reports/Stage1_Design_Specification_Preprocessing_Part_1.txt
- Documentation/Stage2_Evaluation_Report.txt -> Documentation/reports/Stage2_Evaluation_Report.txt
- Documentation/Stage2_Evaluation_Report_v2.txt -> Documentation/reports/Stage2_Evaluation_Report_v2.txt
- Documentation/Stage3_VAE_Design.txt -> Documentation/reports/Stage3_VAE_Design.txt
- Documentation/Stage1_Stage2_TechRef.txt -> Documentation/references/Stage1_Stage2_TechRef.txt
- Documentation/Stage2_Technical_Reference.txt -> Documentation/references/Stage2_Technical_Reference.txt
- Documentation/problem_statement.txt -> Documentation/references/problem_statement.txt
- Documentation/Arc_diag.pdf -> Documentation/diagrams/architecture/Arc_diag.pdf
- Images/Arc_diag.png -> Documentation/diagrams/architecture/Arc_diag.png
- Images/class_diag.png -> Documentation/diagrams/class/class_diag.png
- Images/use_case_diag.png -> Documentation/diagrams/use_case/use_case_diag.png
- Images/Activity_diag/* -> Documentation/diagrams/activity/*
- Images/Seq_diag/* -> Documentation/diagrams/sequence/*

## Planned Reference Patches

### Python Import and Script Path Patches

- src/qids/inference/run_pipeline.py
  - update imports to package form:
    - from qids.inference.pipeline_utils import ...
    - from qids.inference.plotting import ...
  - update default config path:
    - configs/pipeline/config.json
  - update invoked script paths:
    - src/qids/inference/live_inference.py
    - src/qids/inference/hybrid_assembly.py

- src/qids/inference/live_inference.py
  - from qids.explainability.lime_shared_helper import ...
  - from qids.inference.pipeline_utils import ...
  - default config path -> configs/pipeline/config.json

- src/qids/inference/make_sample_excel.py
  - import qids.inference.live_inference as li
  - from qids.inference.pipeline_utils import ...
  - default config path -> configs/pipeline/config.json

- src/qids/explainability/shap_pipeline.py
  - from qids.explainability.lime_shared_helper import ...

- scripts/*.py wrappers
  - thin compatibility entry points that call main() from src/qids/* modules.

### Config Patches

- configs/pipeline/config.json
  - keep model/data artifact paths intact unless moved.
  - update output paths:
    - artifacts/inference/live_predictions.json
    - artifacts/inference/lime_local_explanations.json
    - artifacts/plots
  - update input_glob:
    - artifacts/inference/all_classes_input.csv

### Markdown and Command Patches

- Documentation/commands/command.txt
  - update script invocations to either:
    - python scripts/run_pipeline.py --config configs/pipeline/config.json
    - python -m qids.inference.run_pipeline --config configs/pipeline/config.json

- Documentation/guides/HACKATHON_PPT_GUIDE.md
  - update image links:
    - Images/Arc_diag.png -> Documentation/diagrams/architecture/Arc_diag.png
    - Images/class_diag.png -> Documentation/diagrams/class/class_diag.png
    - Images/use_case_diag.png -> Documentation/diagrams/use_case/use_case_diag.png

- pipeline/README.md and qids_shap/README.md
  - update script and file paths to new src/configs/artifacts layout (or move these READMEs into Documentation/guides and patch there).

## Validation Plan (After Move)

1. Path existence checks for every moved file and target folder.
2. Import sanity check:
   - python -m compileall src/qids scripts
3. Command sanity check:
   - python scripts/run_pipeline.py --help
   - python scripts/live_inference.py --help
   - python scripts/generate_classwise_shap_json.py --help
4. Search for stale old paths:
   - pipeline/run_pipeline.py
   - pipeline/live_inference.py
   - qids_shap/generate_classwise_shap_json.py
   - pipeline/explainability_dashboard.py
   - Images/Arc_diag.png
5. Verify Documentation command examples run with updated paths.

## Execution Mode

Planned execution is preview-first and batch-wise.

Batch 1: Create src/qids package structure and scripts wrappers.
Batch 2: Move inference/preprocessing/explainability Python files and patch imports.
Batch 3: Move config/runtime artifacts and patch JSON/path defaults.
Batch 4: Reorganize Documentation and Images and patch markdown/txt links.
Batch 5: Run validation searches and compile checks.

## Approval Needed

Reply with one of the following to proceed:

- Approve plan as-is
- Approve with changes (list modifications)
- Cancel