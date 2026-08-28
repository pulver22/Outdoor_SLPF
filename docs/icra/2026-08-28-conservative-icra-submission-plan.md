# Evidence-First ICRA Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, anonymous, eight-page ICRA 2027 submission around the validated GNSS-aided semantic-wall particle filter without promoting unsupported row-mixture claims.

**Architecture:** Freeze one baseline configuration and generate a canonical evidence bundle from existing evaluation APIs. Derive every report and paper table from that bundle, correct row/headland metric semantics, then revise and verify the manuscript against a claim-to-evidence matrix.

**Tech Stack:** Python 3, NumPy, pandas-free CSV/JSON tooling, PyTorch/CUDA SPF runtime, evo, pytest, LaTeX/IEEE `ieeeconf`, `latexmk`, Git.

**Spec:** `docs/icra/2026-08-28-icra-scientific-paths-design.md`

## Global Constraints

- Work in `/home/pulver/projects/vineyard_pf/Outdoor_SLPF_iros_revision_plan` from branch `iros_revision_plan` or an isolated branch created from commit `c15803d`.
- Preserve `alpha_huber3_cap50` as the only paper-facing SLPF configuration.
- Use both traversals and seeds `11,22,33` for final SLPF results.
- Use in-row metrics for row-identity claims and headland metrics for transition claims.
- Keep raw map-frame APE, aligned APE, RPE, cross-track, row correctness, and recovery metrics visible.
- Do not claim detector accuracy without a supplied SemanticBLT validation YAML.
- Do not promote GTSAM or the current row-mixture/delayed-correction variants.
- The complete submitted PDF, including references, must be at most eight pages and double anonymous.
- Generated outputs live under `results/icra_submission/`; only compact CSV, JSON, Markdown, LaTeX tables, and selected figures are committed.

---

### Task 1: Register the Canonical Baseline

**Files:**
- Create: `configs/icra/alpha_huber3_cap50.yaml`
- Create: `tests/test_icra_baseline_config.py`
- Modify: `scripts/run_localization_followup_experiments.py:184`

**Interfaces:**
- Consumes: `outdoor_slpf.config.load_yaml_overlay(path)`.
- Produces: `icra_baseline_candidate(config_path: Path) -> dict[str, object]` with ID `baseline_alpha_huber3_cap50` and `--config-yaml` arguments.

- [ ] **Step 1: Write the failing configuration test**

```python
def test_icra_baseline_config_is_complete_and_frozen():
    values = load_yaml_overlay("configs/icra/alpha_huber3_cap50.yaml")
    assert values["pose_backend"] == "alpha"
    assert values["gnss_robust_mode"] == "huber"
    assert values["gnss_outlier_threshold"] == 3.0
    assert values["semantic_penalty_cap"] == 50
    assert values["row_likelihood_mode"] == "single"
    assert values["delayed_row_correction"] is False
    assert values["gnss_row_gating"] is False
```

- [ ] **Step 2: Run the test and confirm it fails because the config is absent**

Run: `python3 -m pytest tests/test_icra_baseline_config.py -q`

Expected: FAIL with `FileNotFoundError` for `configs/icra/alpha_huber3_cap50.yaml`.

- [ ] **Step 3: Add the frozen YAML and candidate helper**

```yaml
pose_backend: alpha
gnss_robust_mode: huber
gnss_outlier_threshold: 3.0
semantic_penalty_cap: 50
row_likelihood_mode: single
delayed_row_correction: false
gnss_row_gating: false
diagnostics_level: full
```

`icra_baseline_candidate()` must return `args = ["--config-yaml", str(config_path)]`; it must not duplicate the numeric values in Python.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_icra_baseline_config.py tests/test_localization_followup.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the baseline registration**

```bash
git add configs/icra/alpha_huber3_cap50.yaml tests/test_icra_baseline_config.py scripts/run_localization_followup_experiments.py
git commit -m "Freeze ICRA baseline configuration"
```

### Task 2: Make Metric Semantics Explicit

**Files:**
- Modify: `scripts/generate_localization_improvement_report.py:143`
- Modify: `scripts/run_localization_followup_experiments.py:52`
- Modify: `tests/test_localization_followup.py`

**Interfaces:**
- Consumes: per-seed fields already produced by `compute_row_metrics()`.
- Produces: `row_metric_view(row: Mapping[str, object]) -> dict[str, float | None]` with `inrow_wrong_sec`, `inrow_row_correct`, `inrow_cross_track`, `headland_cross_track`, and `headland_recovery_m`.

- [ ] **Step 1: Add a failing test that distinguishes in-row and headland metrics**

```python
def test_acceptance_uses_inrow_wrong_duration_not_total_duration():
    row = {
        "wrong_row_duration_sec_mean": "440",
        "inrow_wrong_row_duration_sec_mean": "180",
        "headland_wrong_row_duration_sec_mean": "260",
        "inrow_cross_track_mean_mean": "1.0",
        "headland_cross_track_mean_mean": "1.5",
    }
    view = report.row_metric_view(row)
    assert view["inrow_wrong_sec"] == 180.0
    assert view["headland_wrong_sec"] == 260.0
```

- [ ] **Step 2: Verify the new test fails**

Run: `python3 -m pytest tests/test_localization_followup.py::test_acceptance_uses_inrow_wrong_duration_not_total_duration -q`

Expected: FAIL because `row_metric_view` does not exist.

- [ ] **Step 3: Implement the metric view and extend aggregation keys**

Add all `inrow_*` and `headland_*` operational fields to `KEY_METRICS`. Change report acceptance and compact tables to use in-row wrong-row duration and in-row cross-track. Keep total wrong-row duration in diagnostics with the label `all-frame diagnostic`.

- [ ] **Step 4: Add a metric-definition note to generated Markdown**

The report must state: `Row-identity acceptance uses in-row frames. Headland frames are evaluated with cross-track and transition-recovery metrics because nearest-row identity is ambiguous outside a corridor.`

- [ ] **Step 5: Run report and metric tests**

Run: `python3 -m pytest tests/test_localization_followup.py tests/test_row_operational_metrics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the metric correction**

```bash
git add scripts/generate_localization_improvement_report.py scripts/run_localization_followup_experiments.py tests/test_localization_followup.py
git commit -m "Correct ICRA row metric semantics"
```

### Task 3: Build a Canonical Evidence Bundle

**Files:**
- Create: `scripts/build_icra_evidence_bundle.py`
- Create: `tests/test_icra_evidence_bundle.py`
- Create at runtime: `results/icra_submission/evidence/manifest.json`
- Create at runtime: `results/icra_submission/evidence/method_metrics_per_seed.csv`
- Create at runtime: `results/icra_submission/evidence/method_metrics_aggregate.csv`
- Create at runtime: `results/icra_submission/evidence/claim_checks.json`

**Interfaces:**
- Consumes: `followup_metrics_per_seed.csv`, `localization_metrics_per_seed.csv`, RTAB+NoisyGNSS CSVs, and run protocol JSON files.
- Produces: `build_bundle(inputs: EvidenceInputs, output_dir: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing fixture tests for provenance and completeness**

```python
def test_bundle_rejects_missing_seed(tmp_path):
    rows = make_rows(traversals=["rh_run1", "rh_run2"], seeds=[11, 22])
    with pytest.raises(ValueError, match="missing.*33"):
        build_bundle(write_inputs(tmp_path, rows), tmp_path / "out")

def test_bundle_records_exact_source_hashes(tmp_path):
    inputs = write_complete_inputs(tmp_path)
    data = build_bundle(inputs, tmp_path / "out")
    assert data["git"]["commit"]
    assert all(item["sha256"] for item in data["sources"])
```

- [ ] **Step 2: Run tests and confirm both fail**

Run: `python3 -m pytest tests/test_icra_evidence_bundle.py -q`

Expected: FAIL because the builder module does not exist.

- [ ] **Step 3: Implement strict input validation**

Require exactly one row for each `(method, traversal, seed, profile)` key. Reject duplicate keys, missing seeds, non-finite primary metrics, mixed commits, and protocols whose commands omit the frozen config.

- [ ] **Step 4: Generate paper-ready tables from the validated rows**

Implement deterministic CSV ordering and emit `paper_main_table.tex`, `paper_gnss_stress_table.tex`, and `paper_operational_table.tex`. Numeric formatting must be centralised in `format_mean_std(values, digits=2)`. Add `--paper-output-dir`; when supplied, write the same generated table bodies directly into that directory so the manuscript never depends on a manual copy.

- [ ] **Step 5: Run bundle tests and static compilation**

Run: `python3 -m pytest tests/test_icra_evidence_bundle.py -q`

Run: `python3 -m py_compile scripts/build_icra_evidence_bundle.py`

Expected: PASS.

- [ ] **Step 6: Commit the evidence builder**

```bash
git add scripts/build_icra_evidence_bundle.py tests/test_icra_evidence_bundle.py
git commit -m "Add canonical ICRA evidence bundle"
```

### Task 4: Re-run and Verify the Frozen Baseline

**Files:**
- Create at runtime: `results/icra_submission/baseline/`
- Update at runtime: `results/icra_submission/evidence/`

**Interfaces:**
- Consumes: `configs/icra/alpha_huber3_cap50.yaml` and full Riseholme datasets.
- Produces: six trajectories, diagnostics, per-seed metrics, aggregate metrics, and protocol JSON.

- [ ] **Step 1: Run a 40-frame smoke on the failure traversal**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix row-identity --stage screen --candidate-ids baseline_alpha_huber3_cap50 --traversals rh_run1 --seeds 11 --max-frames 40 --cuda-visible-devices 0 --output-root results/icra_submission/baseline_smoke
```

Expected: one evaluated run, finite APE, and diagnostics naming `row_likelihood_mode=single`.

- [ ] **Step 2: Run the canonical six-seed baseline**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix row-identity --stage full --candidate-ids baseline_alpha_huber3_cap50 --traversals rh_run1,rh_run2 --seeds 11,22,33 --cuda-visible-devices 0 --output-root results/icra_submission/baseline
```

Expected: six unique full rows.

- [ ] **Step 3: Compare the rerun with the existing controlled baseline**

Run: `python3 scripts/build_icra_evidence_bundle.py --baseline results/icra_submission/baseline/followup_metrics_per_seed.csv --reference results/localization_improvement_row_identity/followup_metrics_per_seed.csv --output-dir results/icra_submission/evidence --paper-output-dir paper/generated`

Expected: identical metrics for deterministic outputs, or an explicit per-seed delta report. Any APE delta above 0.01 m blocks manuscript updates until explained.

- [ ] **Step 4: Commit compact baseline evidence**

```bash
git add results/icra_submission/evidence
git commit -m "Record canonical ICRA baseline evidence"
```

### Task 5: Regenerate the Shareable Report from One Source

**Files:**
- Modify: `scripts/generate_localization_improvement_report.py:396`
- Modify: `tests/test_localization_followup.py`
- Create at runtime: `results/icra_submission/report/report.md`
- Create at runtime: `results/icra_submission/report/report_data.json`

**Interfaces:**
- Consumes: `results/icra_submission/evidence/manifest.json`.
- Produces: `generate_icra_report(manifest_path: Path, output_dir: Path) -> dict[str, object]`.

- [ ] **Step 1: Add a regression test for the stale-summary failure**

```python
def test_report_uses_manifest_commit_and_baseline_rows(tmp_path):
    manifest = write_manifest(tmp_path, commit="c15803d", rh1_ape=0.970, rh2_ape=0.875)
    data = generate_icra_report(manifest, tmp_path / "report")
    text = (tmp_path / "report/report.md").read_text()
    assert "c15803d" in text
    assert "0.970" in text and "0.875" in text
    assert data["git"]["commit"] == "c15803d"
```

- [ ] **Step 2: Confirm failure against the current two-summary API**

Run: `python3 -m pytest tests/test_localization_followup.py::test_report_uses_manifest_commit_and_baseline_rows -q`

Expected: FAIL because the manifest API is absent.

- [ ] **Step 3: Replace independent current-summary inputs with the evidence manifest**

Keep the old CLI arguments only long enough to emit a deprecation error with migration guidance. The default command must read `results/icra_submission/evidence/manifest.json` and must fail if its commit differs from `git rev-parse HEAD` unless `--allow-historical-commit` is explicitly supplied.

- [ ] **Step 4: Regenerate and inspect the report**

Run: `python3 scripts/generate_localization_improvement_report.py --evidence-manifest results/icra_submission/evidence/manifest.json --output-dir results/icra_submission/report`

Expected: the opening summary, tables, acceptance result, and interpretation all use the same commit and rows.

- [ ] **Step 5: Commit the report fix and compact outputs**

```bash
git add scripts/generate_localization_improvement_report.py tests/test_localization_followup.py results/icra_submission/report
git commit -m "Generate ICRA report from canonical evidence"
```

### Task 6: Create and Enforce a Claim-to-Evidence Matrix

**Files:**
- Create: `paper/icra_claims_matrix.md`
- Create: `scripts/check_paper_claims.py`
- Create: `tests/test_paper_claims.py`

**Interfaces:**
- Consumes: paper text and evidence manifest.
- Produces: `check_claims(paper_text: str, claims: list[Claim], evidence: dict) -> list[str]` where an empty list means pass.

- [ ] **Step 1: Add tests for prohibited and required claim language**

```python
def test_claim_checker_rejects_unbounded_deployment_claim():
    errors = check_claims("SLPF guarantees reliable deployment in vineyards.", CLAIMS, EVIDENCE)
    assert any("guarantees" in error for error in errors)

def test_claim_checker_accepts_bounded_row_level_claim():
    errors = check_claims("SLPF improves row-level localisation under the evaluated map and GNSS assumptions.", CLAIMS, EVIDENCE)
    assert errors == []
```

- [ ] **Step 2: Verify tests fail before the checker exists**

Run: `python3 -m pytest tests/test_paper_claims.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Write the claims matrix**

Include claim ID, exact manuscript location, supporting artifact/table, permitted wording, prohibited extension, and limitation. Explicitly cover semantic walls, raw APE, in-row row adherence, GNSS stress, RTAB anchoring, detector limitations, map assumptions, and the absence of cross-season validation.

- [ ] **Step 4: Implement a narrow checker**

Check numeric literals against generated LaTeX tables and flag the phrases `deployment ready`, `guarantees recovery`, `centimetre-grade`, `generalises across vineyards`, and `solves perceptual aliasing` unless they occur in a negated limitations statement.

- [ ] **Step 5: Run the checker tests**

Run: `python3 -m pytest tests/test_paper_claims.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the claims contract**

```bash
git add paper/icra_claims_matrix.md scripts/check_paper_claims.py tests/test_paper_claims.py
git commit -m "Add ICRA claim evidence contract"
```

### Task 7: Revise and Compress the ICRA Manuscript

**Files:**
- Modify: `paper/paper.tex`
- Modify: `paper/Makefile`
- Modify: `paper/new.bib`
- Create: `paper/icra_submission_checklist.md`
- Generate: `paper/generated/icra_main_table.tex`, `paper/generated/icra_gnss_stress_table.tex`, `paper/generated/icra_operational_table.tex`

**Interfaces:**
- Consumes: canonical generated tables and claims matrix.
- Produces: anonymous `paper/paper.pdf` with at most eight pages.

- [ ] **Step 1: Replace hand-entered numerical table bodies with generated inputs**

Use `\input{generated/icra_main_table.tex}`, `\input{generated/icra_gnss_stress_table.tex}`, and `\input{generated/icra_operational_table.tex}`. Captions and labels remain in `paper.tex`.

- [ ] **Step 2: Correct metric wording**

Define row correctness and wrong-row duration on in-row frames. State that headland row identity is ambiguous and is evaluated by cross-track and recovery behaviour. Remove any sentence that infers guaranteed recovery from switch counts.

- [ ] **Step 3: Remove unsupported algorithmic claims**

Do not mention persistent multi-hypothesis reasoning as an implemented contribution. If the current row-mixture ablation is retained, describe it as an exploratory negative/partial result in one sentence without adding a table.

- [ ] **Step 4: Anonymise the submission**

Replace authors and affiliations with the anonymous ICRA form, remove acknowledgements and identifiable repository URLs, and inspect PDF metadata. Keep a private author block in Git history rather than introducing a second drifting manuscript.

- [ ] **Step 5: Reduce from nine to eight complete pages**

Prioritise removal of commented legacy prose, duplicated results narration, oversized tables, and template material. Preserve methods, limitations, raw APE, operational metrics, and GNSS stress evidence.

- [ ] **Step 6: Add the submission checklist**

Record eight-page complete limit, double anonymity, PDF compliance, exact title/abstract match, AI-use disclosure decision, video decision, and final artifact hashes.

- [ ] **Step 7: Build and run claim checks**

Run: `python3 scripts/check_paper_claims.py --paper paper/paper.tex --claims paper/icra_claims_matrix.md --evidence results/icra_submission/evidence/manifest.json`

Run: `cd paper && latexmk -pdf -halt-on-error paper.tex`

Run: `pdfinfo paper/paper.pdf | rg '^Pages:'`

Expected: claim checker passes and page count is at most 8.

- [ ] **Step 8: Commit the manuscript revision**

```bash
git add paper/paper.tex paper/Makefile paper/new.bib paper/generated paper/icra_submission_checklist.md
git commit -m "Prepare evidence-first ICRA manuscript"
```

### Task 8: Final Verification and Submission Bundle

**Files:**
- Create at runtime: `results/icra_submission/verification/verification.json`
- Create at runtime: `results/icra_submission/submission_manifest.sha256`

**Interfaces:**
- Consumes: all source, compact evidence, generated tables, figures, and PDF.
- Produces: machine-readable verification record and final checksums.

- [ ] **Step 1: Run the complete Python test suite**

Run: `MPLCONFIGDIR=.tmp_mpl python3 -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 2: Compile changed Python modules**

Run: `python3 -m py_compile scripts/build_icra_evidence_bundle.py scripts/generate_localization_improvement_report.py scripts/check_paper_claims.py scripts/run_localization_followup_experiments.py`

Expected: exit code 0.

- [ ] **Step 3: Rebuild LaTeX from a clean auxiliary state**

Run: `cd paper && latexmk -C paper.tex && latexmk -pdf -halt-on-error paper.tex`

Expected: exit code 0, at most eight pages, no undefined references, and no table extending beyond margins.

- [ ] **Step 4: Verify anonymity and PDF metadata**

Run: `pdftotext paper/paper.pdf - | rg -i 'pulver|lincoln|delft|eth|github|epsrc'`

Expected: no identifying author or funding text in the anonymous submission.

- [ ] **Step 5: Record checksums and repository state**

Run: `sha256sum paper/paper.pdf results/icra_submission/evidence/manifest.json results/icra_submission/report/report_data.json > results/icra_submission/submission_manifest.sha256`

Run: `git status --short --branch`

Expected: only intentionally untracked large run directories remain.

- [ ] **Step 6: Commit compact verification artifacts**

```bash
git add results/icra_submission/verification results/icra_submission/submission_manifest.sha256 paper/paper.pdf
git commit -m "Verify ICRA submission bundle"
```

## Final Go/No-Go

Submit this path when the canonical evidence bundle is internally consistent, the bounded claims pass, and the anonymous PDF is at most eight pages. If baseline reruns differ materially, detector language overreaches, or the paper cannot be reduced without removing essential evidence, stop publication work and resolve that specific blocker; do not substitute row-mixture results to fill the gap.
