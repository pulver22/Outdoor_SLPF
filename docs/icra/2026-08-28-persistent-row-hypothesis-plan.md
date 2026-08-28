# Persistent Row-Hypothesis Localisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a persistent labelled row-mode particle filter with bounded-latency discrete selection improves worst-case row recovery without degrading the strong baseline on the opposite traversal.

**Architecture:** Derive a fixed row-transition topology from the semantic-wall map, attach row labels to particles, preserve the top three row modes through quota-based resampling, and apply GNSS as a prior over mode identity. Add a two-second fixed-lag decoder over discrete modes while retaining separate causal output and the existing continuous PF.

**Tech Stack:** Python 3, NumPy, PyTorch/CUDA, dataclasses, existing Outdoor SLPF map/evaluation APIs, pytest, evo, Matplotlib, LaTeX.

**Spec:** `docs/icra/2026-08-28-icra-scientific-paths-design.md`

## Global Constraints

- Start from the frozen `alpha_huber3_cap50` baseline and do not tune Huber, semantic-cap, or smoothing parameters.
- Keep the semantic-wall map static; derive topology once at startup.
- Use `K=3` persistent modes, a `2.0 s` lag, and one pre-registered parameter set.
- Never use ground truth for online headland detection, row transitions, GNSS gating, or mode selection.
- Emit and evaluate causal and lagged trajectories separately.
- Use in-row metrics for row identity and headland transition metrics for recovery.
- Run `rh_run1` seed 11 first, then a two-traversal seed-11 screen, then the full six-seed matrix only after promotion gates pass.
- Stop this path by 4 September 2026 if the screen fails and execute the evidence-first plan.
- Do not perform a second architecture or scalar-parameter sweep before ICRA submission.

---

### Task 1: Correct the Research Acceptance Metrics

**Files:**
- Modify: `scripts/generate_localization_improvement_report.py:170`
- Modify: `scripts/run_localization_followup_experiments.py:52`
- Modify: `tests/test_localization_followup.py`

**Interfaces:**
- Consumes: existing `inrow_*` and `headland_*` metrics from `compute_row_metrics()`.
- Produces: `evaluate_persistent_mode_gate(per_seed_rows, aggregate_rows, baseline_id, candidate_id) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for the pre-registered gate**

```python
def test_persistent_gate_uses_inrow_metrics_and_numeric_tolerance():
    result = evaluate_persistent_mode_gate(PASSING_ROWS, PASSING_AGGREGATES, "baseline", "persistent")
    assert result["rh1_seed11_lt_1p2"]
    assert result["rh1_mean_lt_1p0"]
    assert result["rh2_within_0p01m"]
    assert result["pooled_inrow_wrong_reduction_ge_10pct"]
    assert result["pooled_inrow_cross_track_reduction_ge_5pct"]
    assert result["accepted"]
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python3 -m pytest tests/test_localization_followup.py::test_persistent_gate_uses_inrow_metrics_and_numeric_tolerance -q`

Expected: FAIL because the gate function is absent.

- [ ] **Step 3: Implement the exact gate from the design**

The return data must include every threshold, measured value, paired per-seed delta, pooled reduction, latency, and runtime ratio. `accepted` is true only when every hard condition passes.

- [ ] **Step 4: Add a test that rejects a candidate with better APE but worse in-row duration**

Run: `python3 -m pytest tests/test_localization_followup.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the pre-registration**

```bash
git add scripts/generate_localization_improvement_report.py scripts/run_localization_followup_experiments.py tests/test_localization_followup.py
git commit -m "Pre-register persistent row hypothesis gates"
```

### Task 2: Extract Fixed Row Topology

**Files:**
- Create: `outdoor_slpf/row_topology.py`
- Create: `tests/test_row_topology.py`
- Modify: `scripts/spf_lidar.py:353`

**Interfaces:**
- Consumes: `grouped_map_points: Mapping[str, list[dict[str, object]]]`.
- Produces: `RowTopology.from_grouped_map(grouped_map_points)`, `nearest_row(xy)`, `near_endpoint(xy, radius_m=3.0)`, and `transition_candidates(row_id, xy)`.

- [ ] **Step 1: Write synthetic topology tests**

```python
def test_parallel_rows_are_neighbours_only_at_matching_endpoints():
    topology = RowTopology.from_grouped_map(three_parallel_rows(spacing=3.0, length=20.0))
    assert topology.transition_candidates("row_1", np.array([0.0, 20.0])) == ("row_0", "row_2")
    assert topology.transition_candidates("row_1", np.array([0.0, 10.0])) == ()

def test_headland_detection_uses_map_endpoints_not_ground_truth():
    topology = RowTopology.from_grouped_map(three_parallel_rows(spacing=3.0, length=20.0))
    assert topology.near_endpoint(np.array([0.2, 20.1]), radius_m=3.0)
    assert not topology.near_endpoint(np.array([0.2, 10.0]), radius_m=3.0)
```

- [ ] **Step 2: Verify tests fail on import**

Run: `python3 -m pytest tests/test_row_topology.py -q`

Expected: FAIL because `row_topology.py` does not exist.

- [ ] **Step 3: Implement deterministic topology construction**

Represent each row as a sorted polyline with two endpoints. Rows are transition neighbours when corresponding endpoints are within `1.75 * median_row_spacing`; compute median spacing from row centreline distances. Sort all returned row IDs for deterministic runs.

- [ ] **Step 4: Replace the diagnostic-only headland flag**

In `spf_lidar.py`, construct one `RowTopology` after loading the map. Set `headland_flag` from `topology.near_endpoint(estimated_pose[:2])` rather than `corridor_dist >= 2.5`.

- [ ] **Step 5: Run topology and existing map tests**

Run: `python3 -m pytest tests/test_row_topology.py tests/test_row_operational_metrics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit topology extraction**

```bash
git add outdoor_slpf/row_topology.py tests/test_row_topology.py scripts/spf_lidar.py
git commit -m "Add fixed semantic row topology"
```

### Task 3: Implement Persistent Labelled Particle Modes

**Files:**
- Create: `outdoor_slpf/row_hypotheses.py`
- Create: `tests/test_row_hypotheses.py`
- Modify: `scripts/spf_lidar.py:594`

**Interfaces:**
- Produces: `RowModeConfig`, `RowHypothesisState`, `initialise_row_labels()`, `propose_row_label_transitions()`, `mode_posteriors()`, `estimate_pose_by_mode()`, and `resample_with_mode_quotas()`.

```python
@dataclass(frozen=True)
class RowModeConfig:
    top_k: int = 3
    min_mode_fraction: float = 0.15
    inrow_switch_probability: float = 0.001
    headland_switch_probability: float = 0.20

@dataclass
class RowHypothesisState:
    labels: np.ndarray
    mode_probabilities: dict[str, float]
    frames_since_support: dict[str, int]
```

- [ ] **Step 1: Write tests that preserve a secondary mode**

```python
def test_resampling_preserves_top_three_modes():
    indices, labels = resample_with_mode_quotas(PARTICLES, WEIGHTS, LABELS, CONFIG, rng=np.random.default_rng(11))
    counts = Counter(labels)
    assert set(counts) == {"row_0", "row_1", "row_2"}
    assert all(count >= int(len(labels) * 0.15) for count in counts.values())

def test_inrow_transition_prior_blocks_nonlocal_switch():
    probs = transition_prior("row_1", candidates=("row_0", "row_2"), in_headland=False, config=CONFIG)
    assert probs["row_1"] > 0.99
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `python3 -m pytest tests/test_row_hypotheses.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement labelled state and quota resampling**

Use one shared particle array plus one row-label array. Initialise labels from the nearest row polyline. `propose_row_label_transitions()` may move labels only to `RowTopology.transition_candidates()` and uses the in-row or headland transition probability from `RowModeConfig`. Compute mode mass by summing normalised particle weights per label. Retain at most three modes whose posterior is at least 0.02 or whose last semantic support is no older than 10 processed frames. Allocate `min_mode_fraction * particle_count` to each retained mode, then distribute remaining samples by posterior mass. Reject `top_k * min_mode_fraction >= 1.0`.

`estimate_pose_by_mode()` must return the weighted circular pose estimate for every retained label; these per-mode estimates are the only continuous states passed to the delayed decoder.

- [ ] **Step 4: Add deterministic collapse and recovery tests**

Create a synthetic sequence where row 1 wins early, row 2 becomes correct later, and ordinary resampling loses row 2. Assert quota resampling retains row 2 and changes the selected mode after later evidence.

- [ ] **Step 5: Run hypothesis tests**

Run: `python3 -m pytest tests/test_row_hypotheses.py -q`

Expected: PASS.

- [ ] **Step 6: Commit persistent mode state**

```bash
git add outdoor_slpf/row_hypotheses.py tests/test_row_hypotheses.py
git commit -m "Preserve persistent row particle modes"
```

### Task 4: Condition the Measurement Model on Row Labels

**Files:**
- Modify: `scripts/spf_lidar.py:1266`
- Modify: `outdoor_slpf/row_hypotheses.py`
- Modify: `tests/test_row_identity_localization.py`

**Interfaces:**
- Consumes: `particle_row_indices: torch.Tensor` and `seg_row_idx: torch.Tensor`.
- Produces: `row_conditioned_semantic_log_likelihood(...) -> torch.Tensor` with one score per particle.

- [ ] **Step 1: Add a failing row-conditioned likelihood test**

```python
def test_row_conditioned_likelihood_scores_only_labelled_row():
    scores = row_conditioned_semantic_log_likelihood(OBS, PARTICLES, labels=[0, 1], segments=TWO_ROW_SEGMENTS)
    assert scores[0] > scores[1]
    assert torch.isfinite(scores).all()
```

- [ ] **Step 2: Confirm the test fails**

Run: `python3 -m pytest tests/test_row_identity_localization.py::test_row_conditioned_likelihood_scores_only_labelled_row -q`

Expected: FAIL because the function is absent.

- [ ] **Step 3: Implement labelled-row scoring**

Apply the existing robust loss inside each particle's labelled row. Do not log-sum-exp over other rows. Preserve the current single-hypothesis and mixture modes behind their existing switch for ablation compatibility.

- [ ] **Step 4: Add mode evidence diagnostics**

Log per-mode semantic log evidence, particle count, posterior mass, age, and last-supported frame. Extend `DIAGNOSTIC_STATS_FIELDS` in `outdoor_slpf/pipeline.py` with compact top-three fields.

- [ ] **Step 5: Run likelihood and pipeline tests**

Run: `python3 -m pytest tests/test_row_identity_localization.py tests/test_row_hypotheses.py -q`

Expected: PASS.

- [ ] **Step 6: Commit row-conditioned likelihood**

```bash
git add scripts/spf_lidar.py outdoor_slpf/row_hypotheses.py outdoor_slpf/pipeline.py tests/test_row_identity_localization.py tests/test_row_hypotheses.py
git commit -m "Condition semantic likelihood on row modes"
```

### Task 5: Add Bounded-Latency Discrete Selection

**Files:**
- Create: `outdoor_slpf/row_mode_decoder.py`
- Create: `tests/test_row_mode_decoder.py`
- Modify: `scripts/spf_lidar.py:2077`

**Interfaces:**
- Produces: `FixedLagRowDecoder(lag_sec=2.0)`, `push(frame: RowModeFrame)`, and `flush() -> list[DecodedRowPose]`.

```python
@dataclass(frozen=True)
class RowModeFrame:
    timestamp: float
    mode_log_probabilities: dict[str, float]
    transition_log_probabilities: dict[tuple[str, str], float]
    pose_by_mode: dict[str, np.ndarray]
```

- [ ] **Step 1: Write a Viterbi-style delayed-selection test**

```python
def test_decoder_revises_ambiguous_frame_after_later_evidence():
    decoder = FixedLagRowDecoder(lag_sec=2.0)
    outputs = feed_ambiguous_then_decisive_sequence(decoder)
    assert outputs[0].row_id == "row_2"
    assert outputs[0].latency_sec <= 2.0
```

- [ ] **Step 2: Verify failure before the decoder exists**

Run: `python3 -m pytest tests/test_row_mode_decoder.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement discrete dynamic programming over the lag window**

Store only timestamps, mode scores, transition scores, back-pointers, and per-mode pose estimates. Do not store or optimise a continuous factor graph. Emit a frame when its age reaches `lag_sec`; select its row and corresponding mode-conditioned pose through the best path.

- [ ] **Step 4: Emit separate trajectories**

Keep `trajectory_0.5.tum` as the causal estimate for backward compatibility. Add `trajectory_row_lagged.tum` and `row_mode_sequence.csv` with `timestamp`, `causal_row_id`, `lagged_row_id`, `latency_sec`, and mode probabilities.

- [ ] **Step 5: Run decoder and trajectory-format tests**

Run: `python3 -m pytest tests/test_row_mode_decoder.py tests/test_row_identity_localization.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the delayed decoder**

```bash
git add outdoor_slpf/row_mode_decoder.py tests/test_row_mode_decoder.py scripts/spf_lidar.py
git commit -m "Add bounded-latency row mode decoder"
```

### Task 6: Move GNSS Influence to the Mode Prior

**Files:**
- Modify: `outdoor_slpf/row_hypotheses.py`
- Modify: `scripts/spf_lidar.py:1094`
- Modify: `tests/test_row_hypotheses.py`

**Interfaces:**
- Produces: `gnss_mode_log_prior(gnss_xy, topology, candidate_rows, sigma_m=1.8, jump_threshold_m=8.0) -> dict[str, float]`.

- [ ] **Step 1: Add GNSS prior tests**

```python
def test_gnss_prior_prunes_distant_row_without_overwriting_semantic_winner():
    prior = gnss_mode_log_prior(np.array([0.0, 0.0]), TOPOLOGY, ["row_0", "row_1", "row_2"])
    posterior = combine_mode_evidence(semantic={"row_0": 4.0, "row_1": 3.8}, gnss=prior, semantic_confidence=0.9)
    assert max(posterior, key=posterior.get) == "row_0"

def test_gnss_jump_returns_neutral_prior():
    prior = gnss_mode_log_prior(JUMP_POINT, TOPOLOGY, ROWS, previous_gnss_xy=ORIGIN)
    assert len(set(prior.values())) == 1
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_row_hypotheses.py -q`

Expected: FAIL because mode-prior helpers are absent.

- [ ] **Step 3: Implement the adaptive mode prior**

Use distance from GNSS to each row polyline with sigma 1.8 m. Make the prior neutral for jumps above 8.0 m. Interpolate semantic/GNSS influence from semantic mode entropy: high semantic confidence reduces GNSS contribution; sparse or high-entropy semantic evidence increases it.

- [ ] **Step 4: Disable continuous GNSS row gating in persistent mode runs**

The existing pose-level robust GNSS term remains part of the frozen baseline. `--persistent-row-modes` must route row disambiguation through the mode prior and log `gnss_mode_prior_*` fields.

- [ ] **Step 5: Run hypothesis and GNSS tests**

Run: `python3 -m pytest tests/test_row_hypotheses.py tests/test_row_identity_localization.py -q`

Expected: PASS.

- [ ] **Step 6: Commit GNSS mode prior**

```bash
git add outdoor_slpf/row_hypotheses.py scripts/spf_lidar.py tests/test_row_hypotheses.py
git commit -m "Use GNSS as a row mode prior"
```

### Task 7: Integrate CLI, Runner, Diagnostics, and Evaluation

**Files:**
- Modify: `scripts/spf_lidar.py:2730`
- Modify: `scripts/run_localization_followup_experiments.py:184`
- Modify: `scripts/analyze_localization_followup.py`
- Modify: `scripts/run_ab_validation.py:480`
- Modify: `tests/test_localization_followup.py`
- Modify: `tests/test_row_identity_localization.py`

**Interfaces:**
- Adds CLI: `--persistent-row-modes`, `--row-mode-top-k 3`, `--row-mode-min-fraction 0.15`, and `--row-mode-lag-sec 2.0`.
- Adds candidate IDs: `persistent_modes_causal`, `persistent_modes_lagged`, and `persistent_modes_lagged_gnss_prior`.
- Adds runner matrix choice `persistent-row` and report CLI `--check-persistent-gate` with `--strict` and `--decision-output`.

- [ ] **Step 1: Write exact candidate-matrix tests**

```python
def test_persistent_candidate_matrix_is_fixed():
    candidates = persistent_row_candidate_matrix()
    assert [item["id"] for item in candidates] == [
        "baseline_alpha_huber3_cap50",
        "row_mixture_delayed_reference",
        "persistent_modes_causal",
        "persistent_modes_lagged",
        "persistent_modes_lagged_gnss_prior",
    ]
    assert candidates[-1]["args"][-2:] == ["--row-mode-lag-sec", "2.0"]
```

- [ ] **Step 2: Verify the matrix test fails**

Run: `python3 -m pytest tests/test_localization_followup.py::test_persistent_candidate_matrix_is_fixed -q`

Expected: FAIL because the matrix is absent.

- [ ] **Step 3: Add the fixed matrix and lagged evaluation routing**

Extend the runner's `--matrix` choices to `followup,row-identity,persistent-row`. `persistent_modes_causal` evaluates `trajectory_0.5.tum`. Lagged variants evaluate `trajectory_row_lagged.tum`. Every row records `trajectory_mode`, `latency_sec_mean`, and `latency_sec_max`.

Extend the report CLI so `--check-persistent-gate --decision-output PATH` writes the exact gate result as JSON. `--strict` exits non-zero when no candidate passes or when the report's accepted candidate disagrees with recomputation from raw rows.

- [ ] **Step 4: Extend diagnostics**

Add plots for mode posterior over time, retained particle counts per mode, causal versus lagged row identity, transition events, and GNSS mode-prior influence. Keep existing ESS, maximum weight, innovation, and entropy plots.

- [ ] **Step 5: Run runner and diagnostic tests**

Run: `python3 -m pytest tests/test_localization_followup.py tests/test_row_identity_localization.py -q`

Expected: PASS.

- [ ] **Step 6: Commit experiment integration**

```bash
git add scripts/spf_lidar.py scripts/run_localization_followup_experiments.py scripts/analyze_localization_followup.py scripts/run_ab_validation.py tests/test_localization_followup.py tests/test_row_identity_localization.py
git commit -m "Integrate persistent row mode experiments"
```

### Task 8: Pass the Synthetic and Targeted Gates

**Files:**
- Create: `tests/fixtures/row_aliasing_sequence.json`
- Create: `tests/test_row_aliasing_scenario.py`
- Create at runtime: `results/icra_persistent_modes/screen/`

**Interfaces:**
- Consumes: fixed topology, persistent modes, decoder, and frozen baseline.
- Produces: synthetic recovery metrics and targeted real-run metrics.

- [ ] **Step 1: Add the deterministic aliasing scenario**

The fixture contains three parallel rows, an initial ambiguous interval, a misleading GNSS burst, a headland transition, and later decisive semantic observations. Baseline collapse and persistent-mode recovery must be deterministic at seed 11.

- [ ] **Step 2: Run the synthetic test**

Run: `python3 -m pytest tests/test_row_aliasing_scenario.py -q`

Expected: baseline loses the correct row mode; persistent modes retain and recover it within the two-second lag.

- [ ] **Step 3: Run the 40-frame integration smoke**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix persistent-row --stage screen --candidate-ids baseline_alpha_huber3_cap50,persistent_modes_lagged_gnss_prior --traversals rh_run1 --seeds 11 --max-frames 40 --cuda-visible-devices 0 --output-root results/icra_persistent_modes/smoke
```

Expected: two finite evaluated runs and a non-empty lagged trajectory.

- [ ] **Step 4: Run the full rh_run1 seed-11 target**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix persistent-row --stage screen --candidate-ids baseline_alpha_huber3_cap50,persistent_modes_causal,persistent_modes_lagged,persistent_modes_lagged_gnss_prior --traversals rh_run1 --seeds 11 --cuda-visible-devices 0 --output-root results/icra_persistent_modes/target
```

Expected promotion gate: lagged+GNSS aligned APE below 1.20 m, in-row wrong duration below its paired baseline, and in-row cross-track no worse than baseline.

- [ ] **Step 5: Run the two-traversal seed-11 screen only if Step 4 passes**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix persistent-row --stage screen --candidate-ids baseline_alpha_huber3_cap50,persistent_modes_lagged_gnss_prior --traversals rh_run1,rh_run2 --seeds 11 --cuda-visible-devices 0 --output-root results/icra_persistent_modes/screen
```

Expected: no seed has APE more than 0.10 m worse than baseline and both traversals reduce in-row wrong duration.

- [ ] **Step 6: Record the hard decision**

Run: `python3 scripts/generate_localization_improvement_report.py --followup-csv results/icra_persistent_modes/screen/followup_metrics_per_seed.csv --followup-aggregate-csv results/icra_persistent_modes/screen/followup_metrics_aggregate.csv --check-persistent-gate --decision-output results/icra_persistent_modes/go_no_go.json --output-dir results/icra_persistent_modes/screen/report`

The JSON must contain `decision`, every measured gate, exact commit, and commands. `decision=no-go` ends this plan and transfers execution to `docs/icra/2026-08-28-conservative-icra-submission-plan.md`.

- [ ] **Step 7: Commit compact screen evidence**

```bash
git add tests/fixtures/row_aliasing_sequence.json tests/test_row_aliasing_scenario.py results/icra_persistent_modes/go_no_go.json
git commit -m "Validate persistent row mode promotion gate"
```

### Task 9: Run the Full Controlled Matrix

**Files:**
- Create at runtime: `results/icra_persistent_modes/full/`
- Create at runtime: `results/icra_persistent_modes/followup_metrics_per_seed.csv`
- Create at runtime: `results/icra_persistent_modes/followup_metrics_aggregate.csv`
- Create at runtime: `results/icra_persistent_modes/followup_protocol_full.json`

**Interfaces:**
- Consumes: the five fixed candidates and full two-traversal data.
- Produces: 30 evaluated rows plus causal/lagged diagnostics.

- [ ] **Step 1: Run the matrix without changing parameters**

Run:

```bash
/home/pulver/projects/vineyard_pf/Outdoor_SLPF/.venv/bin/python scripts/run_localization_followup_experiments.py --matrix persistent-row --stage full --traversals rh_run1,rh_run2 --seeds 11,22,33 --cuda-visible-devices 0 --output-root results/icra_persistent_modes
```

Expected: exactly 30 unique rows.

- [ ] **Step 2: Generate diagnostics and acceptance report**

Run: `python3 scripts/analyze_localization_followup.py --matrix-root results/icra_persistent_modes --output-dir results/icra_persistent_modes/diagnostics`

Run: `python3 scripts/generate_localization_improvement_report.py --followup-csv results/icra_persistent_modes/followup_metrics_per_seed.csv --followup-aggregate-csv results/icra_persistent_modes/followup_metrics_aggregate.csv --output-dir results/icra_persistent_modes/report`

Expected: acceptance report contains causal/lagged metrics, paired deltas, pooled reductions, runtime, and latency.

- [ ] **Step 3: Apply the pre-registered gate**

Run: `python3 scripts/generate_localization_improvement_report.py --check-persistent-gate --output-dir results/icra_persistent_modes/report`

Expected: one explicit `accepted_candidate` or `none`; no weighted-mean-only promotion.

- [ ] **Step 4: Commit compact full evidence**

```bash
git add results/icra_persistent_modes/followup_metrics_per_seed.csv results/icra_persistent_modes/followup_metrics_aggregate.csv results/icra_persistent_modes/followup_protocol_full.json results/icra_persistent_modes/report results/icra_persistent_modes/diagnostics/diagnostic_metrics.csv
git commit -m "Record persistent row mode evaluation"
```

### Task 10: Update the Paper Only After Acceptance

**Files:**
- Modify if accepted: `paper/paper.tex`
- Create if accepted: `paper/generated/icra_persistent_modes_table.tex`
- Modify: `paper/icra_claims_matrix.md`
- Modify: `paper/icra_submission_checklist.md`

**Interfaces:**
- Consumes: accepted full-run report and causal/lagged figures.
- Produces: an anonymous, at-most-eight-page manuscript with explicit latency and causal performance.

- [ ] **Step 1: Stop paper changes when no candidate is accepted**

If `accepted_candidate` is empty, add one sentence to the evidence-first limitations: `A persistent row-mode extension was evaluated but did not satisfy the pre-registered cross-traversal robustness gate.` Then execute the conservative plan.

- [ ] **Step 2: Generate the accepted ablation table**

Include baseline, existing row-mixture+delayed reference, persistent causal, persistent lagged, and persistent lagged+GNSS prior. Report aligned APE, in-row wrong duration, in-row cross-track, row correctness, runtime ratio, and latency.

- [ ] **Step 3: Reframe the contribution only if the gate passes**

Use: `Robust vineyard localisation requires explicit handling of row-identity ambiguity; persistent row-conditioned particle modes improve worst-case robustness and delayed wrong-row recovery beyond a single-mode semantic-wall filter.`

- [ ] **Step 4: State the latency contract**

Distinguish online causal output from two-second delayed output in the abstract, method, experiments, table captions, and limitations. Do not compare lagged candidate metrics against causal baselines without displaying both.

- [ ] **Step 5: Build, anonymise, and reduce to eight pages**

Run: `cd paper && latexmk -pdf -halt-on-error paper.tex`

Run: `pdfinfo paper/paper.pdf | rg '^Pages:'`

Expected: at most eight complete pages and no identifying author text.

- [ ] **Step 6: Commit the accepted scientific path**

```bash
git add paper/paper.tex paper/generated/icra_persistent_modes_table.tex paper/icra_claims_matrix.md paper/icra_submission_checklist.md paper/paper.pdf
git commit -m "Present persistent row modes for ICRA"
```

### Task 11: Final Verification

**Files:**
- Create at runtime: `results/icra_persistent_modes/verification.json`

**Interfaces:**
- Consumes: source, tests, full metrics, report, and manuscript.
- Produces: final machine-readable pass/fail evidence.

- [ ] **Step 1: Run the complete test suite**

Run: `MPLCONFIGDIR=.tmp_mpl python3 -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 2: Compile all changed Python modules**

Run: `python3 -m py_compile outdoor_slpf/row_topology.py outdoor_slpf/row_hypotheses.py outdoor_slpf/row_mode_decoder.py scripts/spf_lidar.py scripts/run_localization_followup_experiments.py scripts/analyze_localization_followup.py scripts/generate_localization_improvement_report.py`

Expected: exit code 0.

- [ ] **Step 3: Verify experiment completeness**

Run: `python3 -c "import csv; rows=list(csv.DictReader(open('results/icra_persistent_modes/followup_metrics_per_seed.csv'))); keys={(r['traversal'],r['candidate_id'],r['seed']) for r in rows if r['stage']=='full'}; assert len(keys)==30; print('verified 30 full runs')"`

Expected: `verified 30 full runs`.

- [ ] **Step 4: Verify the acceptance claim against raw rows**

Run: `python3 scripts/generate_localization_improvement_report.py --check-persistent-gate --strict --output-dir results/icra_persistent_modes/report`

Expected: exit code 0 only when the manuscript's accepted candidate matches the computed candidate.

- [ ] **Step 5: Rebuild and inspect the paper**

Run: `cd paper && latexmk -C paper.tex && latexmk -pdf -halt-on-error paper.tex`

Expected: at most eight pages, no undefined references, and no unacknowledged causal/lagged metric mixing.

- [ ] **Step 6: Commit verification evidence**

```bash
git add results/icra_persistent_modes/verification.json
git commit -m "Verify persistent row mode submission path"
```

## Final Go/No-Go

This path succeeds only through the complete pre-registered gate. A visually better trajectory, lower weighted mean APE, or isolated seed improvement is insufficient. If the gate fails at the targeted or full stage, preserve the negative result and immediately use the evidence-first submission plan; do not tune mode count, lag, quotas, Huber thresholds, semantic cap, or GNSS sigma against the failed full matrix.
