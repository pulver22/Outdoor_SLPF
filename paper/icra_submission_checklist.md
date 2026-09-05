# ICRA submission checklist

This checklist is the release gate for the evidence-first manuscript. It is intentionally explicit about the current evidence boundary.

- [x] Paper-facing configuration is `configs/icra/alpha_huber3_cap50.yaml`.
- [x] Both traversals and seeds `11,22,33` are represented in the canonical source matrix.
- [x] Raw map-frame APE, aligned APE, RPE, in-row cross-track, in-row row correctness, wrong-row duration, headland cross-track, and recovery distance are retained in the evidence bundle.
- [x] Row identity is interpreted only on in-row frames; headland transition metrics are separate.
- [x] The report and manuscript are generated from the canonical evidence manifest.
- [x] No detector accuracy claim is made without a SemanticBLT validation YAML.
- [x] GTSAM and row-mixture/delayed-correction variants are not paper-facing methods.
- [x] Manuscript is double anonymous: author block, affiliations, funding, and acknowledgements removed.
- [x] Complete PDF, including references, is at most eight pages: verify with `pdfinfo paper/paper.pdf`.
- [x] PDF compliance checks pass: no author, affiliation, funder, repository, or contact identifiers in extracted text.
- [x] Title and abstract match the bounded evidence-first claim.
- [x] Baseline provenance is disclosed: dedicated `rh_run1` AMCL/RTAB-Map references are distinguished from historical reference rows.
- [x] AI-use disclosure decision: omit from the anonymous manuscript and retain the decision here for camera-ready review.
- [x] Video decision: no video claim or supplementary video is required for this evidence-first release; decide separately if the submission portal requests one.
- [x] Fresh six-run CUDA rerun is represented by the canonical verification record; the baseline matrix remains mixed-provenance and is labelled as descriptive.
- [ ] Final upload package must be rechecked against the official ICRA call immediately before submission.

## Release artifacts

- Evidence manifest: `results/icra_submission/evidence/manifest.json`
- Shareable report: `results/icra_submission/report/report.md`
- Generated paper tables: `paper/generated/icra_*.tex`
- PDF: `paper/paper.pdf`
- SHA256 file: `results/icra_submission/submission_manifest.sha256` (authoritative; the verification-directory copy is deprecated)
