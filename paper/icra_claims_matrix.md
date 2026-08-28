# ICRA claim-to-evidence matrix

| ID | Manuscript location | Supporting evidence | Permitted wording | Prohibited extension | Limitation |
| --- | --- | --- | --- | --- | --- |
| C1 | Introduction; method | Semantic-wall map and generated main table | Semantic walls provide row-aligned constraints for the evaluated map. | Claims of universal aliasing resolution | One surveyed vineyard map and fixed landmark classes. |
| C2 | Experiments | Raw and aligned APE plus RPE in `results/icra_submission/evidence/` | Report map-frame and aligned localisation errors together. | Treating aligned APE as proof of map-frame accuracy | Alignment removes global transform differences. |
| C3 | Operational results | In-row row correctness, cross-track, and wrong-row duration | State row-level behaviour on in-row frames. | Using nearest-row headland labels as unique identity | Headland identity is ambiguous. |
| C4 | GNSS stress | GNSS degradation metrics and stress table | Describe dependence on the evaluated GNSS degradation profile. | Claiming GNSS-independent localisation | Map quality, landmark visibility, and GNSS remain assumptions. |
| C5 | Baselines | RTAB RGB/RGBD and NoisyGNSS artifacts | Compare against the recorded start-pose anchored baselines. | Claiming superiority over unmeasured systems | Baseline preprocessing is protocol-specific. |
| C6 | Detector description | Detector outputs and model configuration | Describe the detector as an input to localisation. | Claiming detector accuracy without SemanticBLT validation YAML | No supplied validation YAML and no cross-season detector validation. |
| C7 | Limitations | Dataset/map provenance and protocol JSON | Bound conclusions to the surveyed vineyard and fixed map. | `generalises across vineyards` | No cross-season or cross-vineyard validation. |
| C8 | Conclusion | Canonical manifest and claim checks | Claim evaluated row-level localisation and corridor adherence. | `deployment ready`, guaranteed recovery, centimetre-grade, or solving perceptual aliasing | This is an evidence-first submission, not a deployment guarantee. |

