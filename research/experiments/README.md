# experiments

Every figure in the top-level [README](../../README.md) and every entry in
[the paper](../PAPER.md) comes from a script here, and the script is the record: a number
without one is a number nobody can re-derive.

These are **not** part of the installed package. They import `entroptics_jlens` from `../src`,
they take a `--lens` you fetched (`entroptics-jlens fetch --help`), and several need `torch` and
`transformers` because they read a live model. The library itself needs neither.

Each row is the script's own first line of documentation, generated from the tree rather than
maintained by hand, so this table cannot drift from what is actually here.

| script | what it asks |
|---|---|
| [`exp1_transport_spectrum.py`](exp1_transport_spectrum.py) | Experiment 1 -- the resolved rank of every transport in a fitted lens. |
| [`exp2_depth_profile.py`](exp2_depth_profile.py) | Experiment 2 -- the depth profile of the residual transport, across models. |
| [`exp3_two_fit_noise.py`](exp3_two_fit_noise.py) | Experiment 3 -- the estimator noise, from two independent fits of the same model. |
| [`exp4_stream_complement.py`](exp4_stream_complement.py) | Experiment 4 -- what stands outside J-space, measured on real residual streams. |
| [`exp5_cross_model.py`](exp5_cross_model.py) | Experiment 5 -- two models meeting on one screen. |
| [`exp6_coupling_control.py`](exp6_coupling_control.py) | Is cross-model coupling layer-specific, or just shared token statistics? |
| [`exp7_escalation.py`](exp7_escalation.py) | P1 -- escalation. Can the small model tell you, by itself, when to call the large one? |
| [`exp8_selective.py`](exp8_selective.py) | P2 -- selective prediction against ground truth. |
| [`exp9_probe_placement.py`](exp9_probe_placement.py) | P3 -- can an unlabelled read tell you where to attach a probe? |
| [`exp10_lens_ranking.py`](exp10_lens_ranking.py) | B5 -- does this pipeline reproduce a published ordering it played no part in establishing? |
| [`exp11_reversal_mechanism.py`](exp11_reversal_mechanism.py) | Hypothesis 3 for the fit-ranking reversal: threshold instability in basis selection. |
| [`exp12_fit_quality.py`](exp12_fit_quality.py) | Is B4's ground truth actually true at every layer? |
| [`exp13_ranking_sensitivity.py`](exp13_ranking_sensitivity.py) | B4' -- the resolution limit of coverage as a transport-quality read. |
| [`exp14_blend.py`](exp14_blend.py) | Why fit sample size inverts the coverage read: two objectives, not one error. |
| [`exp15_delta_direction.py`](exp15_delta_direction.py) | What is coverage actually responding to when it prefers the larger fit? |
| [`exp16_gain.py`](exp16_gain.py) | Is the disagreement a gain mismatch? |
| [`exp17_affine.py`](exp17_affine.py) | The ground truth was missing a term: a Jacobian linearisation is affine, not linear. |
| [`exp18_prenorm_target.py`](exp18_prenorm_target.py) | sec 7.7 was scored against the wrong target. This corrects it. |
| [`exp19_recalibrate.py`](exp19_recalibrate.py) | B4' recalibrated on the corrected ground truth, and the `resolves_gap` threshold derived. |
| [`exp20_second_model.py`](exp20_second_model.py) | Does the linearisation-reach curve hold at a second width? |
| [`exp21_prompt_spread.py`](exp21_prompt_spread.py) | How much of each headline number is a mean, and how wide is the sample behind it? |
| [`exp22_coverage_report.py`](exp22_coverage_report.py) | The coverage figure the report leads with, as a script that can be re-run. |
| [`exp23_lens_compression.py`](exp23_lens_compression.py) | What the coupling buys: a Jacobian lens applied at a fraction of its cost, at the same read. |
| [`exp24_stream_compression.py`](exp24_stream_compression.py) | The compressible object is the stream, not the transport -- tested causally. |
| [`exp25_concept_transport.py`](exp25_concept_transport.py) | Carry a direction one model resolved into another model, and act with it. |
| [`exp26_cross_tokenizer.py`](exp26_cross_tokenizer.py) | A crossing between two models that share neither a tokenizer nor a width. |
| [`exp27_transport_specificity.py`](exp27_transport_specificity.py) | Does the crossing carry *which* structure, or merely some structure? |
| [`exp28_shared_basis.py`](exp28_shared_basis.py) | Stream compression with a well-sampled basis, fitted offline and applied held-out. |
| [`exp29_constant_sweep.py`](exp29_constant_sweep.py) | Every numeric constant in the path, found and classified. |
| [`exp30_context_handoff.py`](exp30_context_handoff.py) | One vector instead of a context: what a crossing is worth in tokens. |
| [`exp31_prefix_handoff.py`](exp31_prefix_handoff.py) | Hand the receiver tokens it attends to, rather than a perturbation added to its stream. |
| [`exp32_derived_early_exit.py`](exp32_derived_early_exit.py) | Early exit with no threshold: stop when the update falls below the frame's own noise floor. |
| [`exp33_kv_ceiling.py`](exp33_kv_ceiling.py) | How compressible is a KV cache, before any rule is built to compress it. |
| [`exp34_subspace_retrieval.py`](exp34_subspace_retrieval.py) | Retrieval by subspace overlap rather than by a cosine between two points. |
| [`exp35_grounding.py`](exp35_grounding.py) | Can a geometric read catch what the model's own probability gets confidently wrong? |
| [`exp36_structure_before_training.py`](exp36_structure_before_training.py) | Is the structure training keeps already present before training? |
| [`exp38_composition.py`](exp38_composition.py) | Do many written directions compose, or does the tenth destroy the first? |
| [`exp39_lexicon_bridge.py`](exp39_lexicon_bridge.py) | The two lexicons are already connected, by a key that is already in the store. |
| [`exp40_benefit.py`](exp40_benefit.py) | Does an injected ontology entry make the answer BETTER, or only different? |
| [`exp49_quantization_damage.py`](exp49_quantization_damage.py) | Which layers did quantisation break? The reads applied to ordinary weights, not to a lens. |
| [`exp50_quantization_cost.py`](exp50_quantization_cost.py) | P6: does the agreement score predict what quantising a layer costs the model? It does not. |
| [`exp51_the_claim.py`](exp51_the_claim.py) | THE CLAIM: the identity in a lens sets the floor a spectral read of it is counted against. |
| [`exp52_what_the_hidden_modes_name.py`](exp52_what_the_hidden_modes_name.py) | What the modes the identity hides actually name, in tokens. |
| [`exp53_hidden_modes_reproduce.py`](exp53_hidden_modes_reproduce.py) | Do the modes the identity hides name the same tokens in two independent fits? |
| [`exp54_sweep_and_depth.py`](exp54_sweep_and_depth.py) | The two robustness tables of section 1: the false-alarm sweep and the depth profile. |
| [`exp55_structure_free_control.py`](exp55_structure_free_control.py) | How much of the change in resolved count is a property of the transport? |
| [`depth_report.py`](depth_report.py) | The depth-profile page: one curve per model, plotted against relative depth. |
| [`fetch_lens.py`](fetch_lens.py) | Download a published Jacobian lens. |
| [`live_report.py`](live_report.py) | A self-contained local page that fills in while the run is going. |
| [`make_synthetic_lens.py`](make_synthetic_lens.py) | A synthetic lens with a planted workspace band -- the rehearsal rig for experiment 1. |
| [`plot_reach.py`](plot_reach.py) | How far a mean-Jacobian linearisation carries, by depth, as inline SVG. |
| [`plot_sensitivity.py`](plot_sensitivity.py) | The sensitivity curve of B4' as inline SVG, for embedding in the report. |
| [`plot_streams.py`](plot_streams.py) | A self-contained comparison page for the stream reads of experiment 4. |

Numbering is chronological and has gaps. `exp37` imported an adapter from a sibling repository
and `exp41`-`exp48` read a private store, so none of them runs from a checkout of this one and
all were removed from the published tree. Their results stand in the paper and the scripts are
in git history. A negative result is a deliverable here -- several of these ran, failed, and are
kept for that reason.
