# Why spatial reasoning needs evidence selection: MSSR

Research note accompanying **Pursuing Minimal Sufficiency in Spatial Reasoning** (ICLR 2026), by Yejie Guo, Yunzhong Hou, Wufei Ma, Meng Tang and Ming-Hsuan Yang.

## The question

How can a vision-language model answer a spatial question when its images contain incomplete geometry and its context contains distracting information? MSSR treats this as a joint problem of acquiring evidence and deciding which evidence belongs in the final reasoning context.

A useful motivating example is asking what lies to a person's left when they face a doorway. Object positions alone are insufficient: the answer depends on a reference direction and a coordinate transformation. At the same time, a long inventory of every visible object can obscure the facts needed for this particular question. This is an illustrative example, not an additional benchmark result.

## From images to a curated evidence set

MSSR takes multiple views of a scene and a natural-language question. Its Perception Agent writes Python programs that call vision and geometry tools. VGGT supplies reconstruction; GroundingDINO and SAM2 help localize objects; numerical tools support spatial calculations. Execution state is preserved so later requests can build on prior work.

The Reasoning Agent develops a question-specific plan, retains relevant evidence, and requests information that is still missing. The agents repeat this exchange until the reasoning stage decides to answer. Final reasoning discards previous context and uses the selected evidence. The ideal Minimal Sufficient Set is a target that the system approximates, not a proven optimum for every input. [Method, Section 3](https://arxiv.org/html/2510.16688v2#S3)

## Why not keep every perception result?

The paper tests this question with a controlled analysis of a subset of MMSI-Bench problems solved in three iterations. Critical information discovered later is added retrospectively to earlier sets, to normalize sufficiency before comparing set sizes. Average set size decreases from 17.3 to 5.9 elements while reasoning accuracy rises from 45.8% to 48.3%. This supports the importance of selecting relevant evidence in this experimental setting; it is not a universal law that shorter contexts always perform better. [Section 4.3](https://arxiv.org/html/2510.16688v2#S4.SS3)

MSSR operates on structured spatial facts. This differs from pruning visual tokens inside a vision encoder, compressing a model's weights, or shortening a generated answer. Those techniques address different representations and should not be treated as equivalent baselines without checking the task and protocol.

## How does Situated Orientation Grounding work?

Spatial questions often refer to a direction through language: the front of an object, or the direction associated with an action. SOG represents candidate directions as arrows rendered in images and asks a VLM to select among them. It refines candidates from coarse to fine and uses an additional rendered view to reduce ambiguity. This makes language-conditioned 3D orientation a visual selection problem. It remains dependent on perception quality and is not intended for sub-degree pose estimation. [Section 3.2](https://arxiv.org/html/2510.16688v2#S3.SS2)

## Results and responsible comparisons

In arXiv v2 Table 1, MSSR with GPT-4o reports 49.5% on MMSI-Bench and 51.8% on ViewSpatial-Bench; the GPT-4o baseline reports 30.3% and 35.0%. These correspond to gains of 19.2 and 16.8 percentage points under the paper's settings. They are historical reported measurements, not a live leaderboard claim. [Table 1](https://arxiv.org/html/2510.16688v2#S4.T1)

The main inference framework is training-free. A separate preliminary experiment uses MSSR-derived traces for supervised fine-tuning. Those two settings should be distinguished when discussing data requirements or selecting a baseline. Researchers reproducing results should record the exact model version, benchmark split, subset selection, tool checkpoints and iteration settings. The README documents the available runners; results have not been independently reproduced for this note.

## When this work is relevant

- A related-work section on zero-shot, tool-augmented multi-view spatial reasoning.
- A study of how agents request missing perceptual evidence and filter irrelevant context.
- Language-grounded 3D orientation and reference-frame reasoning.
- Analysis of interpretable spatial evidence and generated reasoning traces.

Reconstruction and grounding errors remain important limitations. Iterative calls add latency, and the system can decide prematurely or make logical errors despite obtaining useful evidence. The paper discusses these failure modes and future verification mechanisms. [Appendices H and K](https://arxiv.org/html/2510.16688v2#A8)

[Code and setup](https://github.com/gyj155/mssr) · [Paper](https://arxiv.org/abs/2510.16688) · [Conference record](https://openreview.net/forum?id=bZAKJwyn1n) · [Citation](../CITATION.bib)
