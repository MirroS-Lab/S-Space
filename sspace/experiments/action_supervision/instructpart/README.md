# InstructPart action-supervision experiment

## Status and conclusion

This quantitative experiment is included in the shared experiment registry.
Its pseudo ground truth still has
`review_status=pending`, so results remain **provisional** until the generated
review panels have been inspected and approved.

At layer 18, MolmoAct2 improved depth-direction accuracy over Molmo2-ER from
63.07% to 69.02% (+5.96 percentage points; image-clustered 95% bootstrap CI
+0.89 to +10.93). MolmoAct2-Pretrain reached 67.11% (+4.04 points; CI -1.07 to
+9.16). This supports a depth-specific MolmoAct2 advantage in this setting; it
does not support a general overall-accuracy advantage, and layer 18 belongs to
a diagnostic test sweep rather than an independently selected layer.

## Research question

For an image containing an object and an annotated object part, does the
S-Space displacement between the part token and whole-object token point in the
same H/V/D direction as the annotated part relative to the whole object? The
comparison follows the training progression from Molmo2-ER through
MolmoAct2-Pretrain to MolmoAct2.

This is an explicit named-part localization test. The prompt tells the model
the part name. It does not test whether a model can infer an unnamed affordance
region.

## Data and pseudo ground truth

The source is `IffYuan/InstructPart` at revision
`bcd06969e32582ceeba3e1841183106027b379fb`. The deterministic selection first
takes 360 rows by round-robin sampling over `(object, part, affordance)` groups
with seed 42. Grounding DINO Tiny detects the whole object and retains rows only
when the detected box covers at least 80% of the supplied part mask. The
historical filter retains 321 candidates. SlimSAM receives the detected box and
the first 300 valid rows become the experiment set.

For a binary mask `M`, its 2-D coordinate is the arithmetic mean of its pixel
coordinates. The whole-object coordinate uses the SlimSAM object mask, which is
unioned with the supplied part mask. Horizontal and vertical ground truth are:

```text
dx = (part_centroid_x - object_centroid_x) / object_mask_width
dy = (part_centroid_y - object_centroid_y) / object_mask_height
```

Depth Anything V2 Small predicts relative inverse depth for the original image
and a horizontal flip. The flipped prediction is restored and the two maps are
averaged. The whole-object depth reference excludes the annotated part:

```text
body_mask = object_mask AND NOT part_mask
dz = (mean_depth(part_mask) - mean_depth(body_mask)) / (q95(depth) - q05(depth))
```

Larger inverse depth means closer. A direction is ambiguous when `abs(dx)`,
`abs(dy)`, or `abs(dz)` is below 0.05. Depth is also ambiguous when the original
and flipped predictions disagree in sign. The resulting valid counts are 178
horizontal, 238 vertical, and 225 depth images.

## Model input and projection

Every image is forwarded twice per template: once with the whole-object name as
the target and once with the dataset-provided part name. The action and object
fields come from InstructPart. For the five action-context templates, code
renders those fields into a fixed grammatical action phrase. The measured
state is the last subtoken overlapping the exact target character span.

| ID | Group | Exact template |
|---|---|---|
| T01 | context-free | `Point to {target}.` |
| T02 | context-free | `Find the {target}.` |
| T03 | context-free | `Locate the {target}.` |
| T04 | context-free | `Look for {target} in the image and show me where they are.` |
| T05 | context-free | `Please find {target} and show me where they are.` |
| T06 | action context | `Given the instruction "{instruction}", locate the {target}.` |
| T07 | action context | `The robot needs to {instruction}. Find the {target}.` |
| T08 | action context | `The robot has been asked to {instruction}. Point to the {target}.` |
| T09 | action context | `The robot's task is to {instruction}. Help me find the {target}.` |
| T10 | action context | `To carry out the instruction "{instruction}", find the {target}.` |

Example for action `open`, object `bottle`, and part `cap` under T06:

```text
Given the instruction "open the bottle", locate the bottle.
Given the instruction "open the bottle", locate the cap.
```

At every exported post-block layer, the experiment computes:

```text
object_coordinate = axes @ object_target_state
part_coordinate   = axes @ part_target_state
margin            = part_coordinate - object_coordinate
```

The fixed positive directions are right, below, and close. A prediction is
correct when the margin and GT have the same strict sign. An exact zero margin
is incorrect.

## Evaluation protocol

The report covers layers 17--22. Each of the ten templates receives equal
weight. Overall accuracy pools every valid template/image/axis decision.
For each model and layer, `across_prompt_summary` reports the mean accuracy
and 95% Student-t interval across the ten prompts:
`mean +/- t(0.975, 9) * std(ddof=1) / sqrt(10)`.
Values use accuracy units (0--1); intervals are not clipped. The depth entries
at layer 18 reproduce the historical blog means and error bars from the saved
per-template scores. This formula matches the published bars numerically;
the original plotting script was not recovered.

Paired comparisons keep all ten templates for an image together and bootstrap
images 10,000 times with seed 42. This avoids treating ten prompt variants of
one image as ten independent samples. These intervals describe model
differences, not the blog's per-model error bars.

Historical layer-18 results:

| Model | Horizontal | Vertical | Depth | Overall |
|---|---:|---:|---:|---:|
| Molmo2-ER | 82.75% | 88.57% | 63.07% | 78.00% |
| MolmoAct2-Pretrain | 78.71% | 86.89% | 67.11% | 77.68% |
| MolmoAct2 | 78.26% | 84.92% | 69.02% | 77.49% |

MolmoAct2 depth accuracy is higher than Molmo2-ER for every individual
layer-18 template (67.56%--70.67% versus 60.89%--65.78%). The aggregate
horizontal and vertical scores are lower, which is why the overall score does
not improve.

## Reproduction

From the repository root, install the standard model environments, prepare
the three pseudo-GT models, and run the experiment:

```bash
./scripts/prepare_models.sh action_supervision
./scripts/run_experiment.sh --experiment action_supervision
```

Review panels are written below
`.cache/assets/datasets/instructpart_affordance_parts/pseudo_ground_truth/review_panels/`.
Do not change `review_status` to `passed` before completing that review.

For a bounded chain that cannot overwrite the full output:

```bash
./scripts/run_experiment.sh --workflow action_supervision --limit 10
```

The full result is
`outputs/action_supervision/instructpart_prompt_ensemble/results.json`. The
bounded result is written under
`outputs/small/action_supervision/instructpart_prompt_ensemble/limit_10/`.

The single source of truth is
`configs/experiments/action_supervision/instructpart_prompt_ensemble.json`.
Source images, masks, model weights, generated masks, depth maps, and run
outputs remain outside Git.

## Historical regression identity

The migration is bound to the following historical files:

| Artifact | SHA-256 |
|---|---|
| 360-row source selection | `cb6d668645f7e5bbf40bb34f0ee00af95940141c89fbfee6e9dacf9c13d08c71` |
| 321-row filtered selection | `0a81e740ae35caee96d8f5ee7ca3eba34495a8f93fdc2f71c569bbe7c4e579ac` |
| Grounding DINO boxes | `0741621b7de2eaf10a639ab1198cb66baaa1eaf8d4d6623387c3389102c638f2` |
| 300-row retained selection | `75c592e590d5b37f55d0e38915b851e4d774f0b28f9157f5e72769cd10ded174` |
| Portable pseudo-GT values | `c0168d376730e167ac629dbddc686adc3f09e3c9f15db7ac94b98eba517eb82e` |
| Historical full score report | `0e0e7094a2fd5de6e6ca392bc0a648f87e68732b607daea540b0ead446bc8757` |

The portable pseudo-GT digest excludes only `depth_map` paths and review
status. Paths are now relative, and a later human approval may legitimately
change review status without changing numerical GT.
