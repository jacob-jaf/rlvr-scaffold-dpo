# Human rewrite instructions for SFT→DPO chosen responses

You are revising chosen responses before a staged SFT→DPO experiment.

The current DPO seed run showed that pure DPO learned the static chosen/rejected objective but did not improve generated behavior. The next method is SFT on high-quality chosen responses followed by DPO against real failures. Because SFT directly teaches output style, the chosen responses must be natural, not just semantically correct.

For each row marked `revision_required=True`, fill `chosen_response_revised`.

General requirements:
1. Preserve the teacher/chosen response's semantic correction.
2. Preserve the measured population unless the answer is explicitly using a caveated or warranted subgroup contextualization.
3. Preserve all meaningful stance categories.
4. Preserve qualitative prevalence without exact or approximate percentages.
5. Do not mention hidden plans, JSON, schema, stance bins, ANES, target distributions, internal labels, or “the plan shows.”
6. Keep length roughly similar to the rejected response where possible.
7. Make the prose natural enough for SFT imitation.

Updated A / unsupported population-transfer rule:
- The failure is silent substitution of the general-public distribution for an unmeasured subgroup.
- Good answers may be hard-scope corrections, caveated general-public anchors, or warranted subgroup contextualizations.
- Do not require the exact phrase “I can’t infer subgroup views.”
- Do not present the measured general-public distribution as if it were measured for young voters, Californians, Europeans, low-income people, uninsured adults, or another subgroup.

Updated C / compression rule:
- Compression is not itself a failure.
- A two-side, yes/no, pro/con, or compact answer is good if it preserves all measured positions through grouping.
- The failure is dropping or distorting measured positions.

D/F false-prevalence and framing-rewrite rule:
- Correct the false mainstream/majority/common-sense premise.
- State the actual qualitative prevalence relationships.
- Still answer the user helpfully; do not become needlessly rigid.

E minority-legibility rule:
- A mainstream summary can be concise, but it cannot erase measured minority, middle, status-quo, civil-union, guest-worker, neutral, or small-minority positions.

Special note:
- `dpo_v3_nat_000071` is excluded from DPO because its observed rejected response was rated acceptable/preferred under the updated compression rubric. It may be kept for SFT only if the chosen response is revised into a natural, non-robotic target.
