# Human review instructions: harder scaffold-dropout DPO candidates

You are reviewing DPO pair candidates created from a scaffold-dropout experiment.

The system intentionally weakened the hidden runtime scaffold while preserving the empirical public-opinion substrate. The rejected response is an SFT checkpoint-100 generation under that weak scaffold. The proposed chosen response is the human-revised target from the full scaffold data.

Approve a pair only when both statements are true:

1. The rejected SFT100 weak-scaffold response is a real failure for the user's task and supplied empirical substrate.
2. The proposed chosen response is a valid final public-facing answer under the same weak-scaffold prompt.

Approve real failures such as:

- unsupported population/subgroup transfer;
- false mainstream, majority, dominant, or consensus framing;
- erasure of measured minority, status-quo, neutral, civil-union, guest-worker, or conditional-need positions;
- qualitative prevalence reversal or serious weakening;
- exact/approximate numeric leakage where the public-facing policy is qualitative-only;
- hidden-plan/schema/source leakage;
- over-refusal when a bounded useful answer is possible;
- severe public-facing style breakdown that would not be acceptable to train on.

Do not approve merely because the weak-scaffold answer sounds slightly less polished than the chosen response. Mild style issues are not enough.

Special rule for `H9_insufficient_prevalence_control`:

- These rows intentionally remove prevalence/order information and are diagnostic controls. They should normally not be used for DPO unless the correct chosen response is an abstention/bounded-uncertainty answer. Do not approve a row that asks the model to preserve prevalence after that information has been removed.

Fill these columns:

- `human_rejected_is_real_failure_1_5`: 5 = clear real failure, 1 = not a failure.
- `human_chosen_is_good_for_weak_prompt_1_5`: 5 = valid target answer, 1 = not valid for this weak prompt.
- `human_best_chosen_response_if_revision_needed`: optional revision if the proposed chosen response is semantically right but needs wording changes.
- `human_approve_for_dpo_yes_no`: yes only if the pair should be used for DPO.
- `human_rejection_tags`: concise tags if not approved.
- `human_notes`: brief rationale.

Suggested rejection tags:

- rejected_not_actual_failure
- chosen_not_valid_for_weak_prompt
- missing_empirical_information_in_prompt
- only_mild_style_issue
- numeric_leakage_not_actually_bad_for_prompt
- insufficient_control_not_dpo_eligible
- length_or_format_confounded
- scaffold_leakage
- wrong_population_transfer
- false_prevalence
- stance_erasure
- weak_qualitative_prevalence
