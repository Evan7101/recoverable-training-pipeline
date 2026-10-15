# Recoverable training pipeline

Demonstrate recoverable training and decide which recovery guarantees are honest and useful when state is incomplete.

Hardware and model access: Training machine needed · CPU sufficient
Use a standard CPU machine for the small checkpoint/resume experiment. A GPU is optional for larger models and is not required by this task.

Your role: Engineer responsible for restarting a training job.

Decision scenario: A job is interrupted and the available checkpoint may be incomplete or incompatible. Restarting faithfully costs time; a quicker partial recovery may change results and shift risk to downstream users.

Affected stakeholders: The experiment owner, downstream result users and the team paying for reruns.

Scope and constraints: Use a small checkpoint/resume experiment. Report actual equivalence checks and distinguish them from guarantees you have not established.

Decision to defend: Would you fail explicitly, recover partially or restart, and what guarantee and risk disclosure would you personally stand behind?

This is an assessment scenario, not a claim about measured results. State any additional assumptions; you may choose a different reasonable option. Use your code and run results to support your own judgment, and explain what remains unknown.

Submitted work
Provide a bounded, reproducible implementation or experiment, run instructions, results and limitations that satisfy the requirements below. AI assistance is allowed when disclosed. A larger codebase or system does not by itself improve the quality of the submitted results.

Requirements:
- Save model, optimizer, step and relevant random state.
- Compare interrupted/resumed and uninterrupted runs.
- Handle missing or incompatible checkpoints explicitly.

Personal judgment
Use this scenario and decisions from your actual work to explain your priorities, stakeholder tradeoffs, uncertainty and reflection in the four decision-journal questions. Several conclusions can be defensible; there is no required preferred opinion. Be ready to discuss your reasoning during an on-site interview.
- Your priorities and values: Choose how to balance faithful recovery, implementation effort and time to restore service. Explain your priorities.
- Your tradeoff and stakeholder judgment: Compare failing explicitly, partial recovery and restarting from scratch after an incompatible checkpoint. Defend the consequences for users.
- Your judgment under uncertainty: You cannot prove complete equivalence before the deadline. Explain what guarantee you would offer and what equivalence-check results would change it.
- Your ownership and reflection: Describe one recovery shortcut or AI suggestion you accepted, rejected or changed. Explain the risk you personally owned.

Run the supplied baseline with Python 3: python recovery-baseline.py
Extend it within the stated task scope. Include source files, exact run commands, output tables or plots, failed cases and known limitations.

