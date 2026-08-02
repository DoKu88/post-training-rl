# Grade on private tests first, generated tests only as filler

A rollout is graded against up to 15 tests drawn private-tests-first, longest-input-first,
topped up from generated tests only when a problem has too few private ones. Problems that
cannot reach 5 total tests are dropped. Public tests are executed separately and never feed
the primary reward.

The pools are not equally trustworthy. Generated tests were produced by mutating existing
inputs and validated only by consensus among 30 human solutions; the maintainers concede
they may be invalid, and the generation code was never released. AlphaCode's own measurement
puts the false-positive-or-slow rate for the shipped test suites at **46%**. Public tests are
printed in the problem statement, so a reward computed over them is directly hackable — the
model can learn to print the answers it was shown.

## Consequences

The 5-test floor follows DeepCoder's finding that problems with fewer tests encourage the
model to print memorised answers.

Capping at 15 bounds execution cost per rollout, which would otherwise dominate wall-clock:
CodeContests problems can carry hundreds of generated tests. Longest-input-first is rLLM's
heuristic — longer inputs catch more bugs per execution.

The distribution of private-test counts across the training split is unmeasured. If it turns
out most problems carry fewer than 15, the generated-test filler will be doing more work than
intended and this decision should be revisited.

Captured stdout is capped at 10 MB per execution and compared as truncated rather than
auto-failed; `--rlimit-fsize` does not apply to pipes, so without a parent-side cap a runaway
print loop can exhaust memory in the training process itself.
