# Compare outputs with CodeContests' token comparator, not exact match

Output matching splits both strings on any whitespace, drops empty tokens, compares
token-by-token case-insensitively, and applies a 1e-5 absolute tolerance when either token
parses as a float. Token counts must match. This reimplements `OutputsMatch` from DeepMind's
own evaluator.

Exact string comparison is the obvious implementation and is wrong: it scores correct
solutions as failures over trailing whitespace, line breaks, capitalisation, and float
formatting. Because those failures are indistinguishable from genuine wrong answers, the
error is silent and would corrupt every reward in the system.

## Consequences

Do not substitute LiveCodeBench's comparator, which is case-sensitive, line-strict, and uses
exact `Decimal` comparison. Under LCB semantics, gold solutions that DeepMind accepts score
zero — and at least one widely-used open implementation judges CodeContests this way.

Two known sharp edges inherited from the reference implementation: the float tolerance is
absolute rather than relative, which is dangerous at large magnitudes; and 64-bit integer
answers fall through to the float path because the reference parses integers as 32-bit.
