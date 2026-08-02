# CodeContests RLVR

Reinforcement learning with verifiable rewards for competitive programming: a policy is
trained to write solutions, and the reward comes from executing those solutions against
test cases rather than from any learned model of quality.

## Language

### The learning loop

**Rollout**:
One sampled solution attempt at one problem.
_Avoid_: sample, trajectory, episode

**Completion**:
The raw text a rollout produced, before any code has been recovered from it.
_Avoid_: output, response, generation

**Group**:
The set of rollouts sampled for the same problem within one step. Advantage is computed
relative to the group, so the group — not the rollout — is the unit that carries learning
signal.
_Avoid_: batch

**Advantage**:
A rollout's reward expressed relative to the mean and spread of its own group.

**Degenerate group**:
A group whose rollouts all received the same reward. Its advantage is zero, so it
contributes no gradient despite having cost a full set of rollouts.
_Avoid_: dead group, zero-variance group, collapsed group

**Dynamic sampling**:
A rollout policy that discards degenerate groups and generates replacements rather than
training on them.

### Grading a rollout

**Verifier**:
The component that executes a rollout's code against tests and reports what happened. It
never assigns a reward.
_Avoid_: environment, sandbox, executor, judge

**Scorer**:
The component that turns a verifier's report into a reward. It never executes anything.
_Avoid_: reward model, grader, judge

**Reward function**:
One named, interchangeable rule for turning a verifier report into a number. Several exist;
exactly one drives training in any given run.

**Reward registry**:
The collection of available reward functions, each recorded with the source it came from.

**Extraction tier**:
Which rule succeeded in recovering code from a completion — from a language-tagged fenced
block at the top, down to nothing recovered at all.

**Comparator**:
The rule that decides whether a program's output counts as matching the expected output.

### The problem corpus

**Public tests**:
Test cases printed in the problem statement, and therefore visible to the model.

**Private tests**:
Contest test cases withheld from the model. The most trustworthy pool.

**Generated tests**:
Test cases synthesised by mutating existing inputs, validated only by consensus among human
solutions and known to include invalid cases.

**Multiple-output problem**:
A problem admitting more than one correct answer, but stored with a single expected answer.
A correct solution that emits a different valid answer is graded as wrong.

**Interactive problem**:
A problem requiring bidirectional exchange with a judge rather than reading input and
writing output once.
