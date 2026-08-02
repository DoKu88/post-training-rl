# Format adherence and code extraction in code RLVR — what practitioners actually do

**Research date: 2026-08-01.** Every claim below links to a primary source (library source on GitHub, official docs, arXiv, or a HF dataset/model card). Where a repo has releases, permalinks are pinned to a tag; where it does not, they are pinned to the commit SHA that was `HEAD` when I read the file, because `main` moves.

Companion to [`rlvr-stack.md`](./rlvr-stack.md). That document covers the trainer, the sandbox and the reward *shape*; this one covers the narrower question of getting a runnable string out of the model's mouth in the first place.

Commits/tags read for this document:

| Repo / artifact | Pinned at | Last push visible |
| --- | --- | --- |
| `huggingface/trl` | `v1.9.2` | released 2026-07-28 |
| `huggingface/open-r1` | [`1416fa0`](https://github.com/huggingface/open-r1/tree/1416fa0cf21595d2083b399a2a0bbddd7f6e9563) | 2026-04-02 |
| `agentica-project/rllm` (DeepCoder) | [`7b47687`](https://github.com/agentica-project/rllm/tree/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4) | 2025-09-17 |
| `verl-project/verl` (formerly `volcengine/verl`) | `v0.8.0` | released 2026-06-01 |
| `ganler/code-r1` (the `coder1` path) | [`443f8da`](https://github.com/ganler/code-r1/tree/443f8da8058d07051793eff875da117534fd1ba1) | — |
| `bigcode-project/bigcode-evaluation-harness` | [`8fc5bae`](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd) | 2025-07-22 |
| `LiveCodeBench/LiveCodeBench` | [`28fef95`](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24) | — |
| `evalplus/evalplus` | [`26d6d00`](https://github.com/evalplus/evalplus/tree/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e) | — |
| `huggingface/transformers` | [`b3a3603`](https://github.com/huggingface/transformers/tree/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722) | — |
| `vllm-project/vllm` | `v0.26.0` | released 2026-07-27 |

> **Note on the `verl` URL.** `github.com/volcengine/verl` now 301-redirects to **`github.com/verl-project/verl`**. The GitHub API returns `{"message": "Moved Permanently"}` for the old path unless you follow redirects. Any link you have to `volcengine/verl/blob/...` still resolves, but the canonical repo is `verl-project`.

---

## 1. How open implementations actually extract code

Seven implementations, read line by line. They do **not** agree, and the disagreements are load-bearing.

### 1.1 open-r1 — anchored fence, **last** match, empty string on failure

[`src/open_r1/rewards.py#L476-L482`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L476-L482), verbatim:

```python
def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer
```

Properties, all of which matter:

- **The language tag is mandatory and interpolated into the regex.** A bare ` ``` ` fence matches nothing. So does ` ```Python ` (capital P), ` ```py `, and ` ```python3 ` — wait, `python3` *does* match, because the pattern is not anchored at the closing fence; `rf"```python\n(.*?)```"` will happily match ` ```python3\n… ` only if a literal newline immediately follows `python`, which it does not. `` ```py `` fails. **`` ```python `` followed by anything other than `\n` fails.**
- **Last match wins** (`matches[-1]`).
- **The closing fence is not required to be at a line start.** `(.*?)` + ` ``` ` will stop at the first triple-backtick anywhere, including inside a string literal.
- **Failure returns `""`,** which is then handed to the executor as a program. An empty Python program exits 0 with empty stdout, so it fails the output comparison rather than being flagged as a parse failure.

The reward that consumes it, [`rewards.py#L569`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L569):

```python
code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
```

Note `completion[-1]["content"]` — open-r1 reads the **last message** of the completion, not the first. (Its `format_reward` and `code_format_reward` read `completion[0]["content"]` instead; for single-turn GRPO these are the same object.)

There is also a live `TODO` in the execution template it generates, [`rewards.py#L552`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L552), verbatim:

```python
# TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
```

### 1.2 rLLM / DeepCoder — optional language tag, **last** block, `None` on failure

[`rllm/rewards/code_reward.py#L28-L41`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/code_reward.py#L28-L41), verbatim:

```python
def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()
```

(The docstring says "empty string"; the code returns `None`. Minor, but it is the kind of thing that produces a `TypeError` three layers down.)

This is strictly more permissive than open-r1: `(?:\w+)?` makes the language tag optional and accepts *any* word-character tag, so ` ``` `, ` ```python `, ` ```Python `, ` ```py ` and ` ```cpp ` all match. The `\n` after the tag is still mandatory.

**Crucially, rLLM distinguishes the failure but does not price it differently.** [`code_reward.py#L429-L432`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/code_reward.py#L429-L432):

```python
        model_code = extract_code_from_model(model_response)
        if model_code is None:
            # print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No code found in model response"})
```

and [`rllm/rewards/reward_types.py#L10-L34`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/reward_types.py#L10-L34):

```python
@dataclass
class RewardConfig:
    apply_format_reward: bool = False
    ...
    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0
```

**`format_error_reward == incorrect_reward == 0.0`, and `apply_format_reward` defaults to `False`.** DeepCoder's training entry point uses the defaults — [`examples/deepcoder/train_deepcoder.py`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/examples/deepcoder/train_deepcoder.py) passes `env_args = {"reward_fn": code_reward_fn}`, and [`rllm/rewards/reward_fn.py`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/reward_fn.py) constructs a bare `RewardConfig()`. So **DeepCoder ran a full 14B code-RL campaign with no format reward and no format penalty at all.** Its base model was `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` ([`train_deepcoder_16k.sh`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/examples/deepcoder/train_deepcoder_16k.sh)).

### 1.3 verl — two different policies in the same repo, and one of them has no failure branch

verl routes CodeContests through `prime_code` or `sandbox_fusion` depending on whether a sandbox URL is configured. [`verl/utils/reward_score/__init__.py#L74-L88`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/__init__.py#L74-L88), verbatim:

```python
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
```

**`prime_code` — split, not regex, and the fallback is "treat the whole completion as code".** [`verl/utils/reward_score/prime_code/__init__.py#L21-L23`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/prime_code/__init__.py#L21-L23), verbatim:

```python
def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split("```python")[-1].split("```")[0]
```

Read that carefully. If there is no `` ```python `` anywhere, `split` returns a one-element list, `[-1]` is the entire completion, and `.split("```")[0]` is everything before the first backtick fence. The comment says the quiet part out loud: *"if the completion is pure code, this will not take effect."* **There is no "no code found" branch.** Prose gets shipped to the interpreter and fails with a `SyntaxError`, scoring 0 — indistinguishable from a wrong answer.

`prime_code` also has a latent `UnboundLocalError`: with `continuous=False` and a failing first check, neither `success` (well, `success` is bound at L35) nor `metadata_list` is assigned before `return success, metadata_list` at L73, and that return is **outside** the `try` block. verl always calls it with `continuous=True`, which assigns both, so the bug is unreachable from `default_compute_score` — but do not copy this function.

**`sandbox_fusion` — a three-tier cascade that *does* have an explicit failure branch.** [`verl/utils/reward_score/sandbox_fusion/__init__.py#L47-L61`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/sandbox_fusion/__init__.py#L47-L61), verbatim:

```python
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]
```

This is the most defensive extractor of the seven: try `` ```python `` (last one), else *any* fence (first one), heuristically strip a language tag with `first_line.strip().isalpha()`, else return a **typed error**. Note the inconsistency: the `` ```python `` branch takes the **last** block, the generic-fence branch takes the **first**.

Note also `.isalpha()` rejects `python3` and `c++`. And with `continuous=True` verl scores only the first 10 test cases (`num_to_consider = min(len(res_list), 10)`, [L98](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/sandbox_fusion/__init__.py#L98)).

### 1.4 code-r1 / `coder1` — the outlier: **concatenate every block**, and punish format failure hard

`ganler/code-r1` is the fork verl's ecosystem borrows the "coder1" execution path from (rLLM's firejail sandbox carries its URL in a comment — see [`rlvr-stack.md` §5.A.4](./rlvr-stack.md)). Its extractor, [`verl/utils/reward_score/coder1/__init__.py#L54-L60`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L54-L60), verbatim:

```python
CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)


def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()
```

Three things no one else does:

1. `try_extract_solution` first narrows to the **last `<answer>…</answer>` span** if present ([L41-L51](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L41-L51), credited in a comment to [Logic-RL](https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py)), falling back to the full string.
2. It **joins all remaining blocks with newlines** rather than picking one. That is right for "imports in one block, solution in another" and catastrophically wrong for "here's the naive version, here's the optimised version".
3. The closing fence must be preceded by `\n` (`(.*?)\n```{3}`), which is stricter than open-r1's and rLLM's.

The reward scale, [`coder1/__init__.py#L63-L127`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L63-L127), verbatim in the important places:

```python
def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    reward_log = []

    # ground_truth is not code, but tests
    pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)

    if not pass_fmt or len(solution_code) == 0:  # only print full output when there is an error
        ...
        return -answer_reward - format_reward, "\n".join(reward_log)
```

…and on the two success paths, `return format_reward, …` (tests failed) and `return format_reward + answer_reward, …` (tests passed). So the reward is a **three-level scale: −1.1 / +0.1 / +1.1**. A format failure is not merely worth less than a wrong answer, it is *actively punished* relative to one, by 0.2. This is the only implementation I found that guarantees a format-failing rollout has a different reward from a merely-wrong rollout — i.e. the only one that structurally prevents an all-format-failure group from being reward-degenerate alongside all-wrong groups.

`validate_response_structure` is a `<think>…</think>…<answer>…</answer>` check ([L37-L39](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L37-L39)):

```python
def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))
```

### 1.5 LiveCodeBench — line-scan, **penultimate-to-last fence**, `""` on failure

[`lcb_runner/utils/extraction_utils.py#L4-L17`](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/utils/extraction_utils.py#L4-L17), verbatim:

```python
def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
```

No regex. It finds every **line containing** ` ``` `, and takes what lies between the **last two** such lines — note the commented-out first-block version directly above, i.e. someone deliberately switched from first to last. This is the most robust to a missing language tag (it does not look at tags at all) and the least robust to an odd number of fences: with three fence lines it silently returns the span between #2 and #3, which is prose.

`< 2` fence lines → `""`. Also note the base-model branch: `LMStyle.GenericBase` returns the raw output — i.e. **for base models LiveCodeBench does not attempt fence extraction at all.**

### 1.6 EvalPlus — the only one that ignores fences entirely

EvalPlus does not look for backticks. [`evalplus/sanitize.py#L31-L49`](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/sanitize.py#L31-L49), verbatim:

```python
def code_extract(text: str) -> str:
    # Remove ANSI escape sequences, which was caused by ollama (small qwen3 models, but may be others as well)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])
```

**It returns the longest contiguous run of lines that parses as valid Python.** `syntax_check` is `ast.parse` in a `try` ([`evalplus/syncheck.py`](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/syncheck.py)). This is O(n²) syntax checks in the number of lines, which is why nobody uses it in a hot RL loop — but it is *fence-independent* and therefore cannot be defeated by a missing or malformed fence.

The public entry point layers a tree-sitter pass on top, [`sanitize.py#L173-L177`](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/sanitize.py#L173-L177):

```python
def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code
```

`extract_target_code_or_empty` parses with `tree_sitter_python`, keeps imports plus top-level class/function/assignment definitions, and (given an `entrypoint`) prunes to the transitive dependency closure ([L115-L170](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/sanitize.py#L115-L170)). **For a stdin/stdout CodeContests task the tree-sitter path is wrong** — it drops top-level statements, i.e. the `main()` call and any bare `input()` loop. Only `code_extract` is applicable to us.

### 1.7 bigcode-evaluation-harness — sidesteps the problem with stop words and a prefilled prefix

The APPS task (the closest analogue to CodeContests) is a **completion** task, not a fenced-block task. [`bigcode_eval/tasks/apps.py#L55-L58`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/apps.py#L55-L58) and [L78-L89](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/apps.py#L78-L89), verbatim:

```python
        super().__init__(
            stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
```

```python
        prompt = "\nQUESTION:\n"
        prompt += doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        return prompt
```

and the entire postprocessing, [L95-L108](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/apps.py#L95-L108):

```python
    def postprocess_generation(self, generation, idx):
        ...
        try:
            generation = generation.split("\nANSWER:", 1)[1]
        except IndexError:
            # happens when prompts were very long and got truncated
            pass
        return generation
```

There is no fence. The harness *forces* the format by ending the prompt with `\nANSWER:\n` and stopping generation at the next section marker. That is a prefill in all but name (§4).

For genuinely instruction-tuned models the harness exposes `--instruction_tokens`, described in [`main.py#L80-L83`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/main.py#L80-L83) as *"A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. `<user_message>,<end_user_message>,<assistant_message>`"*, and glues them on in [`bigcode_eval/utils.py#L157-L174`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/utils.py#L157-L174):

```python
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )
```

The matching parser, [`utils.py#L202-L222`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/utils.py#L202-L222), verbatim:

```python
def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]
```

Note `if "```python" in assistant_token:` — the harness explicitly anticipates that you will **put `` ```python `` inside the assistant token**, i.e. prefill the fence. This is the closest thing to a documented endorsement of the prefill trick in an eval harness. Its "no assistant token found" path emits a `warnings.warn` and returns the raw text; **it never reports a count.**

### 1.8 Is there a consensus extraction policy?

| Implementation | Pattern | Which block | Language tag | Fallback when no fence |
| --- | --- | --- | --- | --- |
| open-r1 | `` r"```{language}\n(.*?)```" `` | **last** | **required**, exact | `""` (executed as empty program) |
| rLLM / DeepCoder | `` r"```(?:\w+)?\n(.*?)```" `` | **last** | optional, any `\w+` | `None` → `format_error_reward` (= 0.0) |
| verl `prime_code` | `str.split` | **last** `` ```python `` | required (`` ```python `` literal) | **whole completion executed as code** |
| verl `sandbox_fusion` | `str.split`, cascade | last (`` ```python ``) / **first** (bare) | tolerated via `.isalpha()` | `0.0` + typed error |
| code-r1 / `coder1` | `` r"```(?:\w+)?\n(.*?)\n```" `` | **all, joined** | optional, any `\w+` | `−1.1` (hard penalty) |
| LiveCodeBench | line contains ` ``` ` | **between last two fence lines** | ignored entirely | `""` |
| EvalPlus | none — longest syntactically valid line span | n/a | ignored entirely | best-effort span |

**The consensus, such as it is:**

- **Take the last fenced block, not the first.** Five of seven do (open-r1, rLLM, `prime_code`, LiveCodeBench, and effectively `sandbox_fusion`'s primary branch). LiveCodeBench even has the first-block version commented out directly above the last-block version. The rationale is obvious once you look at real completions: reasoning models quote the problem's example code, sketch a naive version, then give the final one.
  ⚠️ **The most prominent dissenter is Qwen's own evaluation harness**, which uses `re.compile(...).search(text)` — i.e. the **first** block — in [`qwencoder-eval/instruct/multipl_e/chat/evaluate.py#L116-L121`](https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/instruct/multipl_e/chat/evaluate.py). Qwen2.5-Coder's published HumanEval numbers rest on a first-block policy. See §5.2.
- **Accept an optional, arbitrary language tag.** open-r1's hard-coded `` ```python `` is the strictest and the odd one out; rLLM's `(?:\w+)?` is the modal choice.
- **Nobody parses. Everybody regexes or splits.** EvalPlus is the only structural extractor and it is not used in any RL loop I found.
- **There is no consensus at all on what to do when extraction fails.** The five options in the wild are: execute an empty program (open-r1), execute the whole completion (verl `prime_code`), score 0 identically to a wrong answer (rLLM), score 0 with a typed error (verl `sandbox_fusion`), and apply a hard negative penalty (code-r1). Only the last one gives the GRPO group any variance to work with.

---

## 2. Format rewards

### 2.1 DeepSeek-R1 — what it actually rewarded, and what it did not say

From the paper HTML, [arXiv:2501.12948v1 §2.2.2 "Reward Modeling"](https://arxiv.org/html/2501.12948v1) (submitted 2025-01-22), verbatim:

> **Accuracy rewards**: "The accuracy reward model evaluates whether the response is correct. For example, in the case of math problems with deterministic results, the model is required to provide the final answer in a specified format (e.g., within a box), enabling reliable rule-based verification of correctness."
>
> **Format rewards**: "In addition to the accuracy reward model, we employ a format reward model that enforces the model to put its thinking process between `'<think>'` and `'</think>'` tags."

That is the entire specification. Three things follow:

1. **The format reward is about the reasoning delimiters, not about the answer's extractability.** It says nothing about code fences. The *accuracy* reward is what carries the extractability requirement, and it does so through the prompt ("required to provide the final answer in a specified format (e.g., within a box)"), not through a separate term.
2. **The paper gives no numeric weight.** There is no equation combining the two, no coefficient, no ablation. Anyone quoting "DeepSeek-R1 used a 0.1 format weight" is quoting an implementation, not the paper.
3. The training template, [Table 1](https://arxiv.org/html/2501.12948v1), verbatim: *"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively."*

The paper's stated reason for staying rule-based at all, verbatim from the same section:

> "We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline."

And the one place DeepSeek-R1 *does* report the cost of a presentation-oriented reward — the language-consistency reward in §2.3.2, verbatim:

> "To mitigate the issue of language mixing, we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT. **Although ablation experiments show that such alignment results in a slight degradation in the model's performance**, this reward aligns with human preferences, making it more readable."

That is the closest thing in the paper to an ablation of a format-flavoured reward term, and its sign is **negative for capability**. It is a cosmetic reward paid for with accuracy. Worth holding in mind before adding one.

### 2.2 What the code-RL implementations actually weight it at

| Implementation | Format term | Weight | Source |
| --- | --- | --- | --- |
| open-r1, `Qwen2.5-1.5B-Instruct` code GRPO | `format` (`<think>/<answer>` only) | **0.1** vs `code` 1.0 | [`config_demo_code.yaml`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml) |
| open-r1, `Qwen2.5-Coder-7B-Instruct` Codeforces GRPO | `code_format` (tags **+ fence**) | **0.1** vs `cf_code` 1.0 | [`config_codeforces.yaml`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/recipes/Qwen2.5-Coder-7B-Instruct/grpo/config_codeforces.yaml) |
| code-r1 / `coder1` | tags + non-empty extraction | **0.1** vs answer 1.0; **−1.1 on failure** | [`coder1/__init__.py#L63`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L63) |
| rLLM / DeepCoder | none (`apply_format_reward=False`) | `format_error_reward = 0.0` = `incorrect_reward` | [`reward_types.py#L10-L28`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/reward_types.py#L10-L28) |
| verl `prime_code` / `sandbox_fusion` | none | — | [`prime_code/__init__.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/prime_code/__init__.py) |
| TRL itself | ships **`think_format_reward` only** | n/a | [`trl/rewards/__init__.py`](https://github.com/huggingface/trl/blob/v1.9.2/trl/rewards/__init__.py) |

**0.1 is the modal weight, and it is not from any paper** — it is a convention that propagated through open-r1 and code-r1 configs.

Widening beyond code RL, to see whether 0.1 is a code-specific convention (it is not — but the *spread* is enormous):

| Repo | Format term | Values |
| --- | --- | --- |
| TinyZero | `format_score` (parseable `<answer>` + valid equation) | `0` no answer tag / **`0.1`** format only / `1.0` correct — [`countdown.py#L59`](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py) |
| Logic-RL | `format_score = format_reward if format_correct else -abs(format_reward)` | **`+1 / −1`**, added to an answer score of `+2 / −1.5 / −2` — [`kk.py#L171-L197`](https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py). Format is **1/3 of the maximum reward.** Note `answer_reward=1.0` in the signature is dead — the answer values are hard-coded |
| simpleRL-reason (SimpleRL-Zoo) | default mode `mix` has **none** (the format branch is commented out); opt-in `independent` mode | `1.0` correct+boxed / `0.5` correct+unboxed / `−0.5` wrong+boxed / `−1` else — [`hf_math_verify.py#L157-L209`](https://github.com/hkust-nlp/simpleRL-reason/blob/v1/verl/utils/reward_score/hf_math_verify.py) |
| verl's own built-in GSM8K scorer | `format_score` parameter exists but **defaults to `0.0`** | [`gsm8k.py#L52`](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py) |
| OpenRLHF math example | none | `rewards.append(1.0 if is_correct else 0.0)` — [`math_reward_func.py#L38`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/math_reward_func.py) |
| Open-Reasoner-Zero | none | see §2.4 |

So the field ranges from **0.0 to 1/3 of total reward**, with no paper justifying any of it. Treat the number as a free parameter, not a received value. What is *not* free is the structural point in §2.3.

open-r1 is the only implementation with a format reward that actually checks the **code fence**. [`src/open_r1/rewards.py#L595-L617`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L595-L617), verbatim:

```python
def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward
```

Note how brittle this is as a *reward*: it requires the exact byte sequence `<think>\n…\n</think>\n<answer>\n`, a fence with the right tag somewhere inside, and `\n</answer>` at the very end of the string. A completion that is perfectly extractable but omits the `<answer>` wrapper scores 0. **This reward measures template compliance, not extractability**, and the two are only loosely correlated. (This exact class of over-strictness has bitten open-r1 before: [issue #237 "Format reward problem"](https://github.com/huggingface/open-r1/issues/237), opened 2025-02-08, closed — the plain `format_reward` regex was missing `re.DOTALL` and therefore returned 0 for every multi-line completion. The bug is fixed in the current source, which carries `re.DOTALL | re.MULTILINE` at [L89](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L89).)

Also note TRL ships **no** code-format or extractability reward. [`trl/rewards/__init__.py`](https://github.com/huggingface/trl/blob/v1.9.2/trl/rewards/__init__.py), verbatim:

```python
_import_structure = {
    "accuracy_rewards": ["accuracy_reward", "get_cosine_scaled_reward", "reasoning_accuracy_reward"],
    "format_rewards": ["think_format_reward"],
    "other_rewards": ["get_repetition_penalty_reward", "get_soft_overlong_punishment"],
}
```

`think_format_reward`'s pattern is `r"^<think>(?!.*<think>)(.*?)</think>.*$"` ([`trl/rewards/format_rewards.py`](https://github.com/huggingface/trl/blob/v1.9.2/trl/rewards/format_rewards.py)) — reasoning tags only.

### 2.3 The structural argument, which matters more than the weight

The reason to care is not the reward magnitude, it is **group variance**. From [`rlvr-stack.md` §1.4](./rlvr-stack.md), TRL computes `advantages = rewards - mean_grouped_rewards`, so an all-identical group contributes exactly zero gradient, and TRL implements no DAPO-style dynamic sampling.

Given that, the five failure policies in §1.8 divide cleanly:

- **rLLM (`format_error_reward == incorrect_reward == 0.0`), open-r1 (`""` → executes → 0), verl `prime_code` (prose → `SyntaxError` → 0)**: a format-failing rollout is **numerically indistinguishable** from a wrong-answer rollout. On a hard CodeContests problem where all `G` rollouts fail — the common case — the group is degenerate whether the failures were syntactic or semantic. Adding a *separate* format reward function fixes this even at weight 0.1, because TRL's default `multi_objective_aggregation="sum_then_normalize"` sums the weighted terms **before** normalizing ([`grpo_config.py#L767-L778`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L767-L778)), so a mixed group of "no code" and "wrong code" now has non-zero std.
- **code-r1 (−1.1 / +0.1 / +1.1)**: format failure is separated by construction, inside a single reward function.

So the honest framing is: **a format reward's main job in a GRPO loop is not to teach formatting, it is to keep the group from being degenerate.** Whether the model needs to be *taught* the format is §2.4 and §5.

### 2.4 Is a format reward necessary? The literature says *no*, and increasingly says *harmful*

This is the part where the primary sources are unusually clear and unusually one-directional.

**The strongest statement is SimpleRL-Zoo's, and it is a headline finding, not a footnote.** [arXiv:2503.18892v3, "SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild"](https://arxiv.org/abs/2503.18892) (Zeng, Huang, Liu, Liu, He, Ma, He; submitted 2025-03-24). §2.1 "Reward", verbatim:

> "We use a rule-based reward function that assigns +1 for correct answers and 0 for incorrect ones. Unlike prior works (Luo et al., 2025; Chen et al., 2025), **we avoid format-based reward, which may hinder exploration**, particularly for base models struggling with format adherence, as detailed in §3.1."

Key finding 3 in §1, verbatim:

> "**Enforcing rigid format reward (e.g., enclosing answers within boxes) (DeepSeek-AI et al., 2025a) significantly penalizes exploration** (Singh et al., 2023; Wang et al., 2024), particularly for base models that initially struggle with instruction following. This restriction lowers their performance ceiling and often induces overthinking behaviors (Chen et al., 2024). (§3.1)"

§3.1 "Over-Reliance on Format Rewards" in full, verbatim:

> "We find that enforcing strict formatting constraints, such as requiring the final answer to be enclosed in a latex command `\boxed{}`, can hinder model's freely exploration and ultimately degrades performance. This is because many base models cannot follow the format constraint well in the initial stage, and imposing a format reward will penalize many correct explorations. We compare two reward functions: one without format constraints, which rewards responses solely based on answer correctness (our default design in §2.1), and another that strictly enforces formatting by penalizing responses with a reward of -1 if they fail to adhere to the required format.
>
> Figure 6 illustrates weaker models like Llama-3.1-8B struggle under strict formatting requirements, leading to a rapid increase in response length early in training without performance improvement. The model expends excessive effort on adhering to the format but fails to learn how to answer correctly, ultimately resulting in model collapse. Figure 6 (Left) further reveals that **even stronger models, such as Qwen-2.5-7B, which initially comply with formatting constraints, suffer in later training stages.** This includes both performance degradation and a significant reduction in CoT length. These findings highlight that: in a zero RL training setting, rather than imposing rigid formatting rules, we should prioritize maintaining response verifiability while allowing sufficient flexibility for exploration."

(Quoted from the v3 PDF via `pdftotext -layout`; arXiv publishes no HTML for this paper. The abstract, which *is* fetchable, corroborates: *"Leveraging several key design strategies—such as adjusting format reward and controlling query difficulty—we achieve substantial improvements…"*.)

⚠️ **Read the scope carefully.** This is an ablation of a **−1 hard format penalty** on **base** models in a **zero-RL** setting. It is *not* a study of a small positive format bonus on an instruction-tuned model. The Qwen-2.5-7B sentence is the part that generalises furthest toward our setting, and even that is Qwen2.5-**Base**, not `-Instruct`. Figure 6 is a plot; **the paper states no numbers** for the format-reward comparison.

**Open-Reasoner-Zero reaches the same conclusion independently.** [arXiv:2503.24290](https://arxiv.org/abs/2503.24290) §2.1.2, verbatim:

> "Unlike DeepSeek-R1-Zero, our scale-up RL training employs **a simple minimalist rule-based reward function that solely checks answer correctness, without any additional format rewards**."
>
> "we implement a binary reward scheme - awarding a reward of 1 for exact matches with the reference answer, and 0 for all other cases."

and §2.1.3, verbatim — the single most relevant sentence in this entire document for our question:

> "**Surprisingly, we found that with our designed prompt, even unaligned base model can yield well-formatted responses in high probability.**"

Its implementation matches ([`playground/orz_7b_ppo.py#L239-L245`](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/playground/orz_7b_ppo.py), verbatim — I re-fetched this myself):

```python
            # only correct and stoped response can aquire reward
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
            else:
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)
```

**DAPO uses no format reward either.** [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)'s reward is `R(ŷ,y) = 1 if is_equivalent(ŷ,y) else −1`; its only shaping term is *Overlong Reward Shaping*, which is about length, not format. It solves the extractability problem in **data preparation** instead — *"we select and transform the answers into integers, which are easy to parse."* That is a real strategic alternative: make the output space trivially parseable rather than reward the model for parseability.

**Dr. GRPO's authors avoid a format reward specifically to avoid hacking**, and say so in a code comment. [`understand_r1_zero/math_grader.py#L1018-L1023`](https://github.com/sail-sg/understand-r1-zero/blob/main/understand_r1_zero/math_grader.py#L1018-L1023), verbatim (re-fetched and confirmed; the same comment appears three times in the file):

```python
    if is_correct:
        return {"formatted": True}, 1.0  # Correctness reward.
    else:
        return {
            "formatted": True
        }, 0.0  # Formatted but wrong answer; no format reward to avoid hacking.
```

**And Dr. GRPO supplies the specific Qwen2.5 caveat.** [arXiv:2503.20783v2, "Understanding R1-Zero-Like Training: A Critical Perspective"](https://arxiv.org/abs/2503.20783), Table 1 — average score across benchmarks by prompt template:

| Model | 4-shot | R1 template | Qwen template | **No template** |
| --- | --- | --- | --- | --- |
| Qwen2.5-Math-7B | 23.8 | 0.0 | 26.5 | **38.2** |
| Qwen2.5-Math-1.5B | 19.7 | 7.9 | 24.2 | **33.1** |

verbatim from the paper:

> "However, **Qwen2.5 models work best (with 100% answering rate) when no template is used.**"
>
> "we hypothesize that they might pretrain on the concatenated text to maximize log p_θ(q;o) directly. If our hypothesis turns out true, we shall be more careful about using Qwen2.5 models to reproduce DeepSeek-R1-Zero, since **the base models are already SFT-like without templates**."

Note the R1 template scoring **0.0** for Qwen2.5-Math-7B — that is a *format*-induced zero, not a capability-induced one, and it is the largest format effect anywhere in this document. It is also a warning in the opposite direction from the one we are worried about: for Qwen2.5, the risk is that **imposing an unfamiliar template destroys performance**, not that the model fails to format.

### 2.5 Format rewards do get gamed — with receipts

The gaming is documented, specific, and consistently about **satisfying the tag check while emptying it of content**.

**Logic-RL** ([arXiv:2502.14768](https://arxiv.org/abs/2502.14768) §2.2), verbatim:

> "Under our early imperfect rule design, we consistently observed reward hacking phenomena, some of which are listed below: • Skipping the `<think></think>` process and directly answering. • Placing reasoning inside the `<answer></answer>` tag. • Repeatedly guessing answers without proper reasoning. • Including irrelevant nonsense in addition to providing the answer. • **Organizing correct answer in a wrong manner for extraction.** • Revisiting the thinking phase after already outputting an `<answer>` due to insufficient reasoning. • **Repeating the original question or using phrases like 'thinking process here' to avoid true reasoning.**"

Its fix is only structural tag counting, and [issue #67](https://github.com/Unakar/Logic-RL/issues/67) asks how genuine reasoning is enforced without a maintainer answer. Logic-RL's own reward is unusually heavy on format — [`verl/utils/reward_score/kk.py#L171-L197`](https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py) gives `format_score = +1 if format_correct else −1`, added to an answer score of `+2 / −1.5 / −2`.

**Med-RLVR** ([arXiv:2502.19655](https://arxiv.org/abs/2502.19655) §4.2.1), verbatim:

> "Stage 4 (Direct Answer Hacker): The model learns to hack the reward by directly giving away the answer within the thinking step while drastically shortening the thinking length"
>
> "Stage 5 (Step-by-Step Exploit): The model learns to hack the reward through a different strategy: adding step-by-step reasoning before `<think>`, which leads to longer response length"

**open-r1** ([issue #359](https://github.com/huggingface/open-r1/issues/359)), verbatim:

> "In the training process, I found completion with multiple `<answer>` or `<think>` tags still can get a high format reward as long as the begin tag is `<think>` and the end tag is `</think>`."

**What I could NOT find: any documented case of a *code-fence* format reward being gamed.** Every gaming report above is about `<think>`/`<answer>` reasoning tags. That asymmetry is probably not luck — a code fence is a much thinner target, since the reward for emitting `` ```python\n\n``` `` with nothing inside is bounded by the fence weight and the execution reward still returns 0. But it is an absence of evidence, not evidence of absence.

### 2.6 Format adherence before vs. after RL: one number exists

I found exactly **one** primary source reporting format accuracy as a measured quantity across RL training. ["Surrogate Signals from Format and Length: Reinforcement Learning for Solving Mathematical Problems without Ground Truth Answers"](https://arxiv.org/html/2505.19439v2), Table 3, MATH500, columns *Answer Acc / Format Acc*, verbatim:

> "Qwen2.5-Math-7B 61.7 87.3 | GRPO(Correctness) 74.0 95.0 | GRPO(Format-Only) 70.1 96.3 | offline SFT 51.3 88.7 | online SFT 71.3 95.0"

So format accuracy went **87.3 → 95.0** under a *correctness-only* GRPO run of 100 steps (*"All experiments used a batch size of 128 and ran for 100 training steps"*). Two readings, both useful:

- A ~13% baseline format-failure rate on a math benchmark for a *math-specialised* Qwen2.5 model is not nothing. If that transfers to code fences, the concern in this project's premise is real.
- **Correctness-only RL fixed most of it by itself** (87.3 → 95.0), without a format reward. And the paper also states: *"Since both traditional RL with ground truth rewards and our format-based RL mainly learn answer formatting in the first 15 training steps…"* — i.e. format adherence is a *fast, early* thing that RL cleans up on its own, and the format-only run (96.3) barely beat the correctness-only run (95.0) on the format metric it was directly optimising.

Everything else is a plot with no stated numbers: Open-Reasoner-Zero's Figure 4 has a "Correct Format Ratio" panel but the text only says *"in high probability"*; SimpleRL-Zoo's Figure 6 plots accuracy and length, not format rate. **The field essentially does not publish format-adherence-over-training numbers.** See §5.

---

## 3. Constrained / structured decoding

### 3.1 TRL v1.9.2 *does* expose structured outputs during GRPO rollouts

This is newer than most write-ups suggest and I have not seen it discussed. [`trl/trainer/grpo_config.py#L133-L134`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L133-L134), verbatim:

```
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.
```

and the field, [`grpo_config.py#L609-L612`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L609-L612):

```python
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled."},
    )
```

It is wired into the colocate `SamplingParams` at [`trl/generation/vllm_generation.py#L613-L634`](https://github.com/huggingface/trl/blob/v1.9.2/trl/generation/vllm_generation.py#L613-L634), verbatim:

```python
            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0 if min_p is None else min_p,
                "max_tokens": max_completion_length,
                "logprobs": self.logprobs,
            }
            generation_kwargs.update(self.generation_kwargs)

            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)
```

Two escape hatches, then: the named `vllm_structured_outputs_regex`, and the generic `generation_kwargs["structured_outputs"]` dict, which reaches **every** field of vLLM's `StructuredOutputsParams` — including `grammar` (EBNF). The full dataclass, [`vllm/sampling_params.py#L71-L85`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/sampling_params.py#L71-L85), verbatim:

```python
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None
```

Server mode forwards it too ([`vllm_generation.py#L575-L576`](https://github.com/huggingface/trl/blob/v1.9.2/trl/generation/vllm_generation.py#L575-L576)).

**The non-vLLM paths have nothing.** `GRPOConfig.generation_kwargs` is documented as *"Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or `SamplingParams` (if using vLLM)"* ([`grpo_config.py#L102-L106`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L102-L106)), and in the HF-generate branch it is funnelled into `GenerationConfig(**generation_kwargs, disable_compile=True)` ([`grpo_trainer.py#L1109-L1126`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1109-L1126)). There is no `logits_processor`, no `grammar`, no `prefix_allowed_tokens_fn` hook anywhere in `grpo_trainer.py`. So **if you are on `use_transformers_continuous_batching=True` — which [`rlvr-stack.md` §1.9](./rlvr-stack.md) recommends for this single-GPU setup — constrained decoding is not available to you at all.**

### 3.2 Is it sound? The logprobs *are* the constrained ones — verified in vLLM's source

The theoretical concern is real: under a constraint, the behaviour policy is `q(a|s) = π_θ(a|s)·1[a ∈ A]/Z(s)`, not `π_θ`. If you collect logprobs from `q` and update `π_θ` as if they came from `π_θ`, the gradient is wrong.

**Which logprobs does vLLM actually return?** The grammar bitmask is applied to the logits **before** the sampler runs. [`vllm/v1/worker/gpu_model_runner.py#L4526-L4533`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_model_runner.py#L4526-L4533), verbatim:

```python
        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)
```

and inside the sampler, *both* logprob modes read from that same already-masked tensor — [`vllm/v1/sample/sampler.py#L84-L104`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/sample/sampler.py#L84-L104), verbatim:

```python
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            ...
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
```

**So even `logprobs_mode="raw_logprobs"` does not give you the unconstrained distribution.** "Raw" in vLLM means *before penalties/temperature/top-k*, per [`vllm/config/model.py#L229-L238`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/model.py#L229-L238), verbatim:

```python
    logprobs_mode: LogprobsMode = "raw_logprobs"
    """Indicates the content returned in the logprobs and prompt_logprobs.
    Supported mode:
    1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    Raw means the values before applying any logit processors, like bad words.
    Processed means the values after applying all processors, including
    temperature and top_k/top_p.
    Note: for prompt_logprobs, processed_* and raw_* yield identical results
    because prompt tokens do not go through sampling processors.
    """
```

— but the grammar bitmask is not a "logit processor" in that taxonomy; it is applied upstream, in the model runner. There is no vLLM setting that recovers the unconstrained logprobs alongside constrained sampling.

TRL hard-codes `logprobs_mode="processed_logprobs"` when constructing the colocate engine ([`vllm_generation.py#L365`](https://github.com/huggingface/trl/blob/v1.9.2/trl/generation/vllm_generation.py#L365)), so what lands in `sampling_per_token_logps` is `log q(a|s)` — the true behaviour-policy logprob including the constraint, the temperature, and top-p.

**What TRL does with it.** [`grpo_trainer.py#L2578-L2601`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2578-L2601), verbatim:

```python
            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                ...
                vllm_importance_sampling_ratio = torch.exp(logps_diff)
```

`old_per_token_logps` is recomputed by the **training** model, unconstrained. So the ratio TRL forms is exactly `π_θ(a|s) / q(a|s)` — which is the textbook importance weight for correcting a constrained behaviour policy. And it is **on by default** (`vllm_importance_sampling_correction: bool = field(default=True, ...)`, [`grpo_config.py#L915-L923`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L915-L923)).

**My analysis — stated as reasoning, not as a cited finding, because I found no primary source that works this through:**

1. TRL's existing vLLM↔training IS correction *does* structurally cover constrained sampling, because it compares against the actual sampling distribution rather than an assumed one. That is better than I expected going in.
2. But the correction is **clipped or masked**, not unbiased. The default is `vllm_importance_sampling_mode="sequence_mask"` with `vllm_importance_sampling_clip_max=3.0` ([`grpo_config.py#L924-L935`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L924-L935)). Under a constraint, `q ≥ π_θ` on every allowed token, so every per-token ratio is `≤ 1` and the **sequence-level product is `≤ 1` and shrinks geometrically with the number of constrained steps**. For a fence-shaped regex over a 2000-token completion, most tokens are unconstrained (the grammar allows anything inside the block), so the product should stay near 1 — but for a *tight* grammar it will collapse, and with `sequence_mask` a collapsed ratio below `C_min` is zeroed, silently discarding the whole rollout. **Constrained decoding + TRL's default IS mode is a plausible mechanism for silently losing training signal.** I did not run this.
3. Worse, and not fixed by IS at all: **GRPO's advantage baseline is computed from the group, and the group is drawn from `q`.** `mean_grouped_rewards` is an expectation under the constrained policy. TRL applies the IS ratio to the per-token surrogate loss, not to the advantage normalisation ([`grpo_trainer.py#L2684-L2708`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2684-L2708) computes advantages before any IS weighting). So the baseline retains a bias that no amount of token-level IS removes.
4. And there is a specific, nasty interaction with the actual goal here: **a grammar that forbids the failure mode also destroys the reward variance that the failure mode was supplying.** If no rollout can fail to produce a fence, then "no code extracted" never fires, and you are back to the all-wrong degenerate group of §2.3 — you have removed a symptom and kept the disease.

### 3.3 State of the libraries (checked 2026-08-01 via PyPI JSON)

| Package | Latest | Released | Status |
| --- | --- | --- | --- |
| [`xgrammar`](https://github.com/mlc-ai/xgrammar) (mlc-ai) | **0.2.5** | 2026-07-22 | most active; vLLM's primary backend |
| [`outlines`](https://github.com/dottxt-ai/outlines) (dottxt-ai) | **1.3.2** | 2026-07-20 | active |
| [`guidance`](https://github.com/guidance-ai/guidance) | **0.3.1** | 2026-02-03 | slow |
| [`lm-format-enforcer`](https://github.com/noamgat/lm-format-enforcer) | **0.11.3** | **2025-08-24** | **stale — ~11 months since a release** |

vLLM v0.26.0 supports all four as backends. [`vllm/config/structured_outputs.py#L12-L26`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/structured_outputs.py#L12-L26), verbatim:

```python
StructuredOutputsBackend = Literal[
    "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
]


@config
class StructuredOutputsConfig:
    """Dataclass which contains structured outputs config for the engine."""

    backend: StructuredOutputsBackend = "auto"
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
```

Note the naming churn: the old `guided_*` API was **removed**, not deprecated. [`docs/features/structured_outputs.md`](https://github.com/vllm-project/vllm/blob/v0.26.0/docs/features/structured_outputs.md), verbatim:

> "If you are still using the following deprecated API fields which were **removed in v0.12.0**, please update your code to use `structured_outputs` as demonstrated in the rest of this document:
> - `guided_json` -> `{"structured_outputs": {"json": ...}}` or `StructuredOutputsParams(json=...)`
> - `guided_regex` -> `{"structured_outputs": {"regex": ...}}` or `StructuredOutputsParams(regex=...)`
> …
> - `guided_decoding_backend` -> Remove this field from your request"

So any tutorial mentioning `guided_decoding_backend` predates vLLM v0.12.0 and will not run.

### 3.4 Does any RLVR framework actually use it for rollouts? No.

| Framework | Exposes a structured-output knob for rollouts? | Documented recipe using it? |
| --- | --- | --- |
| **TRL** | **yes** (`vllm_structured_outputs_regex`, §3.1) | **no** — zero hits across `docs/` and `examples/`; it appears only in the config fields, `vllm_generation.py`, `vllm_serve.py`, and one test |
| verl | no | — |
| OpenRLHF | no | — |
| SkyRL | no (only an `xgrammar` pin in a sub-package's dependency overrides) | — |
| NeMo-RL | no — its `SamplingParams` construction is a closed set with no structured-output field | — |
| prime-rl | **explicitly disables it** (below) | — |

I corroborated the verl result directly: [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/workers/rollout/vllm_rollout/vllm_rollout.py) and [`verl/trainer/config/rollout/rollout.yaml`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/config/rollout/rollout.yaml) at v0.8.0 contain **zero** occurrences of `guided`, `structured`, `grammar`, or `xgrammar`.

**And prime-rl does not merely omit it — it turns it off on purpose, and says exactly why.** [`src/prime_rl/inference/server.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/inference/server.py), verbatim (I re-fetched and confirmed this myself):

```python
    # vLLM 0.24.0 flipped VLLM_ENFORCE_STRICT_TOOL_CALLING's default to True, which
    # grammar-constrains generation (xgrammar structural tags) for tool_choice
    # "required"/named and strict tools — a sampling distribution the trainer never
    # sees. Keep it off so rollout logprobs stay faithful for importance ratios.
    os.environ.setdefault("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
```

**That is the single most decision-relevant sentence in this entire document.** An RL framework that ships production runs identifies grammar-constrained decoding as *"a sampling distribution the trainer never sees"* and disables it by default so that *"rollout logprobs stay faithful for importance ratios."*

Meanwhile **nobody ever asked TRL about it.** The PR that first added the capability, [huggingface/trl#2811](https://github.com/huggingface/trl/pull/2811), *"Adds the ability to pass vLLM's `GuidedDecodingParams` through to the `llm.generate` call"* (merged 2025-02-18), had the contributor explicitly invite scrutiny — *"Let me know if there's any problem or flaw in my logic with this."* — and the discussion that followed was about JSON-serializability. Across that PR and the follow-ups, no maintainer raises logprob mismatch, off-policy bias, or importance sampling.

### 3.5 Who *has* addressed the bias

The distortion itself is well established in the inference literature:

- **Grammar-Aligned Decoding**, [arXiv:2405.21047](https://arxiv.org/abs/2405.21047) (2024-05-31): *"GCD techniques (and in general constrained decoding techniques) can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM."*
- **DISC**, [arXiv:2504.09135](https://arxiv.org/abs/2504.09135) (2025-04-12): *"it introduces unintended biases into the output distribution. This paper introduces Dynamic Importance Sampling for Constrained Decoding (DISC)… which leverages dynamic importance sampling to achieve theoretically guaranteed asymptotic unbiasedness."* Inference only; no RL.
- **Let Me Speak Freely?**, [arXiv:2408.02442](https://arxiv.org/abs/2408.02442) (2024-08-05): *"we observe a significant decline in LLMs reasoning abilities under format restrictions."* — a separate and independently damning result for a reasoning-heavy task like CodeContests.

The RL-side treatment is essentially one paper. **Ctrl-R, "Learning Structured Reasoning via Tractable Trajectory Control"**, [arXiv:2603.01641](https://arxiv.org/abs/2603.01641) (submitted 2026-03-02), verbatim:

> "Guided rollouts are sampled from an augmented behavior policy μα, which differs from the proximal policy assumed in standard PPO. Ctrl-R therefore adopts a decoupled formulation (Hilton et al., 2022), separating the proximal policy that controls policy updates from the behavior policy used for off-policy correction."

with `μα(xt|x<t) = (1/Zt) πθold(xt|x<t) · γ(α|x<t,xt)`. Note what it does **not** do: γ is a *soft*, tractable guide, chosen precisely so the partition function `Zt` and the resulting IS ratio stay computable. A hard grammar mask is the case where that machinery is hardest.

**And vLLM is building the fix for the analogous problem — but only for top-p/top-k.** [vllm#49577, "[Feature] Mask Replay"](https://github.com/vllm-project/vllm/pull/49577) (opened 2026-07-23, open), verbatim:

> "With top-p sampling, rollout probabilities are normalized over a truncated vocabulary, while training commonly recomputes probabilities over the full vocabulary. This creates a systematic mismatch in importance ratios and KL estimates."

That is *exactly* the grammar-mask problem, one mask weaker. There is no grammar-mask equivalent proposed, and the ecosystem-wide logprob-semantics RFC ([vllm#42259](https://github.com/vllm-project/vllm/issues/42259), open since 2026-05-11) does not mention structured outputs at all. The interaction is also **completely undocumented**: `docs/features/structured_outputs.md` contains zero occurrences of "logprob", as does `vllm/v1/structured_output/`.

**Verdict on the silence: it is principled at the infrastructure end and incidental at the application end.** prime-rl names the failure mode and disables it; Ctrl-R restructures a whole method around it; verl, OpenRLHF, SkyRL and NeMo-RL never added the knob. But TRL added it without discussion, the one direct correctness question went unanswered, and the papers that do use grammar-masked GRPO rollouts say nothing. **Do not read TRL's `vllm_structured_outputs_regex` as an endorsement.**

---

## 4. Assistant prefill / forced prefix

### 4.1 The Qwen2.5 chat template, verbatim, and what it does *not* contain

Fetched from [`Qwen/Qwen2.5-3B-Instruct` `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/tokenizer_config.json). The `Qwen2.5-7B-Instruct` file is **byte-identical** (`md5 898cd7942ca145c7828f36af58f62ddf` for both); `Qwen2.5-Coder-7B-Instruct` differs elsewhere in the file but its `chat_template` string is character-identical to both. The relevant fragments, verbatim:

```jinja
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
...
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

**There is no `continue_final_message` branch, and no `loop.last` special case for an assistant message.** A trailing assistant turn is unconditionally closed with `<|im_end|>\n`. If you naively render `[{"role":"user",...},{"role":"assistant","content":"```python\n"}]` you get a *finished* assistant turn and the model starts a new one.

### 4.2 …but `continue_final_message=True` works anyway, because transformers does it template-agnostically

`transformers` does not require template cooperation. [`src/transformers/utils/chat_template_utils.py#L539-L569`](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/utils/chat_template_utils.py#L539-L569), verbatim:

```python
    continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "
    for chat in conversations:
        ...
        if continue_final_message:
            chat = deepcopy(chat)
            continue_final_message = continue_final_message if isinstance(continue_final_message, str) else "content"

            if (final_message := chat[-1].get(continue_final_message)) is None:
                raise ValueError(...)
            if continue_final_message not in chat_template:
                raise ValueError(...)
            ...
            else:
                chat[-1][continue_final_message] = chat[-1][continue_final_message] + continue_final_message_tag
```

and the truncation, [L588-L605](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/utils/chat_template_utils.py#L588-L605):

```python
        if continue_final_message:
            if (final_message.strip() not in rendered_chat) or (
                continue_final_message_tag.strip() not in rendered_chat
            ):
                raise ValueError(
                    "continue_final_message is set but the final message does not appear in the chat after "
                    "applying the chat template! This can happen if the chat template deletes portions of "
                    "the final message. ..."
                )
            tag_loc = rendered_chat.rindex(continue_final_message_tag.strip())
            if rendered_chat[tag_loc : tag_loc + len(continue_final_message_tag)] == continue_final_message_tag:
                # The template preserves spacing, so things are simple
                rendered_chat = rendered_chat[:tag_loc]
            else:
                # The message has trailing spacing that was trimmed, so we must be more cautious
                rendered_chat = rendered_chat[:tag_loc].rstrip()
```

It appends a sentinel to the final message, renders normally, then **truncates the rendered string at the sentinel**. Everything after — including Qwen's `<|im_end|>\n` — is discarded. So `apply_chat_template(..., continue_final_message=True)` on Qwen2.5 produces exactly:

```
<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n```python\n
```

⚠️ **One real trap.** Look at the `else` branch: if the template trims trailing whitespace, transformers falls back to `rstrip()`. A prefill of `` "```python\n" `` ends in a newline. Qwen's template does *not* trim (`+ message.content +` is verbatim), so the fast path fires and the trailing `\n` is preserved — I verified this by reading the template, not by running it. But if you ever switch to a template that strips, your prefill silently becomes `` "```python" `` with no newline, and open-r1-style extractors that require `` ```python\n `` will then fail on **every** rollout.

The API contract, [`tokenization_utils_base.py#L3033-L3038`](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/tokenization_utils_base.py#L3033-L3038), verbatim:

> `continue_final_message (bool or str, *optional*)`: If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to "prefill" part of the model's response for it. If a string is passed, it will be used as the key for the field to continue (e.g. "reasoning_content"). **Cannot be used at the same time as `add_generation_prompt`.**

and the enforcement, [L3099-L3105](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/tokenization_utils_base.py#L3099-L3105):

```python
        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")
```

vLLM's OpenAI server supports the same flag with the same constraint — [`vllm/entrypoints/openai/chat_completion/protocol.py#L302-L311`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/chat_completion/protocol.py#L302-L311) and the validator at [L918-L922](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/chat_completion/protocol.py#L918-L922).

### 4.3 TRL cannot do it through `chat_template_kwargs` — but it can through a string prompt

TRL v1.9.2 hard-codes `add_generation_prompt=True` when the prompt is conversational. [`trl/trainer/grpo_trainer.py#L1758-L1767`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1758-L1767), verbatim:

```python
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **({"padding": True} if needs_padding_workaround else {}),
                **self.chat_template_kwargs,
            )
```

`GRPOConfig.chat_template_kwargs` exists ([`grpo_config.py#L556-L562`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L556-L562)) but cannot help here: passing `add_generation_prompt=False` raises `TypeError: got multiple values for keyword argument`, and passing `continue_final_message=True` raises the transformers `ValueError` quoted above, because `add_generation_prompt=True` is already there.

**The supported route is the non-conversational branch.** [`grpo_trainer.py#L1778-L1782`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1778-L1782), verbatim:

```python
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            images = None
            multimodal_fields = {}
        return prompt_ids, images, multimodal_fields
```

If your dataset's `prompt` column is a **plain string** rather than a list of messages, TRL tokenizes it verbatim and does nothing else to it. TRL's own docs call this the *standard* format ([`docs/source/dataset_formats.md`](https://github.com/huggingface/trl/blob/v1.9.2/docs/source/dataset_formats.md#prompt-only)):

> ```python
> # Standard format
> prompt_only_example = {"prompt": "The sky is"}
> # Conversational format
> prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
> ```

So: render the chat template yourself (with `continue_final_message=True`), store the resulting string, and TRL will treat the prefill as prompt.

### 4.4 What actually breaks — and what does not

**Does the prefill get trained on?** **No, if you put it in the prompt.** TRL builds `completion_mask` from the generated ids only ([`grpo_trainer.py#L2388`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2388): `completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]`), and the loss mask is `loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask` ([L2426](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2426)). Prompt tokens are concatenated for the forward pass only ([L2431](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2431)). So a prefilled `` ```python\n `` contributes context and zero gradient. That is the *correct* behaviour: you are not teaching the model to emit the fence, you are removing the opportunity to not emit it.

**Does the prefill reach the reward function?** **No.** For a string prompt, [`grpo_trainer.py#L2188-L2189`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2188-L2189):

```python
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
```

Your reward function receives only what the model generated — which will start mid-code-block and contain a *closing* fence but no *opening* one. **Your extractor must re-prepend the prefill** (or run in "assume the completion opens inside a block" mode). This is the single most likely way to get this wrong and silently score 0 on everything.

**Does it work with vLLM?** Yes. TRL passes token ids, not text, to vLLM: `vllm_prompts = [{"prompt_token_ids": ids} for ids in all_prompts]` ([`trl/generation/vllm_generation.py#L667`](https://github.com/huggingface/trl/blob/v1.9.2/trl/generation/vllm_generation.py#L667)). vLLM never re-applies a chat template, so the prefill survives intact.

**Other costs, stated honestly:**

- It **forecloses reasoning before the code**. If you prefill `` ```python\n `` immediately after `<|im_start|>assistant\n`, the model cannot think first. For CodeContests that is a real loss — every implementation in §6 prompts for reasoning. The mitigation is to prefill a *reasoning opener* instead, or to prefill only the fence and accept the model will reason *inside* comments, or to use a two-part prompt whose final instruction makes an immediate code block natural.
- It **shifts the distribution off-policy in a benign direction**: the prompt is different, so you are optimising a different (conditioned) policy. Unlike constrained decoding (§3), there is **no logprob inconsistency** — the model's distribution over the generated tokens is its true conditional distribution given the prefixed context. Prefill is a *prompt* intervention, not a *sampling* intervention. That is why it is safe and constrained decoding is not.
- If you later deploy without the prefill, you are evaluating a different conditioning than you trained. Keep the prefill in eval too, or measure both.

**Who does this in an RL loop?** I found **no** open RLVR implementation that prefills a code fence during GRPO rollouts. bigcode-evaluation-harness does it for *evaluation* (§1.7, `if "```python" in assistant_token`), and the prefill idiom is documented by inference providers, but rLLM, open-r1, verl, and code-r1 all use `add_generation_prompt`-style prompts and take their chances. The nearest RL-side precedent is prefix-conditioned rollout work such as [arXiv:2607.07674, "Max Out GRPO Signal: Adaptive Trace Prefix Control for Hard Reasoning Problems"](https://arxiv.org/abs/2607.07674) (Vladislav Beliaev, submitted 2026-07-08), which prepends correct-solution prefixes to control difficulty and states that *"The method is implemented in data preparation plus a loss mask on prefix tokens; the trainer is otherwise stock."* — same mechanism (prefix in data, mask in loss), different purpose. **Treat "prefill for format" as an unattested-but-sound technique, not standard practice.**

---

## 5. Empirical format adherence — is this a real problem or a phantom?

**Short answer: nobody publishes a parse-failure rate for code generation. Not Qwen, not any code eval harness, not any leaderboard.** Every code harness folds extraction failure into pass@1 by construction — including Qwen's own. This is a genuine hole in the literature, and it means we have to measure it ourselves.

### 5.1 The Qwen2.5 technical reports say nothing

I grepped the full rendered HTML of both papers myself. Term counts:

| Term | Qwen2.5 TR ([arXiv:2412.15115v1](https://arxiv.org/html/2412.15115v1)) | Qwen2.5-Coder TR ([arXiv:2409.12186v3](https://arxiv.org/html/2409.12186v3)) |
| --- | --- | --- |
| `parse failure` / `parsing failure` / `extraction failure` / `format failure` / `failure rate` | 0 | 0 |
| `unparseable`, `malformed`, `sanitiz`, `post-process`/`postprocess` | 0 | 0 |
| `markdown` | 0 | 0 |
| `code block` | 0 | 1 (about FIM *training* tokens, not eval) |

(`arxiv.org/html/2412.15115v2` returns 404; only v1 renders.)

**Neither report states how code is extracted from instruct-model output for any benchmark, nor any decoding parameter, nor any prompt template for code evals.** The most either says is the benchmark list — [Qwen2.5 TR §5.2.1](https://arxiv.org/html/2412.15115v1), verbatim: *"we evaluate on … HumanEval, MBPP, MultiPL-E and LiveCodeBench 2305-2409 (Jain et al., 2024) for coding"*.

For scale, the headroom that is *not* decomposed — Qwen2.5 TR Table 8 (7B+ instruction-tuned) and Table 9 (2B–4B):

| | HumanEval | MBPP | MultiPL-E | LiveCodeBench |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 84.8 | 79.2 | 70.4 | **28.7** |
| Qwen2.5-3B-Instruct | 74.4 | 72.7 | 60.2 | **19.9** |

On LiveCodeBench — the closest published proxy for CodeContests — Qwen2.5-3B-Instruct leaves ~80 points of "not pass@1" and **not one point of it is attributed to formatting versus wrongness**.

The HF model cards ([`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)) say nothing about code output format either; the only format-adjacent claim is the generic *"generating structured outputs especially JSON"*.

### 5.2 Qwen's own harness launders parse failure into wrong answers — verbatim

This is the sharpest single piece of evidence, and I fetched it myself. [`QwenLM/Qwen2.5-Coder`, `qwencoder-eval/instruct/multipl_e/chat/evaluate.py#L113-L125`](https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/instruct/multipl_e/chat/evaluate.py), verbatim:

```python
def extract_func(text, job, language):
    if language == "py" or language == "python":  # python
        def extract_python_code(text) -> str:
            code_block_pattern = re.compile(rf"```.*?\n(.*?)```", re.DOTALL)
            code_block = code_block_pattern.search(text)
            if code_block is not None:
                return code_block.group(1)
            else:
                return text
```

Three things at once:

1. `.search()` — this takes the **FIRST** block, unlike five of the seven implementations in §1.8. Qwen's published HumanEval numbers therefore rest on a first-block policy.
2. On failure it returns **the whole response**, which then fails to compile. Same bug as verl's `prime_code` (§1.3).
3. **There is no counter.** Nothing anywhere records how often the `else` branch fired.

The prompt behind the published 88.4 HumanEval figure is at [L453](https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/instruct/multipl_e/chat/evaluate.py) of the same file, verbatim:

> "Please continue to complete the function and return all completed code in a codeblock. Here is the given code to do completion:"

And the McEval path is blunter still — [`qwencoder-eval/instruct/McEval/eval/eval_all.py#L39-L45`](https://github.com/QwenLM/Qwen2.5-Coder/blob/main/qwencoder-eval/instruct/McEval/eval/eval_all.py), verbatim:

```python
        try:
            code = extract(item["raw_generation"][0], item, lang)
        except:
            print(f'+++++ Extract {item["task_id"]} failed')
            code = "1234"  #avoid code file is empty    
        if code is None:
            code = "1234"
```

**Extraction failure becomes the literal string `"1234"`**, which is executed, fails, and is scored as a wrong answer. It is printed to stdout and never aggregated. Separately, the `QwenLM/Qwen2.5` repo ships no code-generation eval at all, so the extraction procedure behind the Qwen2.5 report's HumanEval and LiveCodeBench numbers is simply **unpublished**.

### 5.3 No harness or leaderboard has a parse-failure bucket

- **LiveCodeBench** — an empty extraction from [`extract_code`](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/utils/extraction_utils.py#L15) (`return ""`) is handed to `run_test` like any other string, fails, and lands in pass@1's denominator. No counter in `compute_code_generation_metrics.py`.
- **BigCodeBench** — the complete status vocabulary in [`bigcodebench/eval/__init__.py`](https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/eval/__init__.py) is `PASS`/`FAIL`/`TIMEOUT` with codes `_SUCCESS`/`_FAILED`/`_TIMEOUT`/`_UNKNOWN`. No syntax or extraction status exists.
- **EvalPlus** — [`docs/cli.md`, "Code post-processing"](https://github.com/evalplus/evalplus/blob/master/docs/cli.md), verbatim: *"LLM-generated text may not be compilable code for including natural language lines or incomplete extra code. We provide a tool namely `evalplus.sanitize` to clean up the code"*, and `evalplus.syncheck` *"will print erroneous code snippets and why they are wrong"* — it **prints**, it does not aggregate a rate.
- **OpenAI simple-evals** — `common.py`'s `ANSWER_PATTERN` is a bare regex and aggregation emits `score` only.

### 5.4 The eval papers name the problem and then design around it

**LiveCodeBench** ([arXiv:2403.07974](https://arxiv.org/abs/2403.07974), March 2024), verbatim:

> "A one-shot prompt is used for all models to avoid any formatting and answer extraction issues."

> "For the base models, we only used in the code generation scenario since they do not easily follow the format for the other scenarios."

So the benchmark's own authors treat format failure as a real enough hazard to change their prompting strategy for it — and then report no rate.

**EvalPlus** ([arXiv:2305.01210](https://arxiv.org/abs/2305.01210)) §4, verbatim:

> "For conversational models (i.e., ChatGPT and GPT-4), we obtain the code fragments by parsing the code blocks (i.e., within "```") in the output. We found ChatGPT tends to repeat problem description with detailed explanation, which can consume more than 512 new tokens to complete a solution for around 11% of problems."

That 11% is a *truncation* rate, not an extraction-failure rate — but it is the same failure surface as our case F, and it is the only percentage either eval paper offers.

### 5.5 The three nearest measurements — all in math, and one in code

Nobody measures this for competitive-programming code. The closest primary sources, in descending relevance:

**(a) Spurious Rewards** ([arXiv:2506.10947](https://arxiv.org/abs/2506.10947), Shao et al., submitted 2025-06-12, revised 2026-02-25), Table 5 — verified verbatim from the PDF:

```
                       Default   MathProblem   SimpleRL-Zoo   Sober   Spurious
    MATH Acc.            49.4        55.8          63.2       61.60     68.8
    % Parsable           78.9        72.1          85.4       93.1      84.1
```

> "Table 5. Accuracy on MATH-500 and percent of parsable (format-following) responses on Qwen2.5-Math-7B with various prompts from Table 4. **Even with a spurious prompt, the model is able to follow the format 84.1% of the time.** It is not obvious that much of the performance can be explained by format-following."

and:

> "As shown in Table 5, Sober prompt brings the highest parsable rate of the answers, while our spurious prompt leads to the highest accuracy on MATH-500. The results indicate that **the model is very sensitive to prompts and the best performance does not necessarily require the highest parsable rate** nor task-relevant information in context."

**This is the most useful number in this document.** A Qwen2.5 model, before RL, parses at **72–93% depending only on the prompt** — a 21-point swing from prompt wording alone. So (i) format failure at the 7–28% level is *real*, not a phantom; (ii) **prompt design moves it more than anything else**; and (iii) the highest parse rate did not give the best accuracy.

**(b) One-shot RLVR** ([arXiv:2504.20571](https://arxiv.org/abs/2504.20571), submitted 2025-04-29), Appendix C.2.3 *"(Only) Format Correction?"* — verified verbatim from the v3 HTML:

> "Table 14: RLVR with only format reward can still improve model performance significantly, while still having a gap compared with that using outcome reward. Numbers with orange color denote the ratio of responses that contain "`\boxed{}`" in evaluation."

Its format-only reward is defined exactly as an extractability reward — verbatim:

> "if the verifier can parse the final answer from model output, then it gets 1 reward no matter if the answer is correct or not, otherwise it gets 0 reward"

and the finding, verbatim:

> "(2) π₁ with outcome reward or format reward have similar `\boxed{}` ratios, but the former still has better test performance (e.g., +7.4% on MATH500…)"

i.e. **outcome-only RLVR fixes formatting just as well as a dedicated format reward does, and gets better accuracy.** That is a direct argument against relying on a format reward to teach format. (I could not machine-extract the individual coloured cell values from Table 14 and have not quoted them.)

**(c) CodeScaler** ([arXiv:2602.17684](https://arxiv.org/abs/2602.17684), 2026-02-04) — the only code-domain source, and its methodology section is the closest thing to an endorsement of §8.2's cascade. Verbatim:

> "**In RLVR, code extraction is straightforward as the execution environment naturally filters flawed or syntactically incorrect code.** However, RM-based training lacks this inherent verification, making it sensitive to the quality of the input. RM may risk in assigning unpredictably high scores to seemingly plausible but syntactically broken code. To improve the reliability of RM evaluations, we implement a strict code extraction pipeline to extract codes 𝒄 from model responses 𝒓: 1. **We extract code only from a single, well-defined code block. Responses containing multiple fragmented code blocks are discarded.** Such cases occur primarily during the early stages of RL training, when the policy occasionally attempts to produce partial or incomplete code segments. 2. **We perform a static Abstract Syntax Tree (AST) check on extracted code.** This ensures that RM only evaluates code that is at least syntactically correct. Codes 𝒄 that fail any of the above criteria are replaced with the empty string ϵ."

Two things to take from this. First, the AST check as an extraction gate is independently arrived at by a code-RL paper, which is exactly §8.2's design. Second — and note the tension with the premise of this whole document — **CodeScaler explicitly says extraction is *not* a hard problem in RLVR**, because execution filters bad code anyway; their strictness is needed only because they replaced execution with a learned reward model. It also confirms that fragmented/partial code blocks *do* occur, and that they are concentrated *"during the early stages of RL training"*.

> ⚠️ A parallel search relayed specific CodeScaler "fragment rate" and "invalid rate" percentages to me. **I searched the v1 HTML and could not find them.** I have therefore not quoted them, and you should not rely on them.

### 5.6 What this means for us

- The measurement we want **does not exist**. Nobody reports parse-failure separately from wrong-answer for code generation, and Qwen's own harness is architecturally incapable of reporting it.
- The nearest evidence says format failure is **real but modest and prompt-dominated**: 7–28% non-parsable on math for a Qwen2.5 model, with a 21-point swing from prompt wording alone (§5.5a).
- It also says RL **fixes it on its own**: outcome-only RLVR reached the same `\boxed{}` ratio as a dedicated format reward (§5.5b), and correctness-only GRPO moved format accuracy 87.3 → 95.0 in 100 steps (§2.6).
- So the honest expected shape for this project is: a **non-zero initial format-failure rate that decays quickly under training**, whose main lever is the prompt, not the reward. The residual risk is concentrated in the **first few hundred steps** — exactly where a degenerate-group problem hurts most.
- **Therefore: measure it, on step 0, before doing anything else.** §8.3 shows how to get the number for free via TRL's `log_metric`.

---

## 6. Prompt design — the actual strings

### 6.1 open-r1

System prompt, identical across both code GRPO recipes — [`recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml) and [`recipes/Qwen2.5-Coder-7B-Instruct/grpo/config_codeforces.yaml`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/recipes/Qwen2.5-Coder-7B-Instruct/grpo/config_codeforces.yaml), verbatim:

```yaml
system_prompt: "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
```

Note this says nothing about code blocks — the fence requirement lives in the *dataset's* prompt column. The actual user prompt from [`open-r1/codeforces`, config `verifiable-prompts`](https://huggingface.co/datasets/open-r1/codeforces/viewer/verifiable-prompts), verbatim from row 0 of the train split (retrieved 2026-08-01 via the datasets-server `/rows` endpoint):

````text
You are an expert competitive programmer. You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution (that must respect memory and time limits), then provide a complete implementation in c++17.

Your solution must read input from standard input (cin), write output to standard output (cout).
Do not include any debug prints or additional output.

Put your final solution within a single code block:
```cpp
<your code here>
```

Execution time limit: 1.0 seconds
Memory limit: 256.0 MB

# Problem
…

Now solve the problem and return the code.
````

Three format devices in one prompt: an explicit instruction (*"Put your final solution within a single code block"*), an **inline one-shot demonstration of the fence** including the language tag, and a terminal reminder (*"Now solve the problem and return the code."*).

The reward weights, from the same two recipes, verbatim:

```yaml
reward_funcs:
- code
- format
reward_weights:
- 1.0
- 0.1
```

```yaml
reward_funcs:
- cf_code
- code_format
reward_weights:
- 1.0
- 0.1
```

### 6.2 rLLM / DeepCoder

DeepCoder's data pipeline routes LiveCodeBench-style problems through [`rllm/data/utils.py#L66-L76`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/data/utils.py#L66-L76), verbatim:

```python
def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt
```

with the constants at [`rllm/system_prompts.py#L272-L276`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/system_prompts.py#L272-L276), verbatim:

```python
LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
```

The `` ```python\n# YOUR CODE HERE\n``` `` block is a **one-shot format example embedded in the prompt** — the closest thing in these codebases to a prefill without actually prefilling. And `"### Answer: (use the provided format with backticks)"` is a terminal format reminder immediately before the generation point. This is the strongest prompt-side format scaffolding of any implementation surveyed, and it comes straight from LiveCodeBench's own harness (rLLM cites the file in a comment).

rLLM also ships a much longer Codeforces persona prompt, [`system_prompts.py#L279-L318`](https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/system_prompts.py#L279-L318), marked in the source with `# no validation yet` — it contains a "Prohibited Actions" list but **no code-fence instruction at all**. Do not copy that one.

### 6.3 code-r1 — and this one is literally CodeContests

`ganler/code-r1` builds its training set from `deepmind/code_contests` directly. [`examples/data_preprocess/coder1.py#L208-L210`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L208-L210):

```python
def codecontests():
    rich.print(Rule("Loading deepmind/code_contests..."))
    dataset = load_dataset("deepmind/code_contests")
```

The system prompt, [`coder1.py#L51-L54`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L51-L54), verbatim:

```python
SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""
```

and the CodeContests user prompt, [`coder1.py#L233-L236`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L233-L236), verbatim:

```python
            prompt = ("Solve the programming task below in a Python markdown code block. "
                      "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
                      "produces outputs through STDOUT (like those in the 'Output' section)."
                      f"\n\n{example['description'].strip()}")
```

Its function-call-style variant, [`coder1.py#L308`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L308):

```python
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['meta']['query'].strip()}"
```

Explicit instruction, no few-shot, no inline fence example. Combined with the `<think>/<answer>` structural requirement and the −1.1 penalty (§1.4), the format burden here is carried by the **reward**, not by the prompt.

### 6.4 verl

**verl ships no code prompt template.** Its [`examples/data_preprocess/`](https://github.com/verl-project/verl/tree/v0.8.0/examples/data_preprocess) directory contains `gsm8k.py`, `math_dataset.py`, `geo3k.py`, `hellaswag.py`, `full_hh_rlhf.py`, `multiturn.py`, `pokemon.py` and search/tool variants — **no APPS, no CodeContests, no LiveCodeBench**. verl's `prime_code` scorer expects `` ```python `` to appear and its README ([`verl/utils/reward_score/prime_code/README.md`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/prime_code/README.md)) only credits LiveCodeBench for the evaluation methodology. The prompt is entirely the user's problem.

### 6.5 Summary of prompt-design practice

| | Reasoning tags required | Explicit "code block" instruction | Inline fence example | Terminal reminder | Few-shot |
| --- | --- | --- | --- | --- | --- |
| open-r1 (codeforces) | `<think>/<answer>` (system) | yes | **yes** (` ```cpp\n<your code here>\n``` `) | yes | no |
| rLLM / DeepCoder (LCB) | no | yes ("enclose your code within delimiters") | **yes** (` ```python\n# YOUR CODE HERE\n``` `) | **yes** ("use the provided format with backticks") | no |
| code-r1 (CodeContests) | `<think>/<answer>` (system) | yes ("in a Python markdown code block") | no | no | no |
| verl | — | — | — | — | — |

**Nobody uses few-shot examples.** Everybody uses an explicit format instruction. The two implementations that also embed an inline fence demonstration (open-r1, rLLM) are the two whose extractors are strictest about the language tag — which is not a coincidence.

---

## 7. Bench: what these extractors actually do on malformed output

None of the sources publish a differential test, so I wrote one. Each extractor is transcribed **verbatim from the source quoted in §1** and run against sixteen hand-written completions covering the failure modes those sources' comments and issues actually mention. This is my own measurement, not a citation; the code is reproducible from the snippets in §1.

Cell meaning: `ok` = returned the intended program; `WRONG` = returned something else; `none` = returned `None`/`""`; `empty` = returned whitespace. For rows where implementations legitimately disagree about the intended answer, the extracted value is shown in parentheses.

```
case                      OPENR1    RLLM      CODER1    PRIME     LCB
A canonical               ok        ok        ok        ok        ok
B no lang tag             none      ok        ok        WRONG     ok
C capital ```Python       none      ok        ok        WRONG     ok
D ```py alias             none      ok        ok        WRONG     ok
E space after tag         none      none      none      ok        ok
F truncated, no close     none      none      none      ok        none
G naive block then final  ok        ok        WRONG     ok        ok
H imports + solution      (print)   (print)   (import)  (print)   (print)
I bare code, no fence     none      none      none      (print)   none
J prose only              none      none      none      (I cann)  none
K ``` inside string lit   WRONG     WRONG     ok        WRONG     WRONG
L odd fence count         ok        ok        ok        ok        empty
M closing fence indented  ok        ok        none      ok        ok
N ```python3 tag          none      ok        ok        WRONG     ok
O prose block then code   ok        ok        WRONG     ok        ok
P last block is output    ok        WRONG     WRONG     ok        WRONG
```

Readings that change the design:

- **open-r1's hard-coded `` ```python `` fails four common variants** (bare fence, `Python`, `py`, `python3`) and returns `""`, which its pipeline then executes as an empty program. Those rollouts score 0 and are indistinguishable from wrong answers.
- **A trailing space after the language tag defeats every regex-based extractor** (case E) because they all require `` ```python\n `` exactly. Only the `split`-based and line-scan approaches survive.
- **Truncation at `max_completion_length` (case F) leaves an unterminated block, and every fence-matching extractor returns nothing.** This is not exotic — it is the single most likely malformation on CodeContests, and it is why [`rlvr-stack.md` §1.3](./rlvr-stack.md) recommends `mask_truncated_completions=True`. Without that flag, truncated rollouts are scored as format failures.
- **code-r1's join-all-blocks is right for case H and wrong for cases G and O** — it concatenates the naive and final solutions, or glues a prose block onto the code.
- **rLLM's and LiveCodeBench's last-block rule breaks on case P** — a model that prints its expected output in a trailing bare fence loses its code entirely. open-r1's language-anchored version is the only one that survives P, precisely because it is strict.

So the two strictness dimensions trade off against each other: *tag-strict* extractors survive trailing output blocks and die on tag variants; *tag-loose* extractors survive tag variants and die on trailing output blocks. **No single-rule extractor in the wild is robust to both.** That is the argument for a cascade.

---

## 8. Recommendation for this project

Context: Qwen2.5-3B-Instruct → 7B, TRL `GRPOTrainer`, `deepmind/code_contests`, Python only, single turn, stdin/stdout judging, reward ladder whose rung 0 is "no code extracted".

### 8.1 Anchor on code-r1, because it is literally this task

§1.4 and §6.3 establish that [`ganler/code-r1`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L208-L210) trains on `deepmind/code_contests`, Python, stdin/stdout, single turn — the same task, the same dataset, the same judging model. It is the closest primary-source precedent that exists. Two of its three decisions should be copied and one should not:

- **Copy: format failure gets a distinct, negative reward.** `−answer_reward − format_reward = −1.1` versus `+0.1` for a wrong answer ([`coder1/__init__.py#L74`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/verl/utils/reward_score/coder1/__init__.py#L74)). This is the only surveyed implementation that guarantees an all-fail group still has reward variance.
- **Copy: an explicit "in a Python markdown code block" instruction in the user prompt** ([`coder1.py#L233-L236`](https://github.com/ganler/code-r1/blob/443f8da8058d07051793eff875da117534fd1ba1/examples/data_preprocess/coder1.py#L233-L236)).
- **Do not copy: `'\n'.join(code_blocks)`.** Case G in §7 shows it concatenates a discarded naive solution with the final one. On CodeContests, where models routinely sketch a brute-force check before the real algorithm, this will produce duplicate-definition programs that run the wrong version.

### 8.2 Extraction policy — a syntax-gated cascade

The §7 bench shows no single rule is robust. Use tiers, and **record which tier fired**, because the tier distribution is the measurement §5 says nobody has published.

```python
import ast, re

FENCE = re.compile(
    r"^[ \t]*```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)",
    re.DOTALL | re.MULTILINE,
)
PY_TAGS = {"python", "python3", "py", "py3", "pycon"}


def syntax_ok(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def extract_python(completion: str, prefill: str = "") -> tuple[str | None, str]:
    """Returns (code, tier). Tier is logged, not just used."""
    text = prefill + completion            # see 8.4 — MUST re-prepend the prefill
    blocks = FENCE.findall(text)
    tagged   = [b for t, b in blocks if t.lower() in PY_TAGS]
    untagged = [b for t, b in blocks if t == ""]
    other    = [b for t, b in blocks]
    for tier, cands in (("tagged", tagged), ("untagged", untagged), ("any", other)):
        valid = [b for b in cands if b.strip() and syntax_ok(b)]
        if valid:
            return valid[-1], tier          # LAST syntactically valid candidate
        if cands and tier == "any":
            return cands[-1], "any_invalid" # a block exists but does not parse
    if text.strip() and syntax_ok(text):
        return text, "bare"                 # no fence at all, but valid Python
    return None, "none"
```

Every design choice traces to a source or to §7:

| Choice | Why |
| --- | --- |
| **Last** candidate, not first | five of seven implementations (§1.8); LiveCodeBench has the first-block version commented out directly above the last-block one ([`extraction_utils.py#L16-L17`](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/utils/extraction_utils.py#L16-L17)) |
| Tag optional but **python-tagged blocks preferred over untagged** | fixes case P (trailing bare output block) *and* cases B/C/D/N (tag variants) — the tradeoff §7 says no single rule solves |
| `[ \t]*` around the tag; `^…$` multiline | fixes case E (trailing space) and case M (indented closing fence) |
| `(?:\n[ \t]*```|\Z)` — closing fence **or end of string** | fixes case F, the truncation case, which is the most likely real malformation |
| **`ast.parse` gate on candidate selection** | the only fence-independent signal any surveyed tool uses; EvalPlus builds its whole extractor on it ([`sanitize.py#L31-L49`](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/sanitize.py#L31-L49)). Cheap here (a handful of `ast.parse` calls) rather than EvalPlus's O(n²) span search |
| One `bare` tier, syntax-gated | recovers case I (valid code, no fence) without verl `prime_code`'s failure of shipping prose to the interpreter ([`prime_code/__init__.py#L23`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/prime_code/__init__.py#L23)) — prose does not parse, so `none` still fires on case J |
| Never fall back to the whole completion unguarded | that is exactly verl `prime_code`'s bug; `sandbox_fusion` instead returns a typed error ([`sandbox_fusion/__init__.py#L61`](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/reward_score/sandbox_fusion/__init__.py#L61)) and is the better model |

This cascade returns `ok` on all sixteen §7 cases plus three prefill cases; the reference implementations return `ok` on 9–12 of them.

### 8.3 Reward ladder, and how the tier feeds it

Keep the binary all-tests-pass primary reward argued for in [`rlvr-stack.md` §5.B](./rlvr-stack.md), and add **one** auxiliary reward function whose only job is to inject group variance:

| Tier / outcome | Suggested value |
| --- | --- |
| `none` (no code at all) | `−1.0` |
| `any_invalid` (a block, but it does not parse) | `−0.5` |
| `bare` (valid Python, no fence) | `0.0` |
| `untagged` | `+0.5` |
| `tagged` | `+1.0` |

then weight it at **0.1** against the 1.0 execution reward, matching open-r1's two code recipes and code-r1 ([§2.2](#22-what-the-code-rl-implementations-actually-weight-it-at)). Note this is a *graded extractability* reward, not open-r1's `code_format_reward`: the latter also demands `<think>/</think>/<answer>` wrappers ([`rewards.py#L609`](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py#L609)) and therefore measures template compliance rather than whether the code can be run. **We only care about the latter.**

Mechanically, in TRL:

- Pass it as a **separate entry in `reward_funcs`** with `reward_weights=[1.0, 0.1]`. TRL's default `multi_objective_aggregation="sum_then_normalize"` sums the weighted terms before normalising ([`grpo_config.py#L767-L778`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L767-L778)), so this genuinely rescues an otherwise-degenerate group.
- Set **`mask_truncated_completions=True`** ([`grpo_config.py#L826`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py#L826)). Without it, case F (truncated → unterminated block) is scored as a format failure, and you will train the model to write shorter programs rather than better ones.
- **Log the tier histogram every step.** TRL passes `log_metric` and `log_extra` into every reward function ([`grpo_trainer.py#L1598-L1609`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1598-L1609), `grpo_trainer.py#L1624-L1627`):

  ```python
      def _log_metric(self, name: str, value: float):
          """
          Log a scalar metric from a reward function. Called via the `log_metric` kwarg. Values are averaged over each
          logging step and reported alongside built-in metrics like `kl` and `entropy`.
          """
  ```

  so `log_metric("format/frac_none", …)` gets you the number §5 says nobody publishes, for free, on your own model and prompt. Watch it against `frac_reward_zero_std` ([`rlvr-stack.md` §1.4](./rlvr-stack.md)).

**Do step 0 first, and be willing to delete this reward function.** Before any training, run the extractor over a few hundred greedy/temperature-1 completions from Qwen2.5-3B-Instruct on the §8.4 prompt and get the tier histogram. Then:

- **If `none` + `any_invalid` is under ~2%**, the auxiliary reward is buying almost no group variance and you should consider dropping it. §5.5b found outcome-only RLVR reaches the same format-adherence as a dedicated format reward *with better accuracy*, and §2.6 found correctness-only GRPO moved format accuracy 87.3 → 95.0 by itself. Adding a term that is already near-saturated mostly adds a way to be wrong.
- **If it is 5–25%** (the range §5.5a's math analogue suggests is plausible), keep it at 0.1 — but expect the metric to decay toward zero within the first tens of steps, at which point it stops contributing variance and the degenerate-group problem reverts to being about wrongness, per [`rlvr-stack.md` §5.B.1](./rlvr-stack.md).
- **If it is above ~25%**, do not reach for a bigger format weight — reach for the **prompt** (§5.5a: a 21-point parse-rate swing from wording alone) and then for **prefill** (§8.5). SimpleRL-Zoo's warning is the operative one here, verbatim: *"imposing a format reward will penalize many correct explorations."*

The failure mode to avoid is escalating the format weight because the metric is not falling fast enough. That is the path SimpleRL-Zoo describes as *"The model expends excessive effort on adhering to the format but fails to learn how to answer correctly."*

### 8.4 Prompt template

Synthesised from the three sources that pin format hardest: open-r1's Codeforces prompt (explicit instruction + inline fence demonstration + terminal reminder, §6.1), rLLM/LiveCodeBench's `### Answer: (use the provided format with backticks)` (§6.2), and code-r1's stdin/stdout framing for CodeContests specifically (§6.3).

System:

```
You are an expert competitive programmer. You write correct, efficient Python 3 solutions.
```

User:

````text
Solve the programming task below in Python 3.

Your program must read from standard input and write to standard output. Do not print
anything other than the required output. Do not read or write files.

Reason about the solution first. Then give your complete final program in a single
Python markdown code block, formatted exactly like this:

```python
# your complete program here
```

{description}

Now solve the problem and return the code in a single ```python block.
````

Deliberate choices:

- **No `<think>/<answer>` tags.** open-r1 and code-r1 both require them, but both were training reasoning-distilled or R1-style models. Qwen2.5-3B-Instruct is not one, and adding a template requirement adds a second way to fail that we do not score and do not need. DeepSeek-R1's own language-consistency ablation ([§2.1](#21-deepseek-r1--what-it-actually-rewarded-and-what-it-did-not-say)) is the cautionary precedent: a presentation-oriented reward term cost measurable capability.
- **Inline fence demonstration**, from open-r1's `` ```cpp\n<your code here>\n``` `` and rLLM's `` ```python\n# YOUR CODE HERE\n``` ``. Both implementations that do this are the ones whose extractors are tag-strict.
- **Terminal reminder naming the tag**, from rLLM's `"### Answer: (use the provided format with backticks)"` — it sits immediately before the generation point, where it has the most conditioning weight.
- **Explicit "single" code block**, from open-r1's *"Put your final solution within a single code block"*, which reduces the case-G/case-H ambiguity at the source rather than in the extractor.

### 8.5 On prefill: hold it in reserve, and if you use it, do it this way

**Do not prefill by default.** Prefilling `` ```python\n `` immediately after `<|im_start|>assistant\n` forecloses reasoning, which every implementation in §6 explicitly asks for and which is where the actual performance on CodeContests comes from.

**But it is safe, and it is the correct escape hatch if §5's measurement comes back bad.** The mechanism is sound in a way constrained decoding is not: prefill conditions the prompt, so the model's distribution over generated tokens remains its own true conditional distribution — there is no behaviour-policy/target-policy gap, no logprob inconsistency, nothing for importance sampling to correct (§4.4). And TRL masks it correctly for free, because prompt tokens carry no loss ([`grpo_trainer.py#L2388`, `#L2426`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L2426)).

If you do it:

1. **Do not** try to pass it through `chat_template_kwargs` — `add_generation_prompt=True` is hard-coded at [`grpo_trainer.py#L1762`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1762) and both `add_generation_prompt=False` and `continue_final_message=True` will raise.
2. Pre-render with `tokenizer.apply_chat_template(msgs, continue_final_message=True, tokenize=False)` and store the **string** in the `prompt` column. TRL's non-conversational branch tokenizes it verbatim ([`grpo_trainer.py#L1779`](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py#L1779)).
3. **Re-prepend the prefill inside the reward function** — `extract_python(completion, prefill="```python\n")`. §4.4 shows the completion reaches you starting *inside* the block. The bench in §7 confirms the failure: forgetting this returns `''` on a well-formed completion, i.e. silently scores every rollout at rung 0.
4. Assert once at startup that the rendered prompt ends with the literal `` ```python\n `` including the newline — §4.2's `rstrip()` fallback in `transformers` is a live hazard if the template ever changes.

### 8.6 On constrained decoding: no

Three independent reasons, in order of force:

1. **It is unavailable on the recommended generation backend.** [`rlvr-stack.md` §1.9](./rlvr-stack.md) recommends `use_transformers_continuous_batching=True` on a single 5090; TRL's structured-output plumbing exists only on the vLLM path (§3.1).
2. **It destroys the variance you are trying to buy.** If the grammar makes rung 0 unreachable, an all-wrong group is degenerate again (§3.2 point 4). Constrained decoding removes the symptom and keeps the disease.
3. **The correction is partial.** TRL's IS ratio does compare against the true constrained sampling distribution (§3.2), which is better than expected — but the GRPO group baseline is still an expectation under the constrained policy, and TRL's default `sequence_mask` mode can silently zero out rollouts whose sequence-level ratio collapses. Prefill achieves the same end with none of this.


---

## 9. Risks and unknowns

Blunt list of what I could **not** verify from a primary source, ordered by how much it could change the recommendation.

### 9.1 The measurement this whole document is about does not exist

1. **Nobody has published a parse-failure rate for code generation, for any model, on any benchmark.** Not the Qwen2.5 or Qwen2.5-Coder technical reports (verified: zero occurrences of `parse failure`, `extraction failure`, `format failure`, `unparseable`, `malformed`, `sanitiz`, `postprocess`, `markdown` across both), not LiveCodeBench, not BigCodeBench, not EvalPlus, not simple-evals. **The specific number the premise of this project depends on — "how often does Qwen2.5-3B-Instruct fail to emit a parseable code block on a CodeContests prompt" — has never been measured in public.** We must measure it ourselves. §8.3 gives the mechanism.
2. **I did not verify `lm-evaluation-harness` or HELM** for a parse-failure bucket. Delegated and not returned. Given the other five harnesses, I would be surprised, but it is unchecked.
3. **The closest analogues are all math, not code.** Spurious Rewards' 72–93% parsable range (§5.5a) is `\boxed{}` on MATH-500 with Qwen2.5-**Math**-7B. Whether a code fence on CodeContests with Qwen2.5-3B-**Instruct** behaves anything like that is an assumption, not a finding. Code fences are arguably easier (the model has seen millions of them in markdown) and the prompt is far more explicit.
4. **CodeScaler's specific "fragment rate" and "invalid rate" percentages were relayed to me and I could not find them in the v1 HTML.** Not quoted, not relied on. Its *methodology* text I verified directly.
5. **I could not machine-extract the individual cell values of One-shot RLVR's Table 14** (coloured markup). I quoted only the caption and prose findings, which I verified.

### 9.2 Things I reasoned about but did not run

6. **The entire §3.2 analysis of constrained decoding × TRL's importance sampling is my own reasoning, not a citation.** I verified every mechanism it rests on (bitmask applied before sampling; both logprob modes read the masked tensor; TRL's IS ratio compares training logprobs against vLLM's sampling logprobs; the default mode is `sequence_mask` with `clip_max=3.0`). But **no primary source works this composition through**, I did not run it, and the claim that a tight grammar would collapse the sequence-level ratio and get rollouts silently masked out is a prediction. It is also the reason §8.6 says "no" — a recommendation that costs us nothing, since constrained decoding is unavailable on the recommended backend anyway.
7. **The §7 differential bench is my own code, not anyone's published test suite.** The extractors are transcribed verbatim from the sources in §1, but the sixteen test cases are hand-written by me to probe failure modes those sources *mention*; they are not sampled from real Qwen2.5 rollouts. **The frequency with which each case occurs in practice is exactly the unknown in 9.1.** A case that never happens does not matter however badly an extractor handles it.
8. **I never executed the recommended cascade in §8.2 against real model output.** It passes my own 19 cases by construction — I wrote both. That is a consistency check, not evidence.
9. **The `rstrip()` prefill hazard in §4.2 is read from source, not observed.** I traced that Qwen2.5's template does not trim trailing whitespace (`+ message.content +` is verbatim) and therefore that the fast path preserves the trailing `\n` in a `` ```python\n `` prefill. I did not run `apply_chat_template` to confirm. §8.5 step 4 exists because of this.

### 9.3 Gaps in the sources themselves

10. **No paper argues about format rewards for *instruction-tuned* models.** SimpleRL-Zoo, Open-Reasoner-Zero and Dr. GRPO all reason about **base** models in a zero-RL setting. SimpleRL-Zoo's Qwen-2.5-7B sentence is the nearest transfer, and that is Qwen2.5-**Base**. Our model is `-Instruct`. **Applying their conclusion to us is an extrapolation.** It happens to be an extrapolation in the safe direction (instruction-tuned models format *better*, so a format reward is even less necessary), but it is unproven.
11. **SimpleRL-Zoo's format-reward ablation states no numbers** — Figure 6 is a plot and I refuse to eyeball values off an axis. Its conclusion is stated in prose and I have quoted it, but the effect size is unknown to me.
12. **No documented case of a *code-fence* format reward being gamed.** Every gaming report in §2.5 is about `<think>`/`<answer>` reasoning tags. The asymmetry is plausible (a fence is a thin target) but it is an absence of evidence.
13. **The 0.1 format weight has no source.** It is a convention that propagated through open-r1 and code-r1 configs. The field-wide range is 0.0 (verl's default, OpenRLHF, ORZ, DeepCoder) to 1/3 of total reward (Logic-RL). DeepSeek-R1 states **no numeric weight anywhere**.
14. **No maintainer anywhere has ever addressed TRL's constrained-decoding correctness.** In [trl#2811](https://github.com/huggingface/trl/pull/2811) the contributor explicitly invited scrutiny and the reply discussed JSON-serializability. That silence is data, but it is not an answer.
15. **verl's `prime_code` `UnboundLocalError`** (§1.3) I found by reading, not by running. It is unreachable via `default_compute_score` because verl always passes `continuous=True`. If you ever call it directly with `continuous=False`, expect a crash.
16. **I did not verify that TRL's `vllm_structured_outputs_regex` actually works end-to-end.** It has exactly one test (`test_train_vllm_structured_outputs`) and zero occurrences across TRL's `docs/` and `examples/`. Moot for us — §8.6 recommends against it — but do not assume it is battle-tested.

### 9.4 The blunt summary

**On extraction**: there is a real consensus — *last fenced block, optional language tag* — and it is contradicted by Qwen's own harness, which takes the first. There is **no** consensus on what to do when extraction fails, and that, not the matching rule, is the decision that interacts with GRPO. Use the syntax-gated cascade in §8.2 and log which tier fired.

**On prefill**: safe, mechanically sound, and unattested in any RL loop. TRL masks it for free because prompt tokens carry no loss. The one real hazard is forgetting to re-prepend it in the reward function, which silently scores every rollout at zero. Hold it in reserve; deploy it only if step-0 measurement shows a bad format-failure rate.

**On whether the problem is real**: **partly a phantom, but not entirely.** The premise — that format failures kill GRPO groups — is structurally correct and confirmed in TRL's source. But the frequency is almost certainly lower than the framing suggests, the literature consistently finds RL fixes formatting on its own within the first tens of steps, and the one code-domain source says outright that *"in RLVR, code extraction is straightforward as the execution environment naturally filters flawed or syntactically incorrect code."* The bigger risk for CodeContests is the one [`rlvr-stack.md` §5.B.1](./rlvr-stack.md) already identified: **groups going degenerate because every rollout is simply wrong.** Format failure is a small contributor to that, worth a cheap 0.1-weighted auxiliary term and worth measuring — but it is not the main event, and a heavy-handed fix (strict format reward, constrained decoding) risks costing more than it saves. SimpleRL-Zoo's warning is the one to keep in view: *"imposing a format reward will penalize many correct explorations."*
