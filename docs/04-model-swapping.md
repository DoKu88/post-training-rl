# Step 4 — Swap in Qwen2.5-7B (Model-Agnostic Architecture)

**Goal:** switch the target model from 3B to **Qwen2.5-7B-Instruct** by changing
**only config** — and, crucially, make the loader accept **any arbitrary HF
model** so future swaps (Llama, DeepSeek, a new Qwen) are also config-only.

**Requirement (verbatim from the task):** *"The architecture should support model
swapping to any arbitrary model."*

---

## Sprints at a glance

| Sprint | Name | Deliverable | Depends on | GPU? |
|--------|------|-------------|------------|------|
| 4.1 | The `load_policy(cfg)` choke point | One config-driven function loads model + tokenizer for any HF model; swap is config-only | — | mostly CPU / tiny model |
| 4.2 | Quantization + vLLM quantized-copy setup | `build_quant_config` (4-bit) and the double-weights fit for both trainer & vLLM copies | 4.1 | GPU for real fit check |
| 4.3 | 7B loads+generates in 32GB + model-swap regression | Qwen2.5-7B loads and generates in budget; same `train.py` runs 3B & 7B via YAML only | 4.1, 4.2 | yes |

Core requirement under test throughout: **a model swap is CONFIG-ONLY** — no
Python edits. Cheap, model-agnostic behavior is tested with a **tiny random HF
model** (CPU / model-free); real-7B and vLLM paths are marked `@pytest.mark.gpu`.

---

## Sprint 4.1 — The `load_policy(cfg)` choke point

**Deliverable:** every model concern lives behind **one function** in
`posttrain.models.loader`; no other file names a model, and pointing a config
field at a different YAML swaps the model.

**Depends on:** —

**Build:**
- Implement `load_policy(cfg: ModelConfig) -> (model, tokenizer)` that is fully
  config-driven — nothing in the signature is Qwen-specific.
- Use `AutoModelForCausalLM` / `AutoTokenizer` (architecture-agnostic), handle the
  common `pad_token is None` gotcha, and rely on the tokenizer's own chat template.
- Add the config layer (`configs/model/*.yaml`, `configs/experiment/*.yaml`) so
  swapping = pointing `model._file` at a different YAML.

```python
# src/posttrain/models/loader.py
def load_policy(cfg: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Config-driven. Nothing in this signature is Qwen-specific."""
    quant_cfg = build_quant_config(cfg.quantization)      # none|4bit|8bit  (doc 5)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name_or_path,
        torch_dtype=cfg.dtype,               # e.g. bfloat16
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_impl,   # "flash_attention_2" if available
    )
    tok = AutoTokenizer.from_pretrained(cfg.name_or_path, trust_remote_code=...)
    if tok.pad_token is None:                # common gotcha across models
        tok.pad_token = tok.eos_token
    if cfg.lora.enabled:                     # doc 5 — untouched here
        model = apply_lora(model, cfg.lora)
    return model, tok
```

Why this satisfies "any arbitrary model":

- **`AutoModelForCausalLM` / `AutoTokenizer`** already abstract architecture.
- **Chat template comes from the tokenizer** (doc 1.4), so prompt formatting
  adapts automatically per model family.
- Model-specific quirks (`pad_token`, `trust_remote_code`, attention impl) are
  handled here, once.

### Config, not code

`configs/model/qwen7b.yaml`:
```yaml
name_or_path: Qwen/Qwen2.5-7B-Instruct
dtype: bfloat16
device_map: auto
trust_remote_code: true
attn_impl: flash_attention_2
quantization: 4bit          # see doc 5 — needed to fit 7B on 32GB
lora: {enabled: true, r: 32, alpha: 64,
       target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}
```

`configs/experiment/train_7b.yaml`:
```yaml
defaults: [base]
model: {_file: model/qwen7b.yaml}
algo:  {_file: algo/grpo.yaml}
data:  {dataset: deepmind/code_contests, split: train, difficulty: [easy, medium], max_problems: null}
train: {max_steps: 2000, gradient_checkpointing: true}
rollout: {engine: vllm, gpu_memory_utilization: 0.45}
```

Swapping model = point `model._file` at a different YAML. That's the whole change.

**Unit tests**
- `tests/test_loader.py::test_pad_token_set_when_missing` — tiny random model
  whose tokenizer has no pad token; assert `tok.pad_token is not None` after
  `load_policy` (== eos). *(CPU / tiny)*
- `tests/test_loader.py::test_chat_template_applies` — assert
  `tok.apply_chat_template([...], tokenize=False)` returns a non-empty str for
  the tiny model without raising. *(CPU / tiny)*
- `tests/test_loader.py::test_generate_produces_text` — `load_policy` on a
  tiny-random causal model, assert `model.generate(...)` decodes to a non-empty
  string. *(CPU / tiny)*
- `tests/test_loader.py::test_swap_is_config_only` — load two different
  `ModelConfig`s (tiny-A, tiny-B) through the **same** `load_policy` call site;
  assert both return `(model, tok)` and that `loader.py` source contains no
  hard-coded `name_or_path` (model name only comes from cfg). *(model-free / CPU)*
- `tests/test_loader.py::test_config_yaml_selects_model` — parse
  `configs/model/*.yaml`, assert changing `model._file` changes the resolved
  `cfg.name_or_path` with zero code change. *(model-free)*
- `tests/test_loader.py::test_load_real_1_5b` — `@pytest.mark.gpu` — load
  `Qwen/Qwen2.5-1.5B-Instruct`, assert generate works. *(GPU)*

**✅ Verify it works (you run)**
```bash
conda activate post-train
python -c "from posttrain.models.loader import load_policy; from posttrain.config import load_cfg; \
m,t = load_policy(load_cfg('configs/model/tiny.yaml')); \
print(t.pad_token, t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False)[:20], \
t.decode(m.generate(**t('hi', return_tensors='pt'), max_new_tokens=5)[0]))"
```
Expected: prints a non-None pad token, a rendered chat prefix, and generated
text — all from config, with no model name hard-coded in Python.

---

## Sprint 4.2 — Quantization + vLLM quantized-copy setup

**Deliverable:** `build_quant_config` produces a working 4-bit config, and the
rollout config loads a **quantized** vLLM copy so both resident base-weight
copies fit the 32GB budget.

**Depends on:** 4.1

**Build:**
- Implement `build_quant_config(spec)` mapping `none|4bit|8bit` → the right
  `BitsAndBytesConfig` (or `None`), wired into `load_policy`.
- Add rollout config knobs for a **4-bit AWQ/GPTQ** vLLM build and document the
  double-weights memory reality below.

### The 7B memory reality (the double-weights trap)

From our earlier analysis — the thing people miss:

> GRPO needs **two** resident copies of the base weights: the **trainer's** copy
> and **vLLM's** copy. They are *not* shared.

| Config | Trainer base | vLLM base | Headroom (of 32GB) | Fits? |
|--------|-------------|-----------|--------------------|-------|
| 7B bf16 + bf16 vLLM | 14 GB | 14 GB | ~4 GB | ❌ OOM |
| **7B 4-bit QLoRA + 4-bit (AWQ/GPTQ) vLLM** | ~5 GB | ~5–6 GB | ~20 GB | ✅ |
| 3B bf16 | 6 GB | 6 GB | ~20 GB | ✅ |

**Therefore:** doc 4 and doc 5 are coupled — you cannot run 7B on this GPU without
the quantization + LoRA from doc 5. The loader already exposes both knobs; this
sprint wires them and 4.3 confirms the model *loads and generates*, doc 5 confirms
it *trains* well.

### vLLM must use a quantized copy too

Set up vLLM to load a **4-bit AWQ/GPTQ** build of Qwen2.5-7B (or bnb-in-vLLM if
your vLLM version supports it), else the vLLM copy alone eats 14 GB. In config:
```yaml
rollout:
  engine: vllm
  model_quant: awq                # vLLM-side quantization
  gpu_memory_utilization: 0.45
```

**Unit tests**
- `tests/test_quant.py::test_build_quant_none_returns_none` — assert
  `build_quant_config("none") is None`. *(model-free)*
- `tests/test_quant.py::test_build_quant_4bit_fields` — assert the 4-bit result
  sets `load_in_4bit=True` and the expected compute dtype / quant type.
  *(model-free)*
- `tests/test_quant.py::test_build_quant_8bit` — assert 8-bit spec sets
  `load_in_8bit=True`. *(model-free)*
- `tests/test_quant.py::test_rollout_config_declares_quant` — parse the 7B
  experiment YAML, assert `rollout.model_quant` is set (not full-precision) so
  the vLLM copy is quantized. *(model-free)*
- `tests/test_quant.py::test_loader_passes_quant_config` — `load_policy` with a
  4-bit `ModelConfig` on the tiny model passes a non-None `quantization_config`
  into `from_pretrained` (patch/spy, no real quantization needed). *(CPU / tiny)*
- `tests/test_quant.py::test_7b_4bit_fits_budget` — `@pytest.mark.gpu` — load
  Qwen2.5-7B in 4-bit, assert peak allocated < ~6 GB for the trainer copy.
  *(GPU)*

**✅ Verify it works (you run)**
```bash
conda activate post-train
python -c "from posttrain.models.loader import build_quant_config; \
c=build_quant_config('4bit'); print(c.load_in_4bit, c.bnb_4bit_quant_type)"
python -c "from posttrain.config import load_cfg; c=load_cfg('configs/experiment/train_7b.yaml'); \
print(c.rollout.engine, c.rollout.model_quant, c.rollout.gpu_memory_utilization)"
```
Expected: 4-bit config reports `True nf4` (or configured type); the 7B rollout
config reports `vllm awq 0.45`, confirming both trainer and vLLM copies are
quantized so the two base copies fit ~20GB headroom.

---

## Sprint 4.3 — 7B loads+generates in 32GB + model-swap regression

**Deliverable:** Qwen2.5-7B-Instruct **loads and generates** within the 32GB
budget (4-bit both copies), and the **exact same `train.py`** runs a few steps
with the 3B and 7B configs — differing only in YAML. This is the proof of
model-agnosticism.

**Depends on:** 4.1, 4.2

**Build:**
- Run the full 7B path end-to-end through `load_policy` + a short generation.
- Add the parametrized model-swap regression test over a **tiny** model plus the
  real ones, and a short `train.py` smoke run driven only by config selection.

**Unit tests**
- `tests/test_model_swap.py::test_generate_across_sizes[tiny]` — parametrized;
  `load_policy` returns a model whose `generate` produces text for the tiny
  random model. *(CPU / tiny)*
- `tests/test_model_swap.py::test_chat_template_across_sizes[tiny]` — tokenizer
  chat template applies without error for the tiny model. *(CPU / tiny)*
- `tests/test_model_swap.py::test_pad_token_across_sizes[tiny]` — `pad_token` is
  set for the tiny model. *(CPU / tiny)*
- `tests/test_model_swap.py::test_train_py_runs_from_config_only` — invoke the
  same `train.py` entrypoint with the 3B-smoke YAML and a tiny YAML; assert both
  complete N steps and that only the config path differed (no code branch on
  model name). *(CPU / tiny where possible)*
- `tests/test_model_swap.py::test_generate_across_sizes[qwen7b]` —
  `@pytest.mark.gpu` — 4-bit Qwen2.5-7B loads and `generate` yields non-empty
  text within budget. *(GPU)*
- `tests/test_model_swap.py::test_train_py_3b_and_7b_via_yaml` —
  `@pytest.mark.gpu` — the same `train.py` runs a few steps with the 3B smoke
  config and the 7B config, only the YAML differs, no OOM. *(GPU)*

**✅ Verify it works (you run)**
```bash
conda activate post-train
# 7B loads + generates in 4-bit within 32GB
python -m posttrain.tools.smoke_generate --config configs/experiment/train_7b.yaml --max-new-tokens 32
nvidia-smi --query-gpu=memory.used --format=csv
# same train.py, config-only swap 3B -> 7B
python -m posttrain.train --config configs/experiment/train_3b_smoke.yaml train.max_steps=3
python -m posttrain.train --config configs/experiment/train_7b.yaml       train.max_steps=3
```
Expected: one config-field change swaps 3B→7B; both **load + generate**; 7B fits
32GB in 4-bit (both copies quantized, `nvidia-smi` well under 32000 MiB); the
identical `train.py` completes 3 steps for each config with no OOM and no Python
edits.

---

## Definition of done

- Changing one config field swaps the model; **no Python edits** (config-only is
  proven by `test_swap_is_config_only` and `test_train_py_runs_from_config_only`).
- Qwen2.5-7B **loads and generates** within the 32GB budget (4-bit both the
  trainer and the vLLM copies).
- `tests/test_model_swap.py` green across ≥3 model sizes (tiny + 1.5B + 7B).
- A short 7B run executes steps without OOM (quality comes in doc 5).
- The loader is the single choke point: no file other than
  `posttrain.models.loader` names a model.

## Run all tests for this step

```bash
conda activate post-train
# fast, model-agnostic subset (CPU / tiny model, no GPU needed):
pytest tests/test_loader.py tests/test_quant.py tests/test_model_swap.py -m "not gpu" -q
# full suite incl. real 7B / vLLM (needs the 32GB GPU):
pytest tests/test_loader.py tests/test_quant.py tests/test_model_swap.py -q
```
