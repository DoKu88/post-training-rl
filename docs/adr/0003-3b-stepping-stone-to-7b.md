# Qwen2.5-3B is a stepping stone; 7B is the target

The deliverable is a 7B run. The 3B run exists to prove the verifier, scorer, and reward
functions are correct, not to produce a good model — its reward curve says nothing about
whether the method works. Model identity, quantisation scheme, and generation backend all
live in YAML so that scaling up is a config change rather than a code change.

## Consequences

3B runs bf16 base + LoRA (~6 GB). 7B requires NF4 QLoRA to fit (~9 GB without vLLM), and
`load_in_4bit=True` alone is not QLoRA — `bnb_4bit_quant_type="nf4"`,
`bnb_4bit_use_double_quant=True` and `bnb_4bit_compute_dtype=torch.bfloat16` must all be set
explicitly.

The two models differ structurally — 3B has 36 layers, 2 KV heads and tied embeddings; 7B
has 28 layers, 4 KV heads and an untied `lm_head` — so anything computing per-layer sizes
must read them from the config rather than assuming.
