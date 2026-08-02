# Generate rollouts with transformers, not vLLM

Rollout generation uses TRL's `use_transformers_continuous_batching=True` rather than vLLM.
On a single 32 GB card vLLM buys throughput we do not need and costs a second full copy of
the weights, a weight-synchronisation path that is unsupported with QLoRA, and several open
sm_120 defects. This is deliberate and should not be "fixed" by adding vLLM back.

## Considered Options

**vLLM colocate mode** is the obvious choice and was rejected on evidence gathered in
`docs/research/rlvr-stack.md`:

- QLoRA + vLLM weight sync is unsupported in server mode (confirmed by a TRL maintainer) and
  appears broken in colocate mode, where TRL pushes dequantised bf16 tensors into packed
  `uint8` parameters. TRL's `test_train_vllm_and_peft` is skipped, so there is no green CI
  test for the combination.
- vLLM's in-tree bitsandbytes support is on a deprecation path to an out-of-tree plugin, and
  its 4-bit kernel is measured 4× slower than FP16 at batch size 1.
- 7B in bf16 with a colocated bf16 vLLM engine needs roughly 35 GB and does not fit.

**Plain `model.generate()`** would also work and is simpler still, but continuous batching
is a drop-in improvement over it with no server, no weight sync, and no second copy of the
weights.

## Consequences

Generation is slower than a vLLM setup would be; on a single card VRAM, not throughput, is
the binding constraint. Continuous batching's interaction with `Linear4bit` is untested
upstream, so it must be verified when moving to 7B, with plain `.generate()` as the fallback.
