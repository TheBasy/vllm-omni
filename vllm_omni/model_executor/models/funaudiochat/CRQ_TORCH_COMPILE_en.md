# CRQ sidecar `torch.compile` optimization (`modeling_funaudiochat.py`)

`crq_torch_compile.patch` in this directory is a performance diff against
**HF model repo `FunAudioLLM/Fun-Audio-Chat`**'s `funaudiochat/modeling_funaudiochat.py`
— it is **not vllm-omni runtime code**. It is kept in vllm-omni's `funaudiochat/`
folder only for version control and traceability, so the optimization lives
alongside the fun_audio_chat integration instead of being scattered across the
model repo / deploy host.

## What the optimization does

Applies `torch.compile(mode="default")` (inductor kernel fusion — **not cudagraph**)
to the CRQ sidecar transformer step inside `FunAudioChatDecoder.crq_generate_forward`:

- Measured: ~18k → ~7k GPU launches/step; ~60ms → ~36ms gpu_busy/step; api_enqueue
  share halved. Relieves the **stage-0 launch-bound** bottleneck on the
  Fun-Audio-Chat 8B sidecar.
- Why not cudagraph (`reduce-overhead`): the sidecar's `DynamicCache` grows each
  step; cudagraph's static buffer overwrites in Qwen3's `apply_rotary_pos_emb`
  and crashes. `default` mode only fuses kernels and does not staticize, so it
  is safe.
- The compiled target is a plain `@staticmethod`
  (`_crq_forward_fn(tower, input_embeds, past_key_values)`, taking module refs +
  tensors) so torch can trace it; KV (`DynamicCache`) is passed in/out;
  **sampling stays eager** (dynamic distribution / multinomial cannot be
  compiled); input slicing + `input_matching` also stay eager (shape-dynamic).
- `_crq_compiled_step` / `_crq_force_eager` are **class-level** attributes;
  `FUNAUDIO_CRQ_COMPILE` is read **once at import**; multiple instances / threads
  share the same compiled fn (staticmethod + `tower` passed by reference) — safe.
- Compiled is **ON by default**. Emergency fallback:
  `export FUNAUDIO_CRQ_COMPILE=0` → eager (equivalent to the original loop).
- Refactor: `crq_generate_forward` is split into `crq_transformer_step` /
  `crq_sample_token` / `crq_embed_token` (pure refactor, eager path bit-exact);
  the compiled path adds `crq_transformer_step_compiled` separately.

## Relationship to vllm-omni (important)

- vllm-omni **requires no changes**. `vllm_omni/model_executor/models/funaudiochat/`
  is the vllm integration wrapper; this patch modifies the **model repo's**
  `modeling_funaudiochat.py`, which is loaded at runtime by serve's
  `--trust-remote-code` from `FUN_AUDIO_CHAT_HOME`.
- So this patch does **not take effect inside the vllm-omni process**; it must be
  applied to the model-repo clone on the deploy host (see below) and the service
  restarted.

## Deployment (on the deploy host)

Prerequisite: `FUN_AUDIO_CHAT_HOME` points to the clone of the `Fun-Audio-Chat`
model repo (contains `funaudiochat/modeling_funaudiochat.py` + `utils/`).

```bash
cd "$FUN_AUDIO_CHAT_HOME"
# `patch -p1` is recommended (more tolerant of whitespace / base drift);
# `git apply` also works.
patch -p1 < <vllm-omni repo>/vllm_omni/model_executor/models/funaudiochat/crq_torch_compile.patch
# or: git apply --3way --reject crq_torch_compile.patch
```

Then:

1. **Verify `--trust-remote-code` actually loads this file** — not a different
   flat `modeling_funaudiochat.py` inside the weights dir `Fun-Audio-Chat-8B/`
   (a common `trust_remote_code` gotcha). Add a temporary `print(__file__)` at
   the top of the file or check the HF loading log to confirm the path. If the
   wrong file is loaded, the change has no effect.
2. Confirm `import os` and `import torch` are present at the top of the file
   (`os.environ.get` / `torch.compile` depend on them; a normal modeling file
   already has them).
3. Compiled is ON by default. To fall back: `export FUNAUDIO_CRQ_COMPILE=0`
   before starting the service.
4. Restart the service. The first iterations trigger `torch.compile` compilation
   (**warmup: the first few steps are slow**); the speedup pays off over a longer
   run. Verify the gain via the gpu_busy / launch counts in the log.

Example serve command (unchanged from the existing deployment):

```bash
export FUN_AUDIO_CHAT_HOME=/home/models/FunAudioLLM/Fun-Audio-Chat
export FUN_AUDIO_CHAT_SPK_INFO=/home/models/FunAudioLLM/Fun-Audio-Chat/utils/new_spk2info.pt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export FUNAUDIO_CRQ_COMPILE=0   # uncomment to fall back to eager
python3 -m vllm_omni.entrypoints.cli.main serve /home/models/FunAudioLLM/Fun-Audio-Chat-8B \
  --omni --port 8091 \
  --stage-configs-path .../fun_audio_chat_cur8_cg_asnc.yaml \
  --stage-init-timeout 1800 \
  --allowed-local-media-path /home/shared/shumai \
  --trust-remote-code
```

## Checks before applying / deploying

1. **The compiled path forces `output_attentions / output_hidden_states = None`**
   (for traceability). If anything in the vllm-omni → model call chain passes
   `output_attentions=True`, the compiled path **silently drops** the attentions
   output. Serving generally does not request attentions, but confirm no caller
   depends on it before deploying; if needed, use `FUNAUDIO_CRQ_COMPILE=0` to
   take the eager path.
2. `FUNAUDIO_CRQ_COMPILE` is read once at import; changing it at runtime has no
   effect — a restart is required.
3. The patch's a/b path is `funaudiochat/modeling_funaudiochat.py`, matching the
   relative path under `$FUN_AUDIO_CHAT_HOME`. If the base differs from the
   patch's `index 76fba82` hash and `git apply` errors, use `patch -p1` or
   `git apply --3way --reject` (the latter leaves conflicts as `.rej` for manual
   resolution).

## Why not PR this back to the upstream model repo

Per the current decision: do not PR back to `FunAudioLLM/Fun-Audio-Chat`; keep
it only as **patch + md** in this vllm-omni repo, and `patch -p1` manually on
the deploy host. If you later change your mind and want it upstream, this patch
can be used directly as the PR diff (the a/b paths already align with the model
repo).