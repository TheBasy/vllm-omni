# CRQ sidecar `torch.compile` 优化(`modeling_funaudiochat.py` 侧)

本目录下的 `crq_torch_compile.patch` 是对 **HF 模型仓库 `FunAudioLLM/Fun-Audio-Chat`** 里 `funaudiochat/modeling_funaudiochat.py` 的一份性能优化 diff,**不是 vllm-omni 运行时代码**。放在 vllm-omni 的 `funaudiochat/` 文件夹里只是为了版本控制和可追溯,让这份优化跟 vllm-omni 的 fun_audio_chat 集成在一起,不散落在模型仓库 / 部署机。

## 这优化做了什么

把 `FunAudioChatDecoder.crq_generate_forward` 里 CRQ sidecar 的 transformer step 用 `torch.compile(mode="default")` 跑(inductor 算子融合,**不是 cudagraph**):

- 实测收益:~18k → ~7k GPU launches/step;~60ms → ~36ms gpu_busy/step;api_enqueue 占比减半。缓解 Fun-Audio-Chat 8B sidecar 的 **stage-0 launch-bound** 瓶颈。
- 为什么不用 cudagraph(`reduce-overhead`):sidecar 的 `DynamicCache` 每步递增,cudagraph 静态缓冲在 Qwen3 的 `apply_rotary_pos_emb` 里覆盖崩溃。`default` 模式只做算子融合、不静态化,安全。
- compiled target 是 plain `@staticmethod`(`_crq_forward_fn(tower, input_embeds, past_key_values)`,传模块引用 + 张量),可被 torch trace;KV(`DynamicCache`)进出;**sampling 留 eager**(动态分布/multinomial 不可编译);input slicing + `input_matching` 也留 eager(shape-dynamic)。
- `_crq_compiled_step` / `_crq_force_eager` 是**类级**属性,`FUNAUDIO_CRQ_COMPILE` 在 **import 时读一次**;多实例 / 多线程共享同一 compiled fn(staticmethod + 传入 `tower` 引用),安全。
- 默认 **compiled ON**。应急回退:`export FUNAUDIO_CRQ_COMPILE=0` → eager(等价于原实现)。
- 重构:`crq_generate_forward` 拆成 `crq_transformer_step` / `crq_sample_token` / `crq_embed_token` 三个纯子操作(eager 路径 bit-exact);compiled 路径另加 `crq_transformer_step_compiled`。

## 跟 vllm-omni 的关系(重要)

- vllm-omni 侧**不需要任何改动**。`vllm_omni/model_executor/models/funaudiochat/` 是 vllm 集成 wrapper;这份 patch 改的是**模型仓库**的 `modeling_funaudiochat.py`,运行时由 serve 的 `--trust-remote-code` 从 `FUN_AUDIO_CHAT_HOME` 加载。
- 所以这份 patch **不在 vllm-omni 进程里生效**;必须 apply 到部署机的模型仓库 clone 里(见下),重启服务才生效。

## 部署(在部署机)

前置:`FUN_AUDIO_CHAT_HOME` 指向 `Fun-Audio-Chat` 模型仓库的 clone(含 `funaudiochat/modeling_funaudiochat.py` + `utils/`)。

```bash
cd "$FUN_AUDIO_CHAT_HOME"
# 推荐 patch -p1(对空白/base 漂移更宽松);git apply 也行
patch -p1 < <vllm-omni 仓>/vllm_omni/model_executor/models/funaudiochat/crq_torch_compile.patch
# 或: git apply --3way --reject crq_torch_compile.patch
```

然后:

1. **确认 trust-remote-code 加载的就是这份文件**——而非权重目录 `Fun-Audio-Chat-8B/` 里另一份 flat 的 `modeling_funaudiochat.py`(trust_remote_code 常见坑)。可在文件头临时加 `print(__file__)` 或看 HF 加载日志确认路径。加载点不对,改了也不生效。
2. 确认文件顶部已 `import os` 和 `import torch`(`os.environ.get` / `torch.compile` 依赖;正常 modeling 文件都有)。
3. 默认 compiled ON。要回退:`export FUNAUDIO_CRQ_COMPILE=0` 再起服务。
4. 重启服务。首次会触发 torch.compile 编译(**warmup 头几步慢**),长跑才回本;收益看日志的 gpu_busy / launch 计数。

启动命令参考(与现有部署一致):

```bash
export FUN_AUDIO_CHAT_HOME=/home/models/FunAudioLLM/Fun-Audio-Chat
export FUN_AUDIO_CHAT_SPK_INFO=/home/models/FunAudioLLM/Fun-Audio-Chat/utils/new_spk2info.pt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export FUNAUDIO_CRQ_COMPILE=0   # 取消注释即回退 eager
python3 -m vllm_omni.entrypoints.cli.main serve /home/models/FunAudioLLM/Fun-Audio-Chat-8B \
  --omni --port 8091 \
  --stage-configs-path .../fun_audio_chat_cur8_cg_asnc.yaml \
  --stage-init-timeout 1800 \
  --allowed-local-media-path /home/shared/shumai \
  --trust-remote-code
```

## 提交前 / apply 前核对点

1. **compiled 路径强制 `output_attentions / output_hidden_states = None`**(为可 trace)。若 vllm-omni → model 的调用链里有人传 `output_attentions=True`,compiled 路径会**静默丢掉** attentions 输出。serving 场景一般不需求,但部署前确认调用点不依赖它;需要时用 `FUNAUDIO_CRQ_COMPILE=0` 走 eager。
2. `FUNAUDIO_CRQ_COMPILE` 在 import 时读一次,运行中改 env 不生效,需重启。
3. patch 的 a/b 路径是 `funaudiochat/modeling_funaudiochat.py`,匹配 `$FUN_AUDIO_CHAT_HOME` 下的相对路径。若 base 与 patch 的 `index 76fba82` 哈希不一致导致 `git apply` 报错,用 `patch -p1` 或 `git apply --3way --reject`(后者会把冲突块留成 `.rej` 手动处理)。

## 为什么不直接提交回上游模型仓库

按当前决定:不 PR 回 `FunAudioLLM/Fun-Audio-Chat`,只在本 vllm-omni 仓内以 **patch + md** 留存,部署机手动 `patch -p1` apply。如以后改主意要回上游,这个 patch 可直接作为 PR 的 diff 使用(a/b 路径已对齐模型仓库)。