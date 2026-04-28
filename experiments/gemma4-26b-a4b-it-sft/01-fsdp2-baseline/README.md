# Experiment 01: fsdp2-naive-baseline

## 元信息

- **模型**：Gemma4-26B-A4B-it（VLM，文本-only SFT）
- **日期**：2026-04-28
- **状态**：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁
- **硬件**：8×H100-80GB（NVLink）
- **框架**：FSDP2 + ms-swift / transformers

## 目标

让 Gemma4-26B-A4B-it 在 swift-ms sft FSDP2 上跑起来

## 原始启动参数
``` Bash
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
NPROC_PER_NODE=8 swift sft \
    --model google/gemma-4-26B-A4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/megatron-sft-recipes/sft-data/train.jsonl \
    --max_length 16384 --truncation_strategy right \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_steps 5 --logging_steps 1 --save_strategy no \
    --tuner_type full --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit true --freeze_aligner true \
    --learning_rate 2e-5 --warmup_ratio 0.05 \
    --use_liger_kernel true \
    --fsdp '{\"fsdp\": \"full_shard auto_wrap\", \"fsdp_config\": {\"fsdp_version\": 2, \"reshard_after_forward\": true, \"auto_wrap_policy\": \"TRANSFORMER_BASED_WRAP\", \"cpu_ram_efficient_loading\": true, \"state_dict_type\": \"SHARDED_STATE_DICT\", \"activation_checkpointing\": true}}' \
    --sequence_parallel_size 2 \
    --output_dir /tmp/q1_run
"
```

```Bash
docker exec fsdp_sft bash -lc 'cat > /tmp/fsdp_q1.json <<EOF
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"],
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": true
    }
}
EOF'

docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
NPROC_PER_NODE=8 swift sft \
    --model google/gemma-4-26B-A4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/megatron-sft-recipes/sft-data/train.jsonl \
    --max_length 16384 --truncation_strategy right \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_steps 5 --logging_steps 1 --save_strategy no \
    --tuner_type full --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit true --freeze_aligner true \
    --learning_rate 2e-5 --warmup_ratio 0.05 \
    --use_liger_kernel true \
    --fsdp /tmp/fsdp_q1.json \
    --sequence_parallel_size 2 \
    --output_dir /tmp/q1_run
"
```

## 问题

### 1. Could not find the transformer layer class Gemma4AudioLayer in the model
```Bash
ValueError: Could not find the transformer layer class Gemma4AudioLayer in the model.
File ".../accelerate/utils/dataclasses.py", line 2059, in __post_init__
```

`accelerate` 在找 `Gemma4AudioLayer` class，但 model 里没有。`TRANSFORMER_BASED_WRAP` 默认从 `model._no_split_modules` 拿 layer 名字列表。

> [transformers/src/transformers/models/gemma4/modeling_gemma4.py#L1443](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py#L1443)
```Python
@auto_docstring
class Gemma4PreTrainedModel(PreTrainedModel):
    ...
    _no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    ...
```

会看到列表里硬编码了 `["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]` ——这是 transformers 给整个 gemma4 family 用的"超集"。但 `gemma-4-26B-A4B` 实例里**没有装 audio encoder**（gemma4 family 里只有 E2B/E4B 边缘多模态版才带 audio；A4B 是 MoE text+vision 版）。

通过下列方式验证：
```Bash
# 看权重文件：扫 safetensors index 有没有 audio tower 权重
docker exec fsdp_sft python3 -c "
import json, glob
idx = glob.glob('/root/.cache/modelscope/models/google/gemma-4-26B-A4B-it/*.safetensors.index.json')[0]
keys = json.load(open(idx))['weight_map'].keys()
print('vision keys:', sum('vision' in k for k in keys))
print('audio  keys:', sum('audio'  in k for k in keys))
print('text   keys:', sum('language_model' in k or 'text_model' in k for k in keys))
"
# 期望：audio keys: 0 / vision keys: > 0

# 实测
vision keys: 356
audio  keys: 0
text   keys: 657
```

需要在 fsdp config 里加显式列表覆盖默认。key 名是什么？这里有个坑——accelerate 文档和 transformers 用的 key 名不一样，写错就静默失效。2条找法：

**(a)顺着 stack trace 读 accelerate 源码**：[accelerate/src/accelerate/utils
/dataclasses.py#L1787](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L1787C5-L1787C64)
```Python
@dataclass
class FullyShardedDataParallelPlugin:
    ...
    transformer_cls_names_to_wrap: Optional[list[str]] = field(...)
    ...
    def __post_init__(self):
        ...
        for layer_class in self.transformer_cls_names_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
```
-> accelerate 内部字段名是 `transformer_cls_names_to_wrap`。

**(b)但真正传的不是 accelerate 的字段**：swift sft 走的是 transformers TrainingArguments -> 再 mapping 到 accelerate。看 transformers 接受哪个 JSON key：
```Python
@dataclass
class TrainingArguments:
    ...
    if isinstance(self.fsdp_config.get("transformer_layer_cls_to_wrap", None), str):
        self.fsdp_config["transformer_layer_cls_to_wrap"] = [self.fsdp_config["transformer_layer_cls_to_wrap"]]
    ..
```
-> transformers 内部字段名是 `transformer_layer_cls_to_wrap`

在 `fsdp_config` JSON里加：
```JSON
"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
```

这里我们可以优化一下，把fsdp参数改写成JSON文件，在命令里引用路径的方式执行，这样就可以不再和转义字符打交道了：
```Bash
# 1. fsdp_config 写到容器里的 /tmp 文件（外层单引号，JSON 内部不用转义）
docker exec fsdp_sft bash -lc 'cat > /tmp/fsdp_q1.json <<EOF
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"],
        "cpu_ram_efficient_loading": true,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": true
    }
}
EOF'

# 2. 验证 JSON 语法（pipe 到 json.tool 出错即坏）
docker exec fsdp_sft cat /tmp/fsdp_q1.json | python3 -m json.tool

# 3. swift sft 命令里把 --fsdp '{...}' 换成 --fsdp /tmp/fsdp_q1.json
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
NPROC_PER_NODE=8 swift sft \
    --model google/gemma-4-26B-A4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/megatron-sft-recipes/sft-data/train.jsonl \
    --max_length 16384 --truncation_strategy delete \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_steps 5 --logging_steps 1 --save_strategy no \
    --tuner_type full --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit true --freeze_aligner true \
    --learning_rate 2e-5 --warmup_ratio 0.05 \
    --use_liger_kernel true \
    --fsdp /tmp/fsdp_q1.json \
    --sequence_parallel_size 2 \
    --output_dir /tmp/q1_run
"
```

### 2. Tensor has no attribute device_mesh
```Bash
AttributeError: 'Tensor' has no attribute 'device_mesh'
File ".../accelerate/utils/fsdp_utils.py", line 537, in <listcomp>
    sharded_state[k] = DTensor.from_local(v, device_mesh=v.device_mesh, ...)
```
这里解释一下几个概念：

**(a) DTensor vs 普通 Tensor 是什么区别**

PyTorch 里有两种 tensor:
```
torch.Tensor       # 普通 tensor: 有 .data, .dtype, .device
DTensor (distributed tensor)   # 分布式 tensor: 多了 .device_mesh, .placements
                               # 知道自己被分片到了哪些 GPU、怎么分的
```
device_mesh 是 DTensor 独有的属性，记录"我这个 tensor 分布在 GPU [0,1,2,...,7] 这个 mesh 上"。普通 Tensor 没有这个属性——你访问 tensor.device_mesh 就会 AttributeError。

**(b) FSDP2 wrap 之后哪些 param 变 DTensor、哪些不变**

FSDP2 不会自动把整个模型里所有 param 都变 DTensor。它只对"被 auto_wrap 命中的子模块"做这件事。

上面我配的 `transformer_layer_cls_to_wrap = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]`。FSDP2 wrap 之后的model 长这样：
```
Gemma4ForConditionalGeneration  (root，未 wrap)
├── language_model
│   ├── embed_tokens.weight        ← 普通 Tensor (root-level，未 wrap)
│   ├── layers.0  [FSDP wrapped]
│   │   ├── self_attn.q_proj.weight ← DTensor ✓ (在 wrap 单元里)
│   │   ├── self_attn.k_proj.weight ← DTensor ✓
│   │   └── ...
│   ├── layers.1  [FSDP wrapped]
│   │   └── ...                     ← 都是 DTensor ✓
│   ├── ...
│   └── norm.weight                 ← 普通 Tensor (root-level，未 wrap)
├── vision_tower
│   ├── ... (vision encoder layers FSDP wrapped → DTensor)
│   └── post_layernorm.weight       ← 普通 Tensor (vision encoder root，未 wrap)
├── multi_modal_projector.weight    ← 普通 Tensor (root-level VLM aligner，未 wrap)
└── lm_head.weight                  ← 普通 Tensor (root-level，未 wrap)
```
核心事实：被 wrap 进 transformer layer class 的 param → DTensor；root-level 散件 param → 还是普通 Tensor。

为啥 root 这些不被 wrap？因为 `transformer_layer_cls_to_wrap` 你只列了"transformer layer class"，没列 `nn.Embedding` / `nn.Linear` 这种通用 class——而 `embed_tokens` / `lm_head` / `multi_modal_projector` 是这些通用 class，不在你的 wrap 列表里。

> 这是 FSDP2 设计取舍：你也可以写 `auto_wrap_policy: SIZE_BASED_WRAP` 让所有大于 N params 的 module 都 wrap，那样 root params 也会变 DTensor。但 `TRANSFORMER_BASED_WRAP` 只盯 transformer layer class——这是大部分 LLM 训练的 best practice（per-layer wrap 平衡 memory 和 communication）。

**(c) `cpu_ram_efficient_loading` 控制两条加载路径**

- 路径A：`cpu_ram_efficient_loading=False`（默认/朴素）
    ```
    Rank 0:  CPU 上 from_pretrained() → 51.6 GB 完整模型 → 移到 GPU → FSDP2 wrap
    Rank 1:  CPU 上 from_pretrained() → 51.6 GB 完整模型 → 移到 GPU → FSDP2 wrap
    Rank 2:  ...同上...
    ...
    Rank 7:  ...同上...
    ```

- 路径B：`cpu_ram_efficient_loading=True`（"省RAM"）
    ```Python
    Rank 0:  CPU 上 from_pretrained() → 51.6 GB 完整模型
    Rank 1:  CPU 上建空架子 (meta tensors，无 storage)
    Rank 2:  ...meta tensors...
    ...
    Rank 7:  ...meta tensors...

    → FSDP2 wrap 结构建立（每 rank 都建好 wrap topology）

    → accelerate 调 fsdp2_load_full_state_dict:
    for param_name, sharded_param in meta_sharded_sd.items():
        device_mesh = sharded_param.device_mesh         # <- 报错
    ```

--- 

阅读源码，看到一个类似 fsdp2_load_full_state_dict 的函数，循环 state_dict 把每个 v 转成 DTensor：
> [accelerate/src/accelerate/utils/fsdp_utils.py#L537](https://github.com/huggingface/accelerate/blob/v1.13.0-release/src/accelerate/utils/fsdp_utils.py#L537)
```Python
def fsdp2_load_full_state_dict(accelerator, model, ...):
    ...
    for param_name, sharded_param in meta_sharded_sd.items():
        device_mesh = sharded_param.device_mesh                  #L537
        ...
```

我们看看`fsdp2_load_full_state_dict`这个函数在哪里被调用了:
```Bash
docker exec fsdp_sft grep -rn 'fsdp2_load_full_state_dict' \
    /usr/local/lib/python3.12/site-packages/accelerate/
```

```Bash
/usr/local/lib/python3.12/site-packages/accelerate/
/usr/local/lib/python3.12/site-packages/accelerate/utils/__init__.py:231:    fsdp2_load_full_state_dict,
grep: /usr/local/lib/python3.12/site-packages/accelerate/utils/__pycache__/fsdp_utils.cpython-312.pyc: 匹配到二进制文件
grep: /usr/local/lib/python3.12/site-packages/accelerate/utils/__pycache__/__init__.cpython-312.pyc: 匹配到二进制文件
/usr/local/lib/python3.12/site-packages/accelerate/utils/fsdp_utils.py:467:def fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict, cpu_offload: bool = False):
/usr/local/lib/python3.12/site-packages/accelerate/utils/fsdp_utils.py:673:        
# Afterwards, when we call `fsdp2_load_full_state_dict`, us creating the state_dict would result into briefly having two copies of model state_dict on the GPU -> VRAM spike
/usr/local/lib/python3.12/site-packages/accelerate/utils/fsdp_utils.py:705:        fsdp2_load_full_state_dict(
```

在源码中找到这一行:
> [accelerate/src/accelerate/utils/fsdp_utils.py#L705](https://github.com/huggingface/accelerate/blob/v1.13.0-release/src/accelerate/utils/fsdp_utils.py#L705)
```Python
def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    ...
    if fsdp2_plugin.cpu_ram_efficient_loading:
        ...
        from torch.distributed.fsdp import CPUOffloadPolicy

        fsdp2_load_full_state_dict(
            accelerator, model, original_sd, cpu_offload=isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy)
        )
```
很明显因为设置了 `cpu_ram_efficient_loading: true` 导致进入分支代码，造成报错。解决方案：

把 `cpu_ram_efficient_loading: true` -> false

true 时 rank 0 在 CPU 上装满 model 然后 broadcast；root params 没被 shard 的话广播完后还是 plain Tensor → accelerate 把 root 都当 DTensor 处理就崩。

代价：每 rank 自己 load 51.6 GB 模型（主机 RAM 峰值 ≈ 400 GB），加载 ~30 sec → ~90 sec。

### 3. swift SP 跟 transformers 5.5.x mask API mismatch 

forward 第一步建 mask 时崩溃：
```Bash
File ".../transformers/models/gemma4/modeling_gemma4.py", line 2063, in create_causal_mask_mapping
    "full_attention": create_causal_mask(**mask_kwargs),
File ".../transformers/masking_utils.py", line 983, in create_causal_mask
    causal_mask = mask_interface(...)
TypeError: SequenceParallel._prepare_flash_attn.<locals>.flash_attention_mask()
    missing 1 required positional argument: 'cache_position'
```

对应的源码如下：

ms-swift：[ms-swift/swift/sequence_parallel/ulysses.py#L185](https://github.com/modelscope/ms-swift/blob/main/swift/sequence_parallel/ulysses.py#L185)
```Python
class SequenceParallel:
    ...
    def _prepare_flash_attn(self, base_model: torch.nn.Module):
        try:
            from transformers import masking_utils

            _origin_flash_attention_mask = masking_utils.flash_attention_mask

            def flash_attention_mask(*args, **kwargs):
                if self.world_size == 1:
                    return _origin_flash_attention_mask(*args, **kwargs)
                attention_mask = kwargs.get('attention_mask')
                if attention_mask is not None:
                    if attention_mask.all():
                        attention_mask = None

                return attention_mask
    ...
```

transformers：[transformers/src/transformers/masking_utils.py#L983C5-L997C23](https://github.com/huggingface/transformers/blob/v5.5-release/src/transformers/masking_utils.py#L983C5-L997C23)
```Python
@deprecate_kwarg("input_embeds", version="5.6.0", new_name="inputs_embeds")
def create_causal_mask( config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None = None, 
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
):
    ...
    # v5.1.x
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype, 
        config=config, 
        use_vmap=use_vmap,
    )

    # v5.5.x
    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype, 
        config=config,  
        use_vmap=use_vmap,
        device=device,
    )

    return causal_mask
```

很明显 `transformers v5.5.x`中，`mask_interface()` 的参数 `cache_position` 改为了 `q_length`和 `q_offset`。所以我们也得在 ms-swift 中相应位置进行兼容：

```Python
# ms-swift/swift/sequence_parallel/ulysses.py
def _prepare_flash_attn(self, base_model: torch.nn.Module):
    try:
        from transformers import masking_utils

        _origin_flash_attention_mask = masking_utils.flash_attention_mask

        def flash_attention_mask(
            batch_size,
            q_length,
            kv_length,
            q_offset,
            kv_offset,
            mask_function = masking_utils.causal_mask_function,
            attention_mask = None,
            **kwargs
        ):
            if self.world_size == 1:
                return _origin_flash_attention_mask(
                    batch_size = batch_size,
                    q_length = q_length,
                    kv_length = kv_length,
                    q_offset = q_offset,
                    kv_offset = kv_offset,
                    mask_function = mask_function,
                    attention_mask = attention_mask,
                    **kwargs
                )
            if attention_mask is not None and attention_mask.all(): 
                attention_mask = None
            return attention_mask
```

### 4. FA2 不支持 `head_dim=512`
```Bash
RuntimeError: FlashAttention forward only supports head dimension at most 256
File "/usr/local/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py", line 91, in _flash_attn_forward
    out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
                                           ^^^^^^^^^^^^^^^^^^
```

transformers：[transformers/src/transformers/models/gemma4/configuration_gemma4.py#L151](https://github.com/huggingface/transformers/blob/v5.5-release/src/transformers/models/gemma4/configuration_gemma4.py#L151)
```Python
@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4TextConfig(PreTrainedConfig):
    ...
    head_dim: int = 256
    ...
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
```
从 configuration 可以看出：
- `head_dim`（默认 256）→ 用在 **sliding_attention** 层
- `global_head_dim`（默认 512）→ 用在 **full_attention**（global）层
- `num_global_key_value_heads`（默认与 `num_key_value_heads` 相同）→ global 层的 KV 头数可以独立设

而截至 2026-04-28，FA2 max=256, FA3 max=256，所以 `--attn_impl flash_attention_2` 全用 FA2 → global 层炸。

我们可以修改 `transformers` 中 Gemma4 这部分代码，跳过识别到 `head_dim` > 256 就 fallback 到 sdpa

> [transformers/src/transformers/models/gemma4/modeling_gemma4.py#L1228](https://github.com/huggingface/transformers/blob/v5.5-release/src/transformers/models/gemma4/modeling_gemma4.py#L1228)

```Python
@use_kernelized_func(apply_rotary_pos_emb)
class Gemma4TextAttention(nn.Module):
    ...
    def forward(self, ...) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...
        # 源码
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 调整
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            FA_IMPLS = {"flash_attention_2", "flash_attention_3", "flash_attention_3_kernels"}

            _impl = self.config._attn_implementation
            if _impl in FA_IMPLS and self.head_dim > 256:
                _impl = "sdpa"
            
            attention_interface = ALL_ATTENTION_FUNCTIONS[_impl]
```

但是这种做法，想要用 transformers 新的改动，可能会存在冲突，需手动解决 rebase 冲突，维护不友好。这里我选择用打 patch 的方式适配:

```Python
# ms-swift/swift/utils/patches/gemma4_fa_patch.py
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_patched_gemma4_fa_attention"

def _fa_supports_head_dim_512() -> bool:
    """
    Check if the flash attention library supports head dimension 512.
    """
    try:
        import flash_attn 
        from packaging.version import Version
    except Exception:
        return False

    fa_512_min_version = os.environ.get("FA_HEAD_DIM_512_MIN_VERSION", "99.99.99")
    try:
        return Version(flash_attn.__version__) >= Version(fa_512_min_version)
    except Exception:
        return False

def apply(force: bool = False) -> bool:
    try:
        import transformers
        from transformers.models.gemma4 import modeling_gemma4
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
            Gemma4TextAttention,
        )

    except Exception as e:
        logger.warning(f"[gemma4_fa_patch] transformers/Gemma4 not importable, skip: {e}")
        return False

    if getattr(Gemma4TextAttention, _PATCH_FLAG, False):
        return True

    if not force and _fa_supports_head_dim_512():
        logger.info("[gemma4_fa_patch] flash_attn already supports head_dim=512, skip patch")
        return False

    Gemma4ForConditionalGeneration._support_flash_attn = True

    _origin_forward = Gemma4TextAttention.forward

    def _patched_forward(self, *args, **kwargs):
        _impl = getattr(self.config, "_attn_implementation", None)
        if (
            _impl in ("flash_attention_2", "flash_attention_3") and
            getattr(self, "layer_type", None) == "full_attention"
        ):
            self.config._attn_implementation = "sdpa"
            try:
                return _origin_forward(self, *args, **kwargs)
            finally:
                self.config._attn_implementation = _impl
        return _origin_forward(self, *args, **kwargs)

    Gemma4TextAttention.forward = _patched_forward
    setattr(Gemma4TextAttention, _PATCH_FLAG, True)

    logger.info(
        f"[gemma4_fa_patch] applied (transformers={transformers.__version__}); "
        f"full_attention layers will fall back to SDPA when FA2/FA3 is requested"
    )
    return True


if os.environ.get("GEMMA4_FA_PATCH_AUTO", "0") == "1":
    apply()
```

在 `ms-swift/swift/model/models/gemma.py` 运用一下
```Python
class Gemma4Loader(ModelLoader):
    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Gemma4ForConditionalGeneration

        # apply patch
        from swift.utils.patches.gemma4_fa_patch import apply as _patch_gemma4_attention
        _patch_gemma4_attention()

        self.auto_model_cls = self.auto_model_cls or Gemma4ForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)
```

再通过 `docker cp` 拷进容器 + editable 安装即可
```Bash
docker cp /home/ubuntu/fyh/ms-swift fsdp_sft:/opt/ms-swift
docker exec fsdp_sft bash -lc "pip install -e /opt/ms-swift --no-deps"
```