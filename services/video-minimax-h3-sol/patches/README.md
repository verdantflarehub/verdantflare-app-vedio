# Sol-H3 实验适配补丁

当前状态（2026-09-10）：四项补丁与 CPU 参数存储保留适配已用于历史 B01 双卡出片，人工质量门未通过；正式 runner 仍保留 4090 blocked 门禁。v0.1.1 构建时将全部补丁校验并封装为独立实验源码，不需要 ConfigMap 挂载。以下 2026-09-09 内容按阶段保留，部署证据缺项与最新结论以中央 H3-Sol 方案为准。

## 0001：CPU 权重逐目标 CUDA 融合

`0001-stage-lora-on-cuda.patch` 只修改官方 `h3_runtime/lora.py`，增加可选 `staging_device="cuda:<rank>"`。未提供该参数时保留原 CUDA 常驻行为；显式启用时要求所有 adapter 目标在 CPU 上，完成全量键名、形状、rank 等检查后逐目标执行：

1. 将一个目标权重复制到指定 CUDA 设备。
2. 按官方 dtype、alpha/rank/scale 执行同一 `addmm_`；混合 adapter 的 `.diff` 仍使用 FP32 累加。
3. 阻塞写回原 CPU Parameter，保留 Parameter 身份，释放本次 CUDA 副本。

不把矩阵乘法改到 CPU，不原地写回 safetensors 文件，不自动启用量化。任意运行时融合失败后都必须丢弃该模型实例，不能继续使用可能部分融合的状态。峰值取决于最大目标矩阵及 delta 临时副本，并非零显存。

`series.json` 记录上游 revision、补丁 SHA-256、目标文件 before/after SHA-256。主机需安装 `patch`；从应用仓库根目录执行：

```bash
python3 services/video-minimax-h3-sol/prepare-patched-source.py \
  --source /path/to/sealed/Sol-H3 \
  --output /path/to/new/experimental-Sol-H3
```

工具先核对全部官方来源，再复制到全新目录，以零 fuzz 应用补丁并验证全部文件；成功才生成 `patches.manifest.json`。拒绝覆盖已有目录。原 runner 的严格来源验证不会接受此实验目录，防止尚未完成的适配被直接用于推理。v0.1.0 镜像没有集成该补丁；原封存镜像已在第三轮发布用于测试，补丁由独立 ConfigMap 挂载。

## 2026-09-09 验证范围

- 组件构造样本：普通 PEFT LoRA、FastVideo v2 混合 delta，在两张独立 RTX4090 上分别逐位对比原始官方实现；4/4 通过。错误形状在修改权重前拒绝，Parameter 身份保持，默认 CPU 融合继续拒绝。
- 真实共享模型只读切片：adapter 完整 SHA-256 匹配锁文件；按 adapter 对应目标权重元素数选择最小与最大目标，在两张独立 RTX4090 上分别比较；4/4 逐位一致。
- 真实目标为 `token_refiner.refiner_blocks.0.attn.to_k`（7168×5376）和 `transformer_blocks.9.ff.net.0.proj`（28672×5376），均为 BF16；这里只证明这些切片，不代表全底座固定 revision 已验证。
- 测试镜像为已有 H3 v0.3.0，Torch 2.13.0+cu130。Sol 固定依赖 Torch 2.10.0+cu130 尚未进行同样 GPU 回归；该差异是后续接入门槛。
- 19 项原契约测试仍通过。完整 72 文件来源复制、补丁应用、输出封存与已有目录拒绝覆盖已实际检查。

GPU 测试脚本位于 `../gpu_tests/`。中央 Job 和不可变 ConfigMap 清单位于设计仓库的 `deploys/k8s.cn-chengdu.bc-cloud.com/verdantflare-video/sol-h3/`，证据位于同模块 `operations/2026-09-09-sol-lora-*.json`。这些组件样本不构成视频输入、生成结果或人工质量验收。

## 0002：AdaLN 逐块预计算后安装卸载钩子

`0002-stage-adaln-before-offload.patch` 为 `precompute` 增加显式 CUDA staging：全部 Transformer 参数须先在 CPU 上，仅短暂把时间嵌入模块以及当前 AdaLN 投影搬到 CUDA。保留官方每步、每条件 schedule 的 GEMM 形状和精度，逐块生成缓存并放回 CPU，释放原投影。

`enable_adaln_precompute` 增加每个 Transformer 实例独立的 staging 参数和一次性 `after_precompute` 回调。必须先替换全部 AdaLN 投影，再在回调中安装 Diffusers block-level group-offload hooks，避免钩子继续引用被删除的投影。默认参数保持原官方路径。失败后的模型实例必须丢弃，不对部分预计算状态继续推理。

2026-09-09 第三轮已改用封存 Sol 镜像的 **Torch 2.10.0+cu130 / 锁定 Diffusers**：

- 缩小维度的官方 H3 Block 栈，三种条件 schedule × 四步 × 两次重复 × 两张卡，共 48 次前向逐位相同；每次前向后参数和缓存均回到 CPU。
- Denoiser wrapper 的回调在缓存替换之后执行且只执行一次，后续调用复用缓存；非法设备、错误 schedule 和重复预计算拒绝检查通过。
- 从共享模型只读加载真实 FP32 `time_embedder` 与第 0/49 层 BF16 AdaLN 投影，两张卡上三种条件缓存均与官方 CUDA 常驻实现逐位一致。两个投影样本 CUDA 峰值由 1,134,358,528 降至 549,682,688 字节；不是整机峰值。
- 同一依赖环境重跑前述两类 LoRA 组件样本和真实切片，全部逐位一致。此前“锁定 Torch 未做 GPU 回归”的差距已在组件范围内补齐。

详细记录见设计仓库 `operations/2026-09-09-sol-adaln-tests.md`。当前实验副本没有接入 engine/runner，未验证 Sol 融合算子与 Ulysses 的完整组合、文本编码器卸载或 VAE 编解码生命周期。**不得将组件通过改写成双卡完整推理已就绪。**

## 0003：完整 Ref2VA CPU offload 接入（实验）

`0003-integrate-ref2va-cpu-offload.patch` 新增内部 `cpu_offload=True` 实验入口，仅允许 Ref2VA + Dense + 双 SM89 GPU。跳过整体 `pipe.to(cuda)`，连接前两项已验证的 staging 能力，并在 AdaLN 完成后安装 Transformer block offload。

Qwen3-VL 按 model/visual/language_model/embed_tokens/lm_head 递归安装分组卸载；embed_tokens 拥有独立钩子，覆盖其被父级 forward 之前直接调用的情况。视频 VAE 的 encoder、decoder、quant_conv、post_quant_conv 分别覆盖实际计算入口；不依赖 VAE.forward。较小的音频 VAE 保持 CUDA 驻留。视频 VAE 采用非批量、非全局批量、不开 compile 的 tile 路径，仍保留官方双卡分片。

当前此补丁仅完成来源封存、导入及 Qwen3-VL 实际配置下的模块结构核对；完整双卡加载正在通过专属 `load-test.yaml` 验证。不得把此描述标为出片通过。主 runner 仍保留原 4090 blocked 门禁。

整机加载测试的内存 request/limit 为 208 GiB、CPU 为 8；监控在容器内存超过 200 GiB 或主机 MemAvailable 低于 64 GiB 时终止测试进程组，最长 1200 秒。GPU 仍为两张整卡，NCCL 使用独立 2 GiB /dev/shm，包含在内存限额内。加载测试不提交生成请求。

2026-09-09 更新：第四补丁支持 768×1344 竖屏。完整 B01 两个真实 Ref2VA 单元已通过 GPU 生成与媒体解码，运行时还使用 retain_cpu_weights.py 的推理专用同步卸载参数存储保留适配；辅助脚本哈希、实际配置与资源见中央 B01 对比回执。S06 嘟嘴质量问题仍在，未标记生产就绪。
