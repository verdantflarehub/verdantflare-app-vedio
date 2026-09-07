# VerdantFlare App Vedio

VerdantFlare App Vedio 是部署在 VerdantFlare Station 上的视频生成应用套件。仓库名及工程资源沿用 `vedio` 拼写；面向用户的领域名称、MCP 工具和媒体类型使用标准的 `video` 拼写。

## 项目职责

本仓库负责 MiniMax H3 Ref2VA 运行时、验证基准和 Video MCP：

| 服务 | 职责 |
| --- | --- |
| `vedio-mcp-server` | 导入项目素材、提交视频任务、查询状态并持久化视频 Artifact。 |
| `vedio-minimax-h3-api` | 提供 MiniMax H3 Base Ref2VA 异步视频生成 API。 |
| `vedio-minimax-h3-sglang-benchmark` | 验证 SGLang、Full INT8 和 RTX 4090 运行配置。 |
| `vedio-minimax-h3-lightx2v-benchmark` | 验证 LightX2V Turbo Ref2VA 配置。 |

Video MCP 暴露以下 v1 工具：

```text
artifact.import
video.generate
video.status
video.result
```

MCP 只返回项目范围的任务 ID、Artifact ID、媒体元数据和下载地址，不返回运行时任务 ID、集群内部地址、节点、GPU 或宿主机路径。模型、输入素材和生成视频均保存在持久化存储中，不进入镜像或 Git。

## 仓库结构

```text
.
├── .github/workflows/
├── deploy/chengdu.beagle/verdantflare-vedio/
└── services/
    ├── vedio-mcp-server/
    ├── vedio-minimax-h3-api/
    ├── vedio-minimax-h3-sglang-benchmark/
    └── vedio-minimax-h3-lightx2v-benchmark/
```

工程、镜像、Workflow 和 Kubernetes 资源统一使用 `vedio-` 前缀。当前版本镜像为：

| 服务 | 镜像 tag |
| --- | --- |
| `vedio-mcp-server` | `vedio-mcp-server-v0.1.4` |
| `vedio-minimax-h3-api` | `vedio-minimax-h3-api-v0.3.0` |
| `vedio-minimax-h3-sglang-benchmark` | `vedio-minimax-h3-sglang-benchmark-v0.1.7` |
| `vedio-minimax-h3-lightx2v-benchmark` | `vedio-minimax-h3-lightx2v-benchmark-v0.1.1` |

版本标签不可覆盖。发布流水线仅在 `release` 分支触发，生产部署不得使用 `latest` 或 SHA 标签。

迁移引导基础镜像 `vedio-minimax-h3-api-v0.1.1` 从已验证的原 H3 基础镜像原样复制，其 linux/amd64 manifest digest 为 `sha256:8970ad335fe07638d39ab75e1b623d2a68d67b8f04efa083806cc6178adf9855`；后续 H3 构建链只引用 `vedio-` 名称。

## 成都验证环境

声明式资源位于 `deploy/chengdu.beagle/verdantflare-vedio/`，使用独立 namespace `verdantflare-vedio`。H3 Runtime 固定调度到 `10.241.109.6`，由 `hami-scheduler` 申请两张不同的完整 RTX 4090，并以 TP2 执行单任务。迁移期间，模型通过 Retain、只读静态 PV 复用节点上已经验证的 H3 模型目录；新的视频项目数据使用独立的 `hostpath` PVC。旧 namespace 的 PVC 和工作负载必须保留到新服务完成真实验收并确认清理目标之后；删除旧模型 PVC 前还必须先把它所绑定 PV 的回收策略改为 `Retain`，否则旧 PV 的 `Delete` 策略会删除共享模型目录。

部署前必须重新确认节点 Ready、完整 GPU、驱动、`hostpath` StorageClass、`/data` 容量、所需镜像以及 `vedio-mcp-auth` Secret。不得输出 Secret 内容。公网入口为：

```text
POST https://mcp.cn-chengdu.bc-cloud.com/video
```

客户端从安全环境注入 Token 后注册：

```bash
codex mcp add verdantflare-video \
  --url https://mcp.cn-chengdu.bc-cloud.com/video \
  --bearer-token-env-var VIDEO_MCP_BEARER_TOKEN
```

公网只路由 MCP 和受保护的 Artifact 下载，不发布 `/health` 或 `/runtime-artifacts/`。流水线、镜像、滚动部署、真实 GPU 推理和人工视频审核全部完成后，才能将版本标记为已发布并通过验收。

MiniMax H3 不是 Apache/MIT 模型。向第三方开放前必须完成模型许可证要求的地域、用户条款、内容治理、报告、披露和界面归因检查。
