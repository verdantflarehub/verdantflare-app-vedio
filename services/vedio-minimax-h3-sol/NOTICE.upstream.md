# Upstream and model notices

This service packages NVIDIA's Sol-H3 directly from NVlabs/Sana at
`2936c47637380842aaa4a4488fac5006cc542b70`, under
`models/minimax_h3/Sol-H3`. It does not package the LightX2V runtime.

The Sol-H3 project page identifies the code license as Apache-2.0 and links to
Sana's main-branch LICENSE. The pinned sol-engine tree does not contain a root
LICENSE. `LICENSE.upstream` preserves the linked Apache license at main commit
`cc99e6546f5d4164155838e73bc6f5636b2a2d5f`; its hash is recorded in
`upstream.lock.json`. Bundled Sol-Attn and FlashAttention notices remain in the
sealed upstream package. No third-party notices are removed.

MiniMax-H3 model weights retain the MiniMax-H3 Community License and applicable
model terms. The Ref2VA adapter comes from `lightx2v/Minimax-h3-Turbo` at the
revision in `models.lock.json`; code licensing does not relicense that adapter
or its base weights. Model terms must be accepted through the authorized model
account before downloads. This image contains no weights or generated media.

Diffusers, cuDNN frontend, PyTorch and other dependencies retain their licenses.
The image records installed packages separately from the upstream source lock.
