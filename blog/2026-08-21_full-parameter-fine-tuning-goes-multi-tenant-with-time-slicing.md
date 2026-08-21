---
title: "Full-parameter fine-tuning goes multi-tenant: slashing accelerator footprint by 67% with llm-d time-slicing"
description: "Demonstrates llm-d time-slicing in a real-world multi-tenant use case: running SFT and full-parameter RL fine-tuning across different base models using OpenRL and the time-slicing stack."
slug: multi-tenant-full-parameter-fine-tuning-time-slicing
date: 2026-09-01T10:00
authors:
  - aishu
  - sunilarora
tags: [blog, updates, llm-d, rl]
---

# Full-parameter fine-tuning goes multi-tenant: slashing accelerator footprint by 67% with llm-d time-slicing

When we [introduced co-operative time-slicing in llm-d](https://llm-d.ai/blog/rl-post-training-co-operative-time-slicing), we made a claim: if RL phases become schedulable units, independent jobs can share accelerators with near-zero waste. Today we're backing that claim with a measured, end-to-end proof.

[OpenRL](https://opensource.googleblog.com/2026/06/introducing-openrl-a-self-hosted-post-training-api-for-fine-tuning-llms.html), an open source, Kubernetes-native, Tinker-compatible, self-hosted fine-tuning service built on the llm-d time-slicing stack, runs supervised fine-tuning (SFT) and full-parameter reinforcement learning for multiple tenants concurrently on the same GPUs.

<!-- truncate -->

## Key Results at a Glance

By time-slicing three distinct workloads (two RL, one SFT) on different base models, we achieved:

- **67% reduction in hardware footprint:** Cut the required physical GPU count from 6 dedicated GPUs down to just 2 shared GPUs.
- **38% savings in GPU-hours:** Reduced total compute consumption from 2.10 GPU-hours to 1.30 GPU-hours.
- **40% more experiments in the same timeframe:** On the same two GPUs, the three workloads finished in 39 minutes time-sliced versus 54 minutes run one after another, at just three tenants.
- **Over 2x increase in GPU duty cycle:** Boosted average trainer GPU utilization from 15.6% to 34.2%, with ample headroom for more tenants before saturation.
- **Zero quality degradation:** All three workloads achieved identical convergence curves compared to their dedicated-GPU baselines.

This post walks through what the time-slicing platform provides, what a managed RL service (RLaaS) adds, and what the combination measures.

## Enterprise RL & Multi-Tenancy

A post-training API is multi-tenant by definition: one team fine-tunes a customer support assistant on internal tickets, another trains a text-to-SQL agent on proprietary database schemas.

For parameter efficient methods like LoRA, engines like vLLM and PyTorch already support many tenants over one frozen base model. However, full-parameter fine-tuning (FFT) breaks this paradigm. Because every training step can mutate all weights, each tenant requires the whole model, whole optimizer state, and whole GPU.

This has forced enterprises into making expensive choices:

1. Siloed, dedicated GPUs: allocating dedicated GPU capacity to every single tenant. Due to the nature of RL runs these dedicated GPUs may sit idle (30-74%) as RL alternates between generation and training.
2. Queuing : forcing teams to wait in line, slowing the development velocity and time to market.

Llm-d time-slicing eliminates this trade-off by enabling full-parameter customization at a fraction of the hardware cost.

## Case study: OpenRL

The [llm-d time-slicing stack](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) supplies the machinery — the Snapshot Agent that snapshots and restores a worker's state between VRAM and host memory, the TimeSlice Orchestrator that grants tenants exclusive access in turn, and a client library that wraps two RPCs, `acquire()` and `yield()`. What a managed service must add is exactly one thing: knowing where its tenants' phase boundaries are. That turns out to be the easy part.

The API *is* the phase structure. OpenRL exposes [Tinker-style primitives](https://github.com/gke-labs/open-rl) — `generate_samples` for rollouts, `forward_backward` for training, and `optim_step` for the weight update. Every tenant's job arrives pre-decomposed into the schedulable units time-slicing was built around. The service wraps `acquire()` and `yield()` around each work unit, users write an ordinary training loop and get time-sliced multi-tenancy without knowing it exists.

The integration comes down to two moves:

1. **Oversubscribe the hardware.** The OpenRL orchestrator provisions dedicated trainer and sampler workers per tenant and binds multiple tenants' workers to the same physical GPUs using [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/). With per-tenant worker processes, tenants can bring different stacks — PyTorch FSDP or Megatron, vLLM or SGLang — to the same shared hardware.
2. **Wrap GPU work units in acquire/yield.** Each worker loop pops a work unit from its tenant queue, calls `acquire()`, runs the work, and calls `yield()`. The orchestrator and Snapshot Agent handle the rest — granting exclusive access in turn and context-switching tenant states. Work that doesn't need the GPU, like persisting checkpoints, runs from the host copy and never takes the lock.

## What we measured

To validate the architecture, we ran a highly heterogeneous, concurrent workload representing typical enterprise workloads:

- Tenant A (Text-to-SQL RL): Fine-tuning Qwen3-1.7B on the Spider dataset.
- Tenant B (Math RL): Fine-tuning Qwen2.5-7B on GSM8K.
- Tenant C (Dialogue SFT): Supervised fine-tuning of Gemma 4 E2B on MultiWOZ.

All three jobs were multiplexed onto a shared pool of two NVIDIA H100 GPUs (one for training, one for sampling).

<div style="text-align:center; margin:20px 0">
  <iframe src="/img/blogs/openrl-time-slicing/blog-interleaving.html" title="GPU occupancy animation" scrolling="no" style="width:100%; height:260px; border:0"></iframe>
</div>

**All three tenants converged — matching baseline learning curves.** The text-to-SQL tenant climbed from 25% to 43% in response accuracy, the math RL tenant roughly doubled its eval accuracy, and the dialogue SFT tenant's eval loss fell from 4.77 to 0.56 over 80 steps — a curve point-for-point identical to its baseline run. Overall, time-slicing changed where the jobs ran — not what they learned.

<div style="text-align:center; margin:20px 0">
  <img src="/img/blogs/openrl-time-slicing/convergence.webp" alt="Convergence: each tenant's time-sliced curve over its dedicated-GPU baseline" style="width:100%; height:auto" />
</div>

**Higher resource efficiency.** With one dedicated GPU per worker, these three tenants would hold six GPUs — a trainer and sampler each. Given the inherent dependency between trainer and sampler in synchronous RL, those GPUs sit largely idle — gaps the job itself cannot fill. Time-slicing fills them with other tenants' work: the same jobs run on two GPUs, lifting duty cycles roughly 2x compared with their dedicated-GPU runs.

<div style="text-align:center; margin:20px 0">
  <img src="/img/blogs/openrl-time-slicing/duty-cycle.webp" alt="Duty cycle: six dedicated baseline GPUs vs two time-sliced GPUs" style="width:100%; height:auto" />
</div>

**Minimal system overhead.** Context switches cost 0.5–1.6 seconds (median). Queue waits per phase run 1–15 seconds (median) — highest for the SFT tenant, whose short steps queue behind the RL tenants' longer phases.

<div style="text-align:center; margin:20px 0">
  <img src="/img/blogs/openrl-time-slicing/switch-wait.webp" alt="Context-switch time and queue wait per phase" style="width:100%; height:auto" />
</div>

The full comparison:

| | GPUs provisioned | Workload run time | GPU-hours |
|---|---|---|---|
| Dedicated, all concurrent | 6 | 21 min | 2.10 |
| Shared, jobs run serially | 2 | 54 min | 1.80 |
| **Time-sliced** | **2** | **39 min** | **1.30** |

*Small demonstration runs like these are time-slicing's hardest case: at production scale, phases run for minutes, switch costs amortize toward zero, and the reclaimable idle only grows.*

- **Versus six dedicated GPUs** — time-slicing provisions two-thirds fewer GPUs and pays for ~38% less GPU time.
- **Versus two GPUs running jobs serially** — identical hardware, but time-slicing finishes the workload ~28% sooner, as it reclaims idle windows.

The trade is per-tenant wall-clock time in favor of accelerator efficiency: an individual job runs 1.9–2.5x longer than it would alone, and the three workloads collectively ran only 1.9x slower even though the GPU count fell 3x; the difference is the reclaimed idle time.

## What's next

We're rolling out a selective state offload backend on Snapshot Agent that allows snapshotting specific GPU memory regions — LoRA adapter weights, accumulated gradients, optimizer state — while the shared base model stays resident. One interface now does it all: whole-process parking for full-parameter tenants, region-level parking for LoRA tenants. In early testing, offloading LoRA adapters through this backend delivered a 2.8x tenant-density gain on a single NVIDIA L4 trainer GPU.

**Get Started Today**

If you are building or scaling a multi-tenant post-training platform, you can start getting efficiency gains today:

- Deploy the [time-slicing stack](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) today
- Run [OpenRL](https://github.com/gke-labs/open-rl) and get this integration out of the box.

**Acknowledgements**

Thanks to [Edwin Hernandez](https://github.com/Edwinhr716), [Jessica Chen](https://github.com/jessicaochen), [Ling Lin](https://github.com/lynnl0927) and [Shuby Mishra](https://github.com/ShubyM) for bringing this to life.
