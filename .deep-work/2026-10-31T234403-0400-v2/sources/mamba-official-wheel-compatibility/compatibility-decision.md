# Official Mamba-2: bounded compatibility check

**Concrete candidate: official Mamba v2.3.1, keeping installed Torch and Triton.**
This worker did not install, SSH, import downloaded modules, train, or change
the sandbox. Original reference SSM remains explicitly a reference adaptation.

## Exact publisher asset

The [official v2.3.1 assets](https://github.com/state-spaces/mamba/releases/expanded_assets/v2.3.1)
include `mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`,
533,333,116 bytes, publisher SHA256
`f9e375b47cee1ef27463c9249073f9065f0f8f0f98b32fabfbb8df4bf75e2b66`.
The release API observations are saved in `release-asset-observations.json`.
The inspected v2.2.6.post2/post3 and v2.3.0 assets have no exact build for this
combination. Their Torch2.7 or NVIDIA-monthly builds are not substitutes for
the installed Torch2.8 ABI. v2.3.2 also has an asset misleadingly named2.3.1;
its hash differs. Use the **v2.3.1 release URL**, not filename alone.

[Direct selected asset](https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl).
Root will verify downloaded bytes, wheel metadata and source against the
publisher; this worker verified asset existence/metadata, not wheel import.

## Dependencies and full import chain

The checked [v2.3.1 setup](https://github.com/state-spaces/mamba/blob/v2.3.1/setup.py#L363-L370)
declares Torch, packaging, ninja, einops, Triton and Transformers. It does not
require TileLang, TVM, Quack or Triton>=3.5. The tag sources are archived
without execution under `v2.3.1/`, with hashes/import ASTs in the receipts.

The unmodified `mamba_ssm.__init__` imports the selective-scan interface,
Mamba1, Mamba2 and MambaLMHeadModel. Therefore even
`from mamba_ssm.modules.mamba2 import Mamba2` first loads the package initializer.
It requires the matching compiled `selective_scan_cuda` extension from this
wheel, Triton, einops, and the complete model/generation import chain.
Mamba2 imports `huggingface_hub.PyTorchModelHubMixin`; generation imports
`transformers.generation.GenerateDecoderOnlyOutput` and `TextStreamer`;
HF utilities import `transformers.utils` and its `cached_file` helper.
Do not delete initializer imports to call a partial package the full official
installation. FlashAttention and causal-conv1d imports are guarded optional
imports. No distributed launcher/process group is required for ordinary use.

Candidate wheel-only runtime pins, checked against public PyPI metadata:
`einops==0.8.1`, `ninja==1.11.1.4`, `transformers==4.51.3`,
`huggingface-hub==0.30.2`. The first two and HF packages have published wheels.
Transformers' base dependencies also include filelock, NumPy, packaging,
PyYAML, regex, requests, tokenizers>=0.21,<0.22, safetensors>=0.4.3 and tqdm.
No extras, model downloads or pretrained weights are needed. Existing packages
may already satisfy these. Use **only binary distributions**, constrain Torch
and Triton to their exact installed versions, and abort a resolver proposal
that changes either. Pins are a conservative candidate, not a successful
environment reproduction; Root must check the actual resolver and import.

## Official path without a new causal-conv1d build

The checked [Mamba2 implementation](https://github.com/state-spaces/mamba/blob/v2.3.1/mamba_ssm/modules/mamba2.py)
supports `use_mem_eff_path=False`. With no causal-conv1d package, its normal
non-packed sequence path uses `torch.nn.Conv1d` and the official Triton chunk
scan, including backward. This is still the author implementation, with a
disclosed unfused-convolution configuration. Set `seq_idx=None` and
`cu_seqlens=None`; packed convolution needs the optional extension. The fully
fused `use_mem_eff_path=True` path invokes causal-conv1d's compiled functions
and is not presently dependency-free. No performance equivalence between
these two configurations is assumed.

## Remaining execution boundary: Triton JIT

Avoiding causal-conv1d does **not** avoid JIT. Official Mamba2 scan and gated
normalization call Triton kernels. The archived official Triton3.4.0
`python/triton/runtime/build.py` invokes a C compiler with
`subprocess.check_call` (line49) for an uncached helper. Its NVIDIA compiler
invokes `ptxas --version` (line42) and `subprocess.run(ptxas_cmd,...)`
(line424). These are actual source requirements, not a guessed training error.

A precompiled Python wheel therefore does not ensure a cold forward/backward
works under a process policy denying child execution or compiler-cache writes.
The existing `/proc/self/task/` CUDA exception is unrelated to that permission.
No bypass, prewarming outside authorization, seccomp change or replacement
kernel was attempted. Root should first test the unmodified official import
and a bounded cold forward/backward under current constraints, recording cold
compilation/autotuning and cache storage. If those permissions block JIT, that
is the specific blocker to report; it must not be hidden by reverting to our
reference SSM or by calling warm-cache time total cost.

**Decision boundary:** an exact official wheel and lower-dependency source
path exist. Wheel contents, full imports, kernel compilation, gradients and
numerical execution remain for Root's actual validation. Neither official
Mamba2 compatibility nor benchmark validity is established by this read-only
check alone.
