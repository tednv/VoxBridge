# VoxBridge

VoxBridge is a focused local inference-runtime adapter for applications that combine
speech recognition with text refinement. It presents a stable Rust-facing boundary
over established backends:

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with runtime-selected
  processor and Vulkan engine variants
- optional [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) with
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) CUDA or optimized processor
  inference
- embedded [llama.cpp](https://github.com/ggml-org/llama.cpp) text completion
- text-only refinement through a local or network [Ollama](https://ollama.com)
  server

VoxBridge owns backend capability detection, process and model lifecycle, warmup,
cancellation, caching, fallback, safe replacement, and normalized results. A
consuming application should not need to know whether a result came from a dynamic
C++ library, an optional managed worker, a GGML/GGUF artifact, or an Ollama request.

## Scope

VoxBridge is not a universal LLM gateway, hosted service, or generic
OpenAI-compatible proxy. Projects such as LiteLLM and LocalAI already cover broad
provider routing. VoxBridge intentionally does not pursue cloud-provider catalogs,
text-to-speech, image generation, vision, embeddings, or custom inference engines.

Its purpose is narrower: provide the runtime boundary needed by local applications
such as VoxBridge Compose while reusing mature inference projects. Product-specific
recording policy, document editing, agents, history, privacy, and interface behavior
belong in the consuming application.

## Why

Local desktop applications need more than a one-shot transcription endpoint. They
need to choose a viable backend for the actual machine, prepare it before a user
records, report useful progress, cancel or supersede stale work, switch safely, and
retain a fallback when an optional runtime is unavailable. VoxBridge centralizes that
mechanical lifecycle without turning application code into backend-specific plumbing.

### Runtime-selected native engines

A single statically built whisper.cpp or llama.cpp binary must choose one processor
instruction-set assumption for every user. An aggressive assumption can crash on
older hardware; a conservative one makes capable machines pay the performance cost
of the lowest common denominator.

VoxBridge builds multiple self-contained native libraries for each workload: a true
SSE4.2 floor, an AVX2/FMA fast path, and a Vulkan graphics variant. Exactly one
variant is loaded for a model after inspecting real processor and graphics
capabilities. Separate libraries avoid symbol collisions and do not depend on
`GGML_BACKEND_DL`, whose module-only model does not fit every static binding.

One development comparison using the same model and audio took 57 seconds on the
SSE4.2 baseline and 6.3 seconds on AVX2. That difference is why dispatch belongs in
the runtime rather than being guessed once at build time.

### Separate recognition and refinement

Growing Whisper prompts are a fragile way to edit continuous dictation. Long or
heavily punctuated prior text can increase latency and may encourage unrelated
output. VoxBridge therefore keeps recognition focused on decoding bounded audio
utterances and provides a separate refinement backend for text that has already
been recognized.

An embedded llama.cpp model or an Ollama server can refine that text independently.
The consuming application decides when and how to apply a completion, including
agent prompts, fidelity checks, retries, document revision rules, and raw-text
fallback. VoxBridge supplies the backend mechanics, not the product's editing policy.

## Backend model

### Speech recognition

- **Faster Whisper/CTranslate2:** default high-efficiency runtime. CTranslate2
  CUDA/FP16 is available on compatible NVIDIA systems; optimized processor inference
  is its fallback. CTranslate2 model directories are separate artifacts from
  whisper.cpp GGML files.
- **whisper.cpp:** compact compatibility path with processor and Vulkan variants,
  including broad NVIDIA and AMD graphics support.

Both backends must produce the same normalized transcription result and participate
in the same discovery, load, warmup, cancellation, status, and replacement lifecycle.

### Text refinement

- **Embedded llama.cpp:** in-process GGUF completion through runtime-selected native
  variants.
- **Ollama:** text-only completion through a user-configured local or network server.

Recognition and refinement remain separate capabilities. A consuming application
may use either refinement backend without changing the speech-recognition contract.

## Layout

- `native/` - native C ABIs and engine submodules:
  - `shim/voxbridge_engine.h` and `.cpp` for whisper.cpp
  - `shim/voxbridge_llm.h` and `.cpp` for llama.cpp
  - `native/whisper.cpp` and `native/llama.cpp` as upstream git submodules
- `src/lib.rs` - whisper.cpp loader, processor capability selection, dynamic library
  loading, and safe Rust model wrapper
- `src/llm.rs` - embedded llama.cpp loader plus the `LlmBackend` abstraction and
  Ollama HTTP adapter
- `src/faster_whisper.rs` - optional Faster Whisper worker process, request protocol,
  model lifecycle, and normalized transcription API
- `runtime/faster_whisper_worker.py` - persistent Faster Whisper/CTranslate2 worker
  used by the optional managed runtime
- `scripts/build-engines.mjs` - builds the native engine variant matrix into an
  explicit output directory
- `examples/` - manual end-to-end smoke tests for embedded and Ollama refinement

## Status

Early and actively developed. Windows and Linux x64 processor and Vulkan variants
are implemented and tested for whisper.cpp and llama.cpp. The Ollama adapter has
been verified against a real server.

The Faster Whisper backend is supported and has passed local Windows CUDA/FP16,
processor fallback, model warmup, transcription, and backend-switch testing.
Portable managed installation, packaging, cancellation, and progress integration
are not finished. macOS is not implemented. VoxBridge is not published to crates.io.

## Roadmap

- Complete portable packaging and managed installation for Faster
  Whisper/CTranslate2 behind the stable VoxBridge API while retaining
  whisper.cpp/Vulkan as the compact compatibility fallback.
- Turn the local persistent worker into a portable optional managed runtime with
  installation, health reporting, model discovery/downloads, preload/unload,
  progress, cancellation, sanitized errors, and normalized results.
- Expose explicit capabilities so consumers can choose CTranslate2 CUDA on
  compatible NVIDIA systems, optimized CTranslate2 processor inference elsewhere,
  or Vulkan on supported NVIDIA and AMD hardware.
- Activate backend changes only at safe utterance boundaries and prevent stale work
  from updating readiness or delivering results after a switch.
- Benchmark latency, word error rate, memory use, package size, and failure recovery
  on Windows and Linux.
- Formalize stable transcription and refinement backend traits so consumers depend
  on capabilities and normalized results instead of backend names.

Broad provider routing and unrelated modalities are explicit non-goals unless a
future consuming product establishes a concrete local-pipeline requirement.

## Building

Native variants require CMake, a C/C++ toolchain, and the Vulkan SDK for Vulkan
builds.

```bash
git submodule update --init --recursive
node scripts/build-engines.mjs --out-dir dist
cargo build
```

The optional Faster Whisper runtime is not yet part of the standard build/package
command.

## Credits

- [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) (MIT) performs
  native Whisper inference. VoxBridge builds, selects, loads, and adapts it.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) performs
  embedded language-model inference. VoxBridge builds, selects, loads, and adapts it.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) and
  [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT) provide the
  optional transcription pipeline and optimized Transformer inference. VoxBridge's
  role is worker lifecycle and result adaptation, not their inference implementation.
- [Ollama](https://ollama.com) provides the optional user-configured local or network
  model service.
- VoxBridge grew from work on a fork of
  [FOSS Voquill](https://github.com/jackbrumley/voquill) (AGPL-3.0) while
  investigating local transcription compatibility and performance. VoxBridge
  contains no FOSS Voquill application code and is licensed independently, but that
  project is the reason this runtime exists.

## License

MIT - see [LICENSE](LICENSE). Dependencies, submodules, optional runtimes, and model
weights retain their own licenses and terms.
