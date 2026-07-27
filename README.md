# VoxBridge

A runtime-dispatched, per-CPU/GPU-variant engine loader for two workloads:
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) speech-to-text, and (newer)
[llama.cpp](https://github.com/ggerganov/llama.cpp)-based small-LLM text cleanup, either
run in-process or proxied to a remote [Ollama](https://ollama.com) instance. VoxBridge
builds several self-contained engine libraries per workload — one per CPU
instruction-set tier (a true SSE4.2 floor, an AVX2+FMA fast path) plus a Vulkan
GPU-accelerated variant — and picks the best one for the running machine at startup via
real CPUID/GPU-availability detection, instead of shipping one generic build tuned to a
guess about "safe" hardware.

## Why

A single statically-built whisper.cpp binary has to pick one CPU instruction-set
assumption for every user. Get it wrong and it crashes on hardware that doesn't support
those instructions (a real, recurring class of bug); get it conservative and every user
pays the performance cost of the lowest common denominator, even on hardware that could
run several times faster.

VoxBridge sidesteps this by building multiple fully self-contained variant libraries —
each a normal, separately-linked build of whisper.cpp/ggml (or llama.cpp/ggml) with
different compiler flags — so there's no symbol collision between them (unlike
`GGML_BACKEND_DL`, which requires MODULE-only libraries and breaks static linking in
bindings like `whisper-rs`). Exactly one variant is loaded per process, chosen by
inspecting the actual CPU/GPU at runtime.

Measured on one development machine, same model and audio: a true SSE4.2 baseline took
57s where the AVX2 variant took 6.3s — the entire point of dispatching per-CPU instead of
guessing once at build time.

### Why an LLM engine too

A natural next step for live dictation is context-aware cleanup: fix punctuation, casing,
and small transcription slips based on what came before. The obvious way to get whisper.cpp
that context is feeding prior text back in as an `initial_prompt` on the next decode - but
in practice this is fragile for a continuous-dictation workload. A long or heavily
punctuated prompt can push whisper.cpp's temperature-fallback retry logic into a bad state,
producing multi-second stalls and, in the worst case, unrelated hallucinated output, on
otherwise-ordinary utterances. That's whisper.cpp working as designed for its actual job
(single-shot, prompt-free transcription of one audio segment) - it's the "feed it a growing
context prompt every decode" usage pattern that doesn't fit.

So VoxBridge keeps whisper.cpp doing exactly what it's fast and reliable at - decode one
utterance, prompt-free, as soon as it's spoken - and added a second, independent engine for
the cleanup pass instead of overloading the first one. A small instruct LLM (embedded via
llama.cpp, or proxied to a separate Ollama instance) runs asynchronously over already-
transcribed text, batching a few sentences of real context at a time, entirely decoupled
from the live transcription stream. Nothing ever blocks on it, and a raw/uncorrected
fallback is always available if the correction pass fails, gets rejected by a fidelity
check, or simply hasn't finished yet.

## Layout

- `native/` — the C ABI for both engines and their shared CMake build:
  - `shim/voxbridge_engine.h`/`.cpp` (whisper.cpp) and `shim/voxbridge_llm.h`/`.cpp`
    (llama.cpp), selected via the `VOXBRIDGE_TARGET` CMake option.
  - `native/whisper.cpp` and `native/llama.cpp` are git submodules pointing at upstream.
- `src/lib.rs` - the whisper.cpp Rust loader: CPUID-based variant selection, dlopen via
  [`libloading`](https://crates.io/crates/libloading), a safe wrapper over the C ABI.
- `src/llm.rs` - the LLM side: an analogous `LlmEngine`/`LlmModel` loader for the embedded
  llama.cpp path, a `LlmBackend` enum unifying it with an `OllamaRemote { base_url, model }`
  proxy behind one `.complete()` call, and the Ollama HTTP client (`ureq`, no async runtime
  needed here).
- `scripts/build-engines.mjs` - builds the variant matrix for the current platform. Takes
  an explicit `--out-dir` (or `VOXBRIDGE_OUT_DIR` env var) since this crate has no opinion
  on where a consuming app wants its built libraries staged.
- `examples/llm_smoke_test.rs`, `examples/ollama_remote_smoke_test.rs` - manual end-to-end
  test harnesses against real GGUF models / a real Ollama instance, kept as runnable
  examples rather than thrown away once they passed.

## Status

Early / actively developed. Windows and Linux (x64) CPU + Vulkan GPU variants are
implemented and tested for both the whisper.cpp and llama.cpp engines, and the Ollama
remote-proxy path has been verified against a real instance. macOS is not yet implemented
(see `scripts/build-engines.mjs`'s `buildMacosVariants` for the intended design). Not yet
published to crates.io.

## Building

Requires CMake, a C/C++ toolchain, and (for the Vulkan variant) the Vulkan SDK.

```bash
git submodule update --init --recursive
node scripts/build-engines.mjs --out-dir dist
cargo build
```

## Credits

- [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) (MIT) is the engine
  VoxBridge builds and dispatches between - all the actual transcription work happens
  there.
- This project grew out of experiments on a fork of
  [jackbrumley/voquill](https://github.com/jackbrumley/voquill) (AGPL-3.0), a voice
  dictation app, while investigating CPU-crash and performance issues in its local
  transcription backend. VoxBridge itself contains no code from that project - it's
  written directly against whisper.cpp - so it's licensed independently under MIT. Thanks
  to the voquill project for being the reason this exists.

## License

MIT - see [LICENSE](LICENSE).
