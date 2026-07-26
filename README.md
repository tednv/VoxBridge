# VoxBridge

A runtime-dispatched, per-CPU/GPU-variant [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
engine loader. VoxBridge builds several self-contained engine libraries — one per CPU
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
each a normal, separately-linked build of whisper.cpp/ggml with different compiler
flags — so there's no symbol collision between them (unlike `GGML_BACKEND_DL`, which
requires MODULE-only libraries and breaks static linking in bindings like `whisper-rs`).
Exactly one variant is loaded per process, chosen by inspecting the actual CPU/GPU at
runtime.

Measured on one development machine, same model and audio: a true SSE4.2 baseline took
57s where the AVX2 variant took 6.3s — the entire point of dispatching per-CPU instead of
guessing once at build time.

## Layout

- `native/` — the C ABI (`shim/voxbridge_engine.h`/`.cpp`) and CMake build
  (`CMakeLists.txt`) that produces one engine library per variant. `native/whisper.cpp`
  is a git submodule pointing at upstream.
- `src/lib.rs` — the Rust loader: CPUID-based variant selection, dlopen via
  [`libloading`](https://crates.io/crates/libloading), a safe wrapper over the C ABI.
- `scripts/build-engines.mjs` — builds the variant matrix for the current platform. Takes
  an explicit `--out-dir` (or `VOXBRIDGE_OUT_DIR` env var) since this crate has no opinion
  on where a consuming app wants its built libraries staged.

## Status

Early / actively developed. Windows and Linux (x64) CPU + Vulkan GPU variants are
implemented and tested. macOS is not yet implemented (see `scripts/build-engines.mjs`'s
`buildMacosVariants` for the intended design). Not yet published to crates.io.

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
