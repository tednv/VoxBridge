#ifndef VOXBRIDGE_ENGINE_H
#define VOXBRIDGE_ENGINE_H

/*
 * Fixed C ABI for a VoxBridge transcription engine DLL. Each engine is a
 * self-contained build of whisper.cpp/ggml (a specific CPU ISA target, or
 * the Vulkan GPU backend), compiled into its own DLL so that multiple
 * variants can coexist on disk without symbol collisions - only ONE gets
 * loaded into the process at runtime via LoadLibrary/dlopen, chosen by the
 * Rust host based on CPUID / GPU availability.
 *
 * This header is the contract between the Rust loader (the `voxbridge` crate)
 * and every engine DLL. Keep it stable - changing a signature here means
 * rebuilding every variant.
 */

#ifdef _WIN32
#define VOXBRIDGE_ENGINE_API __declspec(dllexport)
#else
#define VOXBRIDGE_ENGINE_API __attribute__((visibility("default")))
#endif

#include <stddef.h> /* size_t - unsigned long would be 32-bit on Windows LLP64 but
                        64-bit on Linux LP64, making the ABI inconsistent across
                        platforms. size_t is the right width on both. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct voxbridge_engine voxbridge_engine;

/* 0 = success. Negative = failure (see voxbridge_engine_last_error). */
typedef int voxbridge_status;

/*
 * Loads a whisper.cpp model from `model_path` (UTF-8, null-terminated).
 * Returns NULL on failure - call voxbridge_engine_last_error() on the
 * temporary handle is not possible in that case, so failures here are
 * reported by returning NULL only; the caller has nothing else to inspect
 * yet since no handle exists.
 */
VOXBRIDGE_ENGINE_API voxbridge_engine* voxbridge_engine_load(const char* model_path);

/*
 * Runs transcription on 16kHz mono f32 PCM samples. `text_out` receives a
 * newly allocated, null-terminated UTF-8 buffer that the caller must free
 * with voxbridge_engine_free_string(); on failure `*text_out` is set to NULL.
 * `language` may be NULL for auto-detect.
 */
VOXBRIDGE_ENGINE_API voxbridge_status voxbridge_engine_transcribe(
    voxbridge_engine* engine,
    const float* samples,
    size_t sample_count,
    const char* language,
    const char* initial_prompt,
    char** text_out
);

VOXBRIDGE_ENGINE_API void voxbridge_engine_free_string(char* text);

VOXBRIDGE_ENGINE_API void voxbridge_engine_unload(voxbridge_engine* engine);

/* Human-readable description of the last error on this engine handle. */
VOXBRIDGE_ENGINE_API const char* voxbridge_engine_last_error(voxbridge_engine* engine);

/* Identifies which variant this DLL is, e.g. "cpu-avx2", "cpu-baseline", "vulkan". */
VOXBRIDGE_ENGINE_API const char* voxbridge_engine_variant_name(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXBRIDGE_ENGINE_H */
