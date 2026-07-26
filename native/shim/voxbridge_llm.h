#ifndef VOXBRIDGE_LLM_H
#define VOXBRIDGE_LLM_H

/*
 * Fixed C ABI for a VoxBridge embedded-LLM engine DLL. Each engine is a
 * self-contained build of llama.cpp/ggml (a specific CPU ISA target, or the
 * Vulkan GPU backend), compiled into its own DLL - same pattern as the
 * transcription engines in voxbridge_engine.h, just for short text-completion
 * calls (e.g. "clean up this transcript fragment") instead of audio
 * transcription. Only ONE variant gets loaded into the process at runtime.
 *
 * This header is the contract between the Rust loader (the `voxbridge` crate)
 * and every LLM engine DLL. Keep it stable - changing a signature here means
 * rebuilding every variant.
 */

#ifdef _WIN32
#define VOXBRIDGE_LLM_API __declspec(dllexport)
#else
#define VOXBRIDGE_LLM_API __attribute__((visibility("default")))
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct voxbridge_llm voxbridge_llm;

/* 0 = success. Negative = failure (see voxbridge_llm_last_error). */
typedef int voxbridge_status;

/*
 * Loads a GGUF model from `model_path` (UTF-8, null-terminated).
 * `n_ctx` is the context window size in tokens (e.g. 2048); pass 0 for the
 * model's own default. `n_gpu_layers` is how many layers to offload to GPU
 * (0 = CPU-only, a large number like 999 = offload everything that fits).
 * Returns NULL on failure.
 */
VOXBRIDGE_LLM_API voxbridge_llm* voxbridge_llm_load(
    const char* model_path,
    int n_ctx,
    int n_gpu_layers
);

/*
 * Runs one independent completion: `system_prompt` (may be NULL) plus
 * `user_prompt` are formatted through the model's own chat template, then
 * generated up to `max_tokens` tokens or end-of-generation, whichever comes
 * first. Each call is stateless with respect to previous calls on the same
 * handle - the model's KV cache is cleared first, so this is a one-shot
 * "transform this text" call, not a multi-turn chat session.
 *
 * `text_out` receives a newly allocated, null-terminated UTF-8 buffer that
 * the caller must free with voxbridge_llm_free_string(); on failure
 * `*text_out` is set to NULL.
 */
VOXBRIDGE_LLM_API voxbridge_status voxbridge_llm_complete(
    voxbridge_llm* llm,
    const char* system_prompt,
    const char* user_prompt,
    int max_tokens,
    char** text_out
);

VOXBRIDGE_LLM_API void voxbridge_llm_free_string(char* text);

VOXBRIDGE_LLM_API void voxbridge_llm_unload(voxbridge_llm* llm);

/* Human-readable description of the last error on this engine handle. */
VOXBRIDGE_LLM_API const char* voxbridge_llm_last_error(voxbridge_llm* llm);

/* Identifies which variant this DLL is, e.g. "cpu-avx2", "cpu-baseline", "vulkan". */
VOXBRIDGE_LLM_API const char* voxbridge_llm_variant_name(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXBRIDGE_LLM_H */
