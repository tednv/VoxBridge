#include "voxbridge_engine.h"
#include "whisper.h"

#include <cstring>
#include <string>

#ifndef VOXBRIDGE_ENGINE_VARIANT_NAME
#define VOXBRIDGE_ENGINE_VARIANT_NAME "unknown"
#endif

struct voxbridge_engine {
    whisper_context* ctx = nullptr;
    std::string last_error;
};

static char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

extern "C" {

VOXBRIDGE_ENGINE_API voxbridge_engine* voxbridge_engine_load(const char* model_path) {
    if (model_path == nullptr) {
        return nullptr;
    }

    whisper_context_params cparams = whisper_context_default_params();
    whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (ctx == nullptr) {
        return nullptr;
    }

    voxbridge_engine* engine = new voxbridge_engine();
    engine->ctx = ctx;
    return engine;
}

VOXBRIDGE_ENGINE_API voxbridge_status voxbridge_engine_transcribe(
    voxbridge_engine* engine,
    const float* samples,
    size_t sample_count,
    const char* language,
    const char* initial_prompt,
    char** text_out
) {
    if (text_out != nullptr) {
        *text_out = nullptr;
    }
    if (engine == nullptr || engine->ctx == nullptr || samples == nullptr) {
        if (engine != nullptr) {
            engine->last_error = "invalid arguments";
        }
        return -1;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = language;
    wparams.translate = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_special = false;
    wparams.print_timestamps = false;
    if (initial_prompt != nullptr && initial_prompt[0] != '\0') {
        wparams.initial_prompt = initial_prompt;
    }

    int result = whisper_full(engine->ctx, wparams, samples, static_cast<int>(sample_count));
    if (result != 0) {
        engine->last_error = "whisper_full failed with code " + std::to_string(result);
        return result;
    }

    std::string text;
    const int n_segments = whisper_full_n_segments(engine->ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* segment = whisper_full_get_segment_text(engine->ctx, i);
        if (segment != nullptr) {
            text += segment;
        }
    }

    if (text_out != nullptr) {
        *text_out = dup_cstr(text);
        if (*text_out == nullptr) {
            engine->last_error = "out of memory copying result";
            return -2;
        }
    }

    return 0;
}

VOXBRIDGE_ENGINE_API void voxbridge_engine_free_string(char* text) {
    std::free(text);
}

VOXBRIDGE_ENGINE_API void voxbridge_engine_unload(voxbridge_engine* engine) {
    if (engine == nullptr) {
        return;
    }
    if (engine->ctx != nullptr) {
        whisper_free(engine->ctx);
    }
    delete engine;
}

VOXBRIDGE_ENGINE_API const char* voxbridge_engine_last_error(voxbridge_engine* engine) {
    if (engine == nullptr) {
        return "null engine handle";
    }
    return engine->last_error.c_str();
}

VOXBRIDGE_ENGINE_API const char* voxbridge_engine_variant_name(void) {
    return VOXBRIDGE_ENGINE_VARIANT_NAME;
}

} // extern "C"
