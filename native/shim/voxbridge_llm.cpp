#include "voxbridge_llm.h"
#include "llama.h"

#include <cstring>
#include <string>
#include <vector>

#ifndef VOXBRIDGE_LLM_VARIANT_NAME
#define VOXBRIDGE_LLM_VARIANT_NAME "unknown"
#endif

struct voxbridge_llm {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
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

VOXBRIDGE_LLM_API voxbridge_llm* voxbridge_llm_load(
    const char* model_path,
    int n_ctx,
    int n_gpu_layers
) {
    if (model_path == nullptr) {
        return nullptr;
    }

    // Only print errors - the examples default to verbose logging, which we don't want
    // leaking into this process's own logs.
    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            std::fprintf(stderr, "%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    if (n_ctx > 0) {
        ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
        ctx_params.n_batch = static_cast<uint32_t>(n_ctx);
    }

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        return nullptr;
    }

    voxbridge_llm* llm = new voxbridge_llm();
    llm->model = model;
    llm->ctx = ctx;
    return llm;
}

VOXBRIDGE_LLM_API voxbridge_status voxbridge_llm_complete(
    voxbridge_llm* llm,
    const char* system_prompt,
    const char* user_prompt,
    int max_tokens,
    char** text_out
) {
    if (text_out != nullptr) {
        *text_out = nullptr;
    }
    if (llm == nullptr || llm->ctx == nullptr || llm->model == nullptr || user_prompt == nullptr) {
        if (llm != nullptr) {
            llm->last_error = "invalid arguments";
        }
        return -1;
    }

    // Stateless per call: clear the KV cache so this completion doesn't see any
    // previous call's tokens. This engine is used for independent "clean up this
    // fragment" calls, not a running multi-turn chat.
    llama_memory_clear(llama_get_memory(llm->ctx), true);

    const llama_vocab* vocab = llama_model_get_vocab(llm->model);
    const char* tmpl = llama_model_chat_template(llm->model, /* name */ nullptr);

    std::vector<llama_chat_message> messages;
    if (system_prompt != nullptr && system_prompt[0] != '\0') {
        messages.push_back({"system", system_prompt});
    }
    messages.push_back({"user", user_prompt});

    std::vector<char> formatted(4096);
    int formatted_len = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, formatted.data(), static_cast<int32_t>(formatted.size())
    );
    if (formatted_len > static_cast<int>(formatted.size())) {
        formatted.resize(formatted_len);
        formatted_len = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), true, formatted.data(), static_cast<int32_t>(formatted.size())
        );
    }
    if (formatted_len < 0) {
        llm->last_error = "failed to apply chat template";
        return -2;
    }
    std::string prompt(formatted.data(), formatted_len);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(llm->ctx), 0) == -1;
    const int n_prompt_tokens = -llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, is_first, true
    );
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()), is_first, true
        ) < 0) {
        llama_sampler_free(smpl);
        llm->last_error = "failed to tokenize prompt";
        return -3;
    }

    std::string response;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    const int effective_max_tokens = max_tokens > 0 ? max_tokens : 512;

    for (int generated = 0; generated < effective_max_tokens; ++generated) {
        const int n_ctx = llama_n_ctx(llm->ctx);
        const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(llm->ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            llm->last_error = "context size exceeded";
            break;
        }

        if (llama_decode(llm->ctx, batch) != 0) {
            llama_sampler_free(smpl);
            llm->last_error = "llama_decode failed";
            return -4;
        }

        llama_token new_token_id = llama_sampler_sample(smpl, llm->ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            llama_sampler_free(smpl);
            llm->last_error = "failed to convert token to piece";
            return -5;
        }
        response.append(buf, n);

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_sampler_free(smpl);

    if (text_out != nullptr) {
        *text_out = dup_cstr(response);
        if (*text_out == nullptr) {
            llm->last_error = "out of memory copying result";
            return -6;
        }
    }

    return 0;
}

VOXBRIDGE_LLM_API void voxbridge_llm_free_string(char* text) {
    std::free(text);
}

VOXBRIDGE_LLM_API void voxbridge_llm_unload(voxbridge_llm* llm) {
    if (llm == nullptr) {
        return;
    }
    if (llm->ctx != nullptr) {
        llama_free(llm->ctx);
    }
    if (llm->model != nullptr) {
        llama_model_free(llm->model);
    }
    delete llm;
}

VOXBRIDGE_LLM_API const char* voxbridge_llm_last_error(voxbridge_llm* llm) {
    if (llm == nullptr) {
        return "null llm handle";
    }
    return llm->last_error.c_str();
}

VOXBRIDGE_LLM_API const char* voxbridge_llm_variant_name(void) {
    return VOXBRIDGE_LLM_VARIANT_NAME;
}

} // extern "C"
