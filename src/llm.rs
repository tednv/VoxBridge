//! Embedded small-LLM engine loader, plus a remote-proxy alternative.
//!
//! Mirrors the transcription `Engine`/`Model` split in `lib.rs`: `LlmEngine` loads a
//! variant DLL/SO (`voxbridge_llm_<variant>`), `LlmModel` is a loaded GGUF model handle
//! within it. `LlmBackend` wraps either an embedded `LlmModel` or a proxy to a remote
//! Ollama instance (local or over the network) behind one `complete()` call, so a
//! consuming app can switch between them without caring which one is actually running
//! the model.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::DLL_EXTENSION;

fn llm_variant_path(engines_dir: &Path, variant: &str) -> PathBuf {
    engines_dir.join(format!("voxbridge_llm_{}.{}", variant, DLL_EXTENSION))
}

#[cfg(target_arch = "x86_64")]
fn llm_candidate_variants() -> Vec<(&'static str, bool)> {
    vec![
        (
            "cpu-avx2",
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        ),
        ("cpu-baseline", true),
    ]
}

#[cfg(not(target_arch = "x86_64"))]
fn llm_candidate_variants() -> Vec<(&'static str, bool)> {
    vec![("cpu-baseline", true)]
}

type FnLlmLoad = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type FnLlmComplete = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *mut *mut c_char,
) -> i32;
type FnLlmFreeString = unsafe extern "C" fn(*mut c_char);
type FnLlmUnload = unsafe extern "C" fn(*mut c_void);
type FnLlmLastError = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FnLlmVariantName = unsafe extern "C" fn() -> *const c_char;

/// A loaded LLM engine DLL/SO and its resolved symbols. Like `Engine`, the underlying
/// `Library` is leaked - never unloaded for the process's lifetime (see `Engine`'s doc
/// comment on why: ggml worker threads and process-exit state aren't safe to unload out
/// from under).
pub struct LlmEngine {
    variant: String,
    load: Symbol<'static, FnLlmLoad>,
    complete: Symbol<'static, FnLlmComplete>,
    free_string: Symbol<'static, FnLlmFreeString>,
    unload: Symbol<'static, FnLlmUnload>,
    last_error: Symbol<'static, FnLlmLastError>,
}

/// A loaded GGUF model handle within an `LlmEngine`. Owns an `Arc<LlmEngine>` so it can
/// be cached alongside the engine it came from.
pub struct LlmModel {
    engine: Arc<LlmEngine>,
    handle: *mut c_void,
}

// Safety: same assumption as `Model` - the llama.cpp context behind `handle` is only
// accessed through `&self` methods; callers serialize access (e.g. behind a `Mutex`) if
// shared across threads.
unsafe impl Send for LlmModel {}

impl LlmEngine {
    pub fn load_best(engines_dir: &Path) -> Result<Arc<Self>, String> {
        let mut last_err = String::from("no candidate variants for this CPU/platform");

        for (variant, supported) in llm_candidate_variants() {
            if !supported {
                continue;
            }
            let path = llm_variant_path(engines_dir, variant);
            if !path.exists() {
                continue;
            }
            match Self::load_variant(&path, variant) {
                Ok(engine) => return Ok(Arc::new(engine)),
                Err(e) => last_err = format!("variant '{}' failed to load: {}", variant, e),
            }
        }

        Err(last_err)
    }

    pub fn load_best_gpu(engines_dir: &Path) -> Result<Arc<Self>, String> {
        let path = llm_variant_path(engines_dir, "vulkan");
        if !path.exists() {
            return Err("no vulkan llm engine variant present in engines-dist".to_string());
        }
        Self::load_variant(&path, "vulkan").map(Arc::new)
    }

    fn load_variant(path: &Path, variant: &str) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }.map_err(|e| e.to_string())?;
        let lib: &'static Library = Box::leak(Box::new(lib));

        unsafe {
            let variant_name: Symbol<FnLlmVariantName> = lib
                .get(b"voxbridge_llm_variant_name\0")
                .map_err(|e| e.to_string())?;
            let reported_name = CStr::from_ptr(variant_name()).to_string_lossy().into_owned();

            Ok(LlmEngine {
                variant: reported_name,
                load: lib.get(b"voxbridge_llm_load\0").map_err(|e| e.to_string())?,
                complete: lib.get(b"voxbridge_llm_complete\0").map_err(|e| e.to_string())?,
                free_string: lib
                    .get(b"voxbridge_llm_free_string\0")
                    .map_err(|e| e.to_string())?,
                unload: lib.get(b"voxbridge_llm_unload\0").map_err(|e| e.to_string())?,
                last_error: lib
                    .get(b"voxbridge_llm_last_error\0")
                    .map_err(|e| e.to_string())?,
            })
        }
        .map_err(|e: String| format!("{} (expected variant '{}')", e, variant))
    }

    pub fn variant_name(&self) -> &str {
        &self.variant
    }

    /// `n_ctx` is the context window in tokens (0 = the model's own default);
    /// `n_gpu_layers` is how many layers to offload to GPU (0 = CPU-only).
    pub fn load_model(
        self: &Arc<Self>,
        model_path: &str,
        n_ctx: i32,
        n_gpu_layers: i32,
    ) -> Result<LlmModel, String> {
        let model_cstr = CString::new(model_path).map_err(|e| e.to_string())?;
        let handle = unsafe { (self.load)(model_cstr.as_ptr(), n_ctx, n_gpu_layers) };
        if handle.is_null() {
            return Err(format!(
                "llm engine '{}' returned a null handle loading the model",
                self.variant
            ));
        }
        Ok(LlmModel {
            engine: Arc::clone(self),
            handle,
        })
    }
}

impl LlmModel {
    /// One independent, stateless completion - see `voxbridge_llm_complete`'s doc
    /// comment in the native shim for why this isn't a running chat session.
    pub fn complete(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        max_tokens: i32,
    ) -> Result<String, String> {
        let system_cstr = system_prompt.map(|s| CString::new(s).unwrap_or_default());
        let system_ptr = system_cstr
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());

        let user_cstr = CString::new(user_prompt).map_err(|e| e.to_string())?;

        let mut text_out: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            (self.engine.complete)(
                self.handle,
                system_ptr,
                user_cstr.as_ptr(),
                max_tokens,
                &mut text_out,
            )
        };

        if status != 0 {
            let err =
                unsafe { CStr::from_ptr((self.engine.last_error)(self.handle)).to_string_lossy() };
            return Err(format!("llm completion failed (status={}): {}", status, err));
        }

        let text = unsafe { CStr::from_ptr(text_out).to_string_lossy() }.into_owned();
        unsafe { (self.engine.free_string)(text_out) };
        Ok(text.trim().to_string())
    }

    pub fn engine_variant_name(&self) -> &str {
        &self.engine.variant
    }
}

impl Drop for LlmModel {
    fn drop(&mut self) {
        unsafe { (self.engine.unload)(self.handle) };
    }
}

/// Where a completion actually runs: embedded in-process (`LlmEngine`/`LlmModel` above),
/// or proxied to a remote Ollama instance - the same machine (`http://localhost:11434`)
/// or another one on the network. Both sides of this are optional/independent; a
/// consuming app picks one based on its own settings.
pub enum LlmBackend {
    Embedded(LlmModel),
    OllamaRemote { base_url: String, model: String },
}

impl LlmBackend {
    pub fn complete(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        max_tokens: i32,
    ) -> Result<String, String> {
        match self {
            LlmBackend::Embedded(model) => model.complete(system_prompt, user_prompt, max_tokens),
            LlmBackend::OllamaRemote { base_url, model } => {
                ollama_complete(base_url, model, system_prompt, user_prompt, max_tokens)
            }
        }
    }
}

#[derive(serde::Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Serialize)]
struct OllamaOptions {
    num_predict: i32,
}

#[derive(serde::Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
    options: OllamaOptions,
    /// Reasoning models (e.g. the Qwen3 family) default to emitting a `<think>` trace
    /// before the real answer, which Ollama returns in a separate `thinking` field -
    /// `content` stays empty until the model finishes reasoning, which a short
    /// `num_predict` budget (sized for a quick text-cleanup pass, not a long think) may
    /// never reach. Disabling it goes straight to the answer.
    think: bool,
}

#[derive(serde::Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

#[derive(serde::Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

/// Checks whether an Ollama instance is actually reachable at `base_url` (a real
/// connectivity check, not just "the URL is well-formed"), so a caller can show a clear
/// "not connected" state instead of failing silently on the first real request.
pub fn ollama_is_reachable(base_url: &str, timeout: Duration) -> bool {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    ureq::get(&url)
        .timeout(timeout)
        .call()
        .map(|resp| resp.status() == 200)
        .unwrap_or(false)
}

fn ollama_complete(
    base_url: &str,
    model: &str,
    system_prompt: Option<&str>,
    user_prompt: &str,
    max_tokens: i32,
) -> Result<String, String> {
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(OllamaMessage {
            role: "system",
            content: system,
        });
    }
    messages.push(OllamaMessage {
        role: "user",
        content: user_prompt,
    });

    let request = OllamaChatRequest {
        model,
        messages,
        stream: false,
        options: OllamaOptions {
            num_predict: if max_tokens > 0 { max_tokens } else { 512 },
        },
        think: false,
    };

    let response = ureq::post(&url)
        .timeout(Duration::from_secs(60))
        .send_json(&request)
        .map_err(|e| format!("Ollama request to {} failed: {}", url, e))?;

    let parsed: OllamaChatResponse = response
        .into_json()
        .map_err(|e| format!("Ollama response from {} was not valid JSON: {}", url, e))?;

    Ok(parsed.message.content.trim().to_string())
}
