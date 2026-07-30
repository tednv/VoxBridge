//! VoxBridge: a focused local speech-recognition and text-refinement runtime adapter.
//!
//! The crate normalizes established inference backends behind a Rust-facing lifecycle:
//! runtime-selected whisper.cpp and llama.cpp libraries, an optional managed Faster
//! Whisper/CTranslate2 worker, and text-only Ollama refinement. It owns backend
//! capability detection, loading, warmup, caching, fallback, and result adaptation;
//! consuming applications own recording, documents, agents, history, and UI policy.
//!
//! VoxBridge is deliberately not a generic provider gateway or inference engine. It
//! reuses the upstream projects and exposes only the focused runtime boundary required
//! by local transcription-and-refinement applications.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_float, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod llm;
pub mod faster_whisper;
pub use llm::{LlmBackend, LlmEngine, LlmModel};
pub use faster_whisper::{
    FasterWhisperBackend, FasterWhisperConfig, FasterWhisperDevice,
    FasterWhisperRuntime,
};

#[cfg(target_os = "windows")]
const DLL_EXTENSION: &str = "dll";
#[cfg(target_os = "linux")]
const DLL_EXTENSION: &str = "so";
#[cfg(target_os = "macos")]
const DLL_EXTENSION: &str = "dylib";

/// Directory name a build script (e.g. `scripts/build-engines.mjs`) writes variant
/// output under. Consuming apps append this to wherever they stage resources.
pub fn platform_arch_dir() -> String {
    #[cfg(target_os = "windows")]
    {
        "windows-x64".to_string()
    }
    #[cfg(target_os = "linux")]
    {
        if cfg!(target_arch = "aarch64") {
            "linux-arm64".to_string()
        } else {
            "linux-x64".to_string()
        }
    }
    #[cfg(target_os = "macos")]
    {
        if cfg!(target_arch = "aarch64") {
            "macos-arm64".to_string()
        } else {
            "macos-x64".to_string()
        }
    }
}

/// CPU variants this platform knows how to pick between, in preference order (first
/// supported+present wins). GPU variants aren't in this list - this loader doesn't do
/// GPU-availability detection itself (that's app-specific, e.g. Vulkan runtime presence
/// checks); see `Engine::load_best_gpu` instead, which a caller invokes once it has
/// already decided GPU should be tried.
#[cfg(target_arch = "x86_64")]
fn candidate_variants() -> Vec<(&'static str, bool)> {
    vec![
        (
            "cpu-avx2",
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        ),
        ("cpu-baseline", true), // SSE4.2 floor: assumed always present on x86_64
    ]
}

#[cfg(not(target_arch = "x86_64"))]
fn candidate_variants() -> Vec<(&'static str, bool)> {
    // aarch64: no ISA tiering, ggml uses a single -march= string for ARM rather than a
    // boolean matrix like x86.
    vec![("cpu-baseline", true)]
}

fn variant_path(engines_dir: &Path, variant: &str) -> PathBuf {
    engines_dir.join(format!("voxbridge_engine_{}.{}", variant, DLL_EXTENSION))
}

/// Resolves the directory containing this platform's built engine libraries, by trying
/// each of `candidate_bases` in order and appending `<base>/engines-dist/<platform>`
/// (the layout `scripts/build-engines.mjs` writes). Returns the first that exists.
///
/// This crate has no opinion on what those bases should be - a consuming app passes its
/// own resource directory, a dev-mode fallback path, or whatever else makes sense for
/// its own packaging, so this crate never needs to know the consuming app's directory
/// layout.
pub fn resolve_engines_dir(candidate_bases: &[&Path]) -> Option<PathBuf> {
    let arch_dir = platform_arch_dir();
    for base in candidate_bases {
        let candidate = base.join("engines-dist").join(&arch_dir);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

type FnLoad = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type FnTranscribe = unsafe extern "C" fn(
    *mut c_void,
    *const c_float,
    usize,
    *const c_char,
    *const c_char,
    *mut *mut c_char,
) -> i32;
type FnFreeString = unsafe extern "C" fn(*mut c_char);
type FnUnload = unsafe extern "C" fn(*mut c_void);
type FnLastError = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FnVariantName = unsafe extern "C" fn() -> *const c_char;

/// A loaded engine DLL/SO and its resolved symbols. The underlying `Library` is leaked
/// (never `FreeLibrary`'d): whisper.cpp/ggml spin up worker threads and register
/// process-exit state that isn't safe to unload out from under while the process is
/// still running - confirmed via a real segfault-on-exit during prototyping. Consumers
/// only ever want one engine loaded per process lifetime anyway, so "load once, never
/// unload, let the OS reclaim it at exit" is correct behavior, not a workaround.
pub struct Engine {
    variant: String,
    load: Symbol<'static, FnLoad>,
    transcribe: Symbol<'static, FnTranscribe>,
    free_string: Symbol<'static, FnFreeString>,
    unload: Symbol<'static, FnUnload>,
    last_error: Symbol<'static, FnLastError>,
}

/// A loaded whisper model handle within an `Engine`. Dropping this calls the engine's
/// `unload` (frees the model/context) but does NOT unload the DLL itself. Owns an
/// `Arc<Engine>` (rather than borrowing) so it can be stored in a cache alongside the
/// engine it came from without a self-referential struct.
pub struct Model {
    engine: Arc<Engine>,
    handle: *mut c_void,
}

// Safety: the whisper.cpp context behind `handle` is only ever accessed through
// `&self`/`&mut self` methods on `Model`; callers are expected to serialize access
// (e.g. behind a `Mutex`) if shared across threads, same assumption whisper-rs's
// `WhisperContext`/`WhisperState` make.
unsafe impl Send for Model {}

impl Engine {
    /// Tries each candidate variant in preference order, returning the first one that
    /// both exists on disk and successfully loads its symbol table. Does not attempt to
    /// load a model yet - see `load_model`. Returned wrapped in `Arc` since that's the
    /// only way it's ever consumed (`load_model` needs `Arc<Self>` so `Model` can share
    /// ownership rather than borrow with a lifetime, which is what makes this cacheable).
    pub fn load_best(engines_dir: &Path) -> Result<Arc<Self>, String> {
        let mut last_err = String::from("no candidate variants for this CPU/platform");

        for (variant, supported) in candidate_variants() {
            if !supported {
                continue;
            }
            let path = variant_path(engines_dir, variant);
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

    /// Loads the GPU variant, if present - the caller decides whether GPU should be
    /// tried at all (e.g. by checking Vulkan runtime availability first); this function
    /// just does the loading. Only one GPU backend variant exists in the build matrix
    /// today ("vulkan" - Windows/Linux; whisper.cpp's context defaults `use_gpu = true`,
    /// so simply loading this variant is enough to get GPU-accelerated inference, no
    /// extra flag needed). A "metal" variant would need its own equivalent once macOS
    /// platform support exists - see `scripts/build-engines.mjs`'s `buildMacosVariants`.
    pub fn load_best_gpu(engines_dir: &Path) -> Result<Arc<Self>, String> {
        let path = variant_path(engines_dir, "vulkan");
        if !path.exists() {
            return Err("no vulkan engine variant present in engines-dist".to_string());
        }
        Self::load_variant(&path, "vulkan").map(Arc::new)
    }

    fn load_variant(path: &Path, variant: &str) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }.map_err(|e| e.to_string())?;
        // Leak to 'static: exactly one engine is loaded for the process's lifetime (see
        // struct doc comment on why it's never unloaded).
        let lib: &'static Library = Box::leak(Box::new(lib));

        unsafe {
            let variant_name: Symbol<FnVariantName> = lib
                .get(b"voxbridge_engine_variant_name\0")
                .map_err(|e| e.to_string())?;
            let reported_name = CStr::from_ptr(variant_name()).to_string_lossy().into_owned();

            Ok(Engine {
                variant: reported_name,
                load: lib.get(b"voxbridge_engine_load\0").map_err(|e| e.to_string())?,
                transcribe: lib
                    .get(b"voxbridge_engine_transcribe\0")
                    .map_err(|e| e.to_string())?,
                free_string: lib
                    .get(b"voxbridge_engine_free_string\0")
                    .map_err(|e| e.to_string())?,
                unload: lib.get(b"voxbridge_engine_unload\0").map_err(|e| e.to_string())?,
                last_error: lib
                    .get(b"voxbridge_engine_last_error\0")
                    .map_err(|e| e.to_string())?,
            })
        }
        .map_err(|e: String| format!("{} (expected variant '{}')", e, variant))
    }

    pub fn variant_name(&self) -> &str {
        &self.variant
    }

    pub fn load_model(self: &Arc<Self>, model_path: &str) -> Result<Model, String> {
        let model_cstr = CString::new(model_path).map_err(|e| e.to_string())?;
        let handle = unsafe { (self.load)(model_cstr.as_ptr()) };
        if handle.is_null() {
            return Err(format!("engine '{}' returned a null handle loading the model", self.variant));
        }
        Ok(Model {
            engine: Arc::clone(self),
            handle,
        })
    }
}

impl Model {
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<String, String> {
        let language_cstr = language.map(|l| CString::new(l).unwrap_or_default());
        let language_ptr = language_cstr
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());

        let prompt_cstr = prompt.map(|p| CString::new(p).unwrap_or_default());
        let prompt_ptr = prompt_cstr
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());

        let mut text_out: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            (self.engine.transcribe)(
                self.handle,
                samples.as_ptr(),
                samples.len(),
                language_ptr,
                prompt_ptr,
                &mut text_out,
            )
        };

        if status != 0 {
            let err = unsafe { CStr::from_ptr((self.engine.last_error)(self.handle)).to_string_lossy() };
            return Err(format!("transcribe failed (status={}): {}", status, err));
        }

        let text = unsafe { CStr::from_ptr(text_out).to_string_lossy() }.into_owned();
        unsafe { (self.engine.free_string)(text_out) };
        Ok(text.trim().to_string())
    }

    pub fn engine_variant_name(&self) -> &str {
        &self.engine.variant
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.engine.unload)(self.handle) };
    }
}
