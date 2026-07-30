//! Optional Faster Whisper/CTranslate2 transcription backend.
//!
//! VoxBridge owns the worker protocol and process lifecycle so consuming applications
//! see a stable model-oriented API regardless of the active recognition runtime.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FasterWhisperDevice {
    Auto,
    Cpu,
    Cuda,
}

impl FasterWhisperDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FasterWhisperConfig {
    pub model: String,
    pub device: FasterWhisperDevice,
    pub compute_type: String,
    pub model_cache_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct FasterWhisperRuntime {
    pub python: PathBuf,
    pub worker: PathBuf,
}

impl FasterWhisperRuntime {
    pub fn new(python: impl Into<PathBuf>, worker: impl Into<PathBuf>) -> Self {
        Self {
            python: python.into(),
            worker: worker.into(),
        }
    }

    pub fn discover(candidate_bases: &[&Path]) -> Result<Self, String> {
        if let Ok(python) = std::env::var("VOXBRIDGE_FASTER_WHISPER_PYTHON") {
            let worker = std::env::var("VOXBRIDGE_FASTER_WHISPER_WORKER")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("runtime/faster_whisper_worker.py")
                });
            return Self::validate(Self::new(python, worker));
        }

        for base in candidate_bases {
            let runtime = base.join("faster-whisper-runtime");
            #[cfg(target_os = "windows")]
            let python = runtime.join(".venv/Scripts/python.exe");
            #[cfg(not(target_os = "windows"))]
            let python = runtime.join(".venv/bin/python");
            let worker = runtime.join("faster_whisper_worker.py");
            if python.exists() && worker.exists() {
                return Self::validate(Self::new(python, worker));
            }
        }

        Err("Faster Whisper runtime is not installed".to_string())
    }

    fn validate(runtime: Self) -> Result<Self, String> {
        if !runtime.python.is_file() {
            return Err("Faster Whisper Python runtime was not found".to_string());
        }
        if !runtime.worker.is_file() {
            return Err("Faster Whisper worker was not found".to_string());
        }
        Ok(runtime)
    }
}

#[derive(Serialize)]
struct WorkerRequest<'a> {
    id: u64,
    command: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_cache_dir: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    samples_f32_le_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
}

#[derive(Deserialize)]
struct WorkerResponse {
    id: u64,
    ok: bool,
    #[serde(default)]
    text: String,
    #[serde(default)]
    error: String,
    #[serde(default)]
    device: String,
    #[serde(default)]
    compute_type: String,
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl WorkerProcess {
    fn request(&mut self, mut request: WorkerRequest<'_>) -> Result<WorkerResponse, String> {
        self.next_id += 1;
        request.id = self.next_id;
        serde_json::to_writer(&mut self.stdin, &request).map_err(|e| e.to_string())?;
        self.stdin.write_all(b"\n").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;

        let mut line = String::new();
        let count = self.stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("Faster Whisper worker exited unexpectedly".to_string());
        }
        let response: WorkerResponse = serde_json::from_str(&line)
            .map_err(|e| format!("Invalid Faster Whisper worker response: {e}"))?;
        if response.id != self.next_id {
            return Err("Faster Whisper worker response was out of sequence".to_string());
        }
        if !response.ok {
            return Err(response.error);
        }
        Ok(response)
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct FasterWhisperBackend {
    worker: Mutex<WorkerProcess>,
    model: String,
    device: String,
    compute_type: String,
}

impl FasterWhisperBackend {
    pub fn load(runtime: &FasterWhisperRuntime, config: FasterWhisperConfig) -> Result<Self, String> {
        std::fs::create_dir_all(&config.model_cache_dir).map_err(|e| e.to_string())?;
        let mut child = Command::new(&runtime.python)
            .arg("-u")
            .arg(&runtime.worker)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to start Faster Whisper worker: {e}"))?;
        let stdin = child.stdin.take().ok_or("Worker stdin was unavailable")?;
        let stdout = child.stdout.take().ok_or("Worker stdout was unavailable")?;
        let mut worker = WorkerProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
        };

        let cache = config
            .model_cache_dir
            .to_str()
            .ok_or("Faster Whisper model cache path is invalid")?;
        let response = worker.request(WorkerRequest {
            id: 0,
            command: "load",
            model: Some(&config.model),
            device: Some(config.device.as_str()),
            compute_type: Some(&config.compute_type),
            model_cache_dir: Some(cache),
            samples_f32_le_hex: None,
            language: None,
            prompt: None,
        })?;

        Ok(Self {
            worker: Mutex::new(worker),
            model: config.model,
            device: response.device,
            compute_type: response.compute_type,
        })
    }

    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<String, String> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                samples.as_ptr().cast::<u8>(),
                std::mem::size_of_val(samples),
            )
        };
        let mut encoded = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            use std::fmt::Write as _;
            let _ = write!(encoded, "{byte:02x}");
        }
        let response = self.worker.lock().unwrap().request(WorkerRequest {
            id: 0,
            command: "transcribe",
            model: None,
            device: None,
            compute_type: None,
            model_cache_dir: None,
            samples_f32_le_hex: Some(encoded),
            language,
            prompt,
        })?;
        Ok(response.text.trim().to_string())
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn device_name(&self) -> &str {
        &self.device
    }

    pub fn compute_type(&self) -> &str {
        &self.compute_type
    }
}
