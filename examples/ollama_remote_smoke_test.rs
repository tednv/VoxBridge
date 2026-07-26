//! Manual smoke test for LlmBackend::OllamaRemote against a real Ollama instance.
//!
//! Usage: cargo run --example ollama_remote_smoke_test -- <base_url> <model>

use voxbridge::LlmBackend;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <base_url> <model>", args[0]);
        std::process::exit(1);
    }
    let base_url = args[1].clone();
    let model = args[2].clone();

    println!("Checking reachability of {}...", base_url);
    let reachable = voxbridge::llm::ollama_is_reachable(&base_url, std::time::Duration::from_secs(5));
    println!("Reachable: {}", reachable);
    if !reachable {
        eprintln!("Not reachable, aborting.");
        std::process::exit(1);
    }

    let backend = LlmBackend::OllamaRemote { base_url, model };

    let system_prompt = "You clean up rough speech-to-text transcripts. Fix punctuation, \
        capitalization, and obvious transcription errors, but do not change the meaning or \
        add new content. Reply with only the corrected text, nothing else.";
    let user_prompt = "hello can you hear me im testing the voxbridge compose feature";

    println!("Running completion via remote Ollama...");
    let start = std::time::Instant::now();
    let result = backend
        .complete(Some(system_prompt), user_prompt, 128)
        .expect("completion failed");
    let elapsed = start.elapsed();

    println!("\n--- Result ({:.2}s) ---", elapsed.as_secs_f64());
    println!("Input:  {}", user_prompt);
    println!("Output: {}", result);
    println!("--- End ---\n");
}
