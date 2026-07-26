//! Manual smoke test for the embedded LLM engine: loads a real GGUF model and runs one
//! completion, printing the result so it can be eyeballed for sanity.
//!
//! Usage: cargo run --example llm_smoke_test -- <engines_dist_dir> <model.gguf>

use std::path::PathBuf;
use voxbridge::llm::LlmEngine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <engines_dist_dir> <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let engines_dir = PathBuf::from(&args[1]);
    let model_path = &args[2];

    println!("Loading engine from {:?}...", engines_dir);
    let engine = LlmEngine::load_best(&engines_dir).expect("failed to load LLM engine");
    println!("Loaded engine variant: {}", engine.variant_name());

    println!("Loading model {}...", model_path);
    let model = engine
        .load_model(model_path, 2048, 0)
        .expect("failed to load model");
    println!("Model loaded.");

    let system_prompt = "You clean up rough speech-to-text transcripts. Fix punctuation, \
        capitalization, and obvious transcription errors, but do not change the meaning or \
        add new content. Reply with only the corrected text, nothing else.";
    let user_prompt = "hello can you hear me im testing the voxbridge compose feature";

    println!("Running completion...");
    let start = std::time::Instant::now();
    let result = model
        .complete(Some(system_prompt), user_prompt, 128)
        .expect("completion failed");
    let elapsed = start.elapsed();

    println!("\n--- Result ({:.2}s) ---", elapsed.as_secs_f64());
    println!("Input:  {}", user_prompt);
    println!("Output: {}", result);
    println!("--- End ---\n");
}
