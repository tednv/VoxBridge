#!/usr/bin/env node
//
// Builds VoxBridge engine DLLs/SOs: self-contained builds of whisper.cpp/ggml + the
// voxbridge_engine C ABI (../native/shim/), compiled for a specific CPU ISA target or GPU
// backend, so multiple variants can coexist on disk without symbol collisions.
//
// This script is self-contained to the `voxbridge/` directory - it takes an explicit
// --out-dir (or VOXBRIDGE_OUT_DIR env var) rather than assuming anything about a consuming
// app's layout, so `voxbridge/` can be lifted into its own repository without rework. A
// consuming app decides where engines-dist/ should land for its own packaging (e.g. the
// voquill app's scripts/tauri-runner.mjs calls this with --out-dir pointing at
// src-tauri/engines-dist, which its tauri.conf.json bundles as a resource).
//
// Usage: node build-engines.mjs --out-dir <path>
//
// Platform status:
//   - Windows (x64): implemented, built and verified during development.
//   - Linux (x64): implemented, using the same variant matrix as Windows.
//     NOT verified on an actual Linux machine this session - written from the
//     same ggml CMake options used successfully on Windows, but flag any
//     Linux-specific build failures if they show up.
//   - macOS: NOT implemented - see buildMacosVariants() below for the intended design
//     and why (no consuming app in this codebase has macOS platform support yet).

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const nativeDir = path.resolve(__dirname, "..", "native");

function parseOutDir() {
  const flagIndex = process.argv.indexOf("--out-dir");
  if (flagIndex !== -1 && process.argv[flagIndex + 1]) {
    return path.resolve(process.argv[flagIndex + 1]);
  }
  if (process.env.VOXBRIDGE_OUT_DIR) {
    return path.resolve(process.env.VOXBRIDGE_OUT_DIR);
  }
  // Default: engines-dist/ next to this script's voxbridge/ root, for standalone use.
  return path.resolve(__dirname, "..", "engines-dist");
}

const distRoot = parseOutDir();
const familyFlagIndex = process.argv.indexOf("--family");
const requestedFamilies = familyFlagIndex !== -1 && process.argv[familyFlagIndex + 1]
  ? [process.argv[familyFlagIndex + 1]]
  : ["whisper", "llm"];

// Same MAX_PATH reasoning as CARGO_TARGET_DIR in the voquill app's tauri-runner.mjs:
// CMake/Ninja build trees nest deeply enough to blow past Windows' ~260 character path
// limit when built under a long project path. Override with VOXBRIDGE_BUILD_DIR if needed.
const buildRoot =
  process.env.VOXBRIDGE_BUILD_DIR ??
  (process.platform === "win32" ? "C:\\voxbridge-engines-build" : path.join(os.tmpdir(), "voxbridge-engines-build"));

function run(command, args, options = {}) {
  console.log(`\n$ ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    stdio: "inherit",
    shell: false,
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): ${command} ${args.join(" ")}`);
  }
}

function commandExists(command) {
  const locator = process.platform === "win32" ? "where.exe" : "which";
  const result = spawnSync(locator, [command], { stdio: "ignore" });
  return result.status === 0;
}

/**
 * @param {string} variant - e.g. "cpu-baseline", "cpu-avx2", "vulkan"
 * @param {string[]} extraCmakeFlags - additional -D flags for this variant
 * @param {string} dllExtension - "dll" | "so" | "dylib"
 */
function buildVariant(target, variant, extraCmakeFlags, dllExtension) {
  const buildDir = path.join(buildRoot, target, variant);
  const generatorFlags = process.env.CMAKE_GENERATOR ? [] : ["-G", "Ninja"];

  run("cmake", [
    "-S",
    nativeDir,
    "-B",
    buildDir,
    ...generatorFlags,
    `-DVOXBRIDGE_VARIANT=${variant}`,
    `-DVOXBRIDGE_TARGET=${target}`,
    "-DCMAKE_BUILD_TYPE=Release",
    ...extraCmakeFlags,
  ]);
  run("cmake", ["--build", buildDir, "--config", "Release"]);

  const family = target === "llm" ? "voxbridge_llm" : "voxbridge_engine";
  const builtName = `${family}_${variant}.${dllExtension}`;
  return findBuiltLibrary(buildDir, builtName);
}

// Ninja puts the output directly in buildDir; a multi-config generator (rare here,
// but e.g. Visual Studio generators) would nest it under Release/ - check both.
function findBuiltLibrary(buildDir, fileName) {
  const candidates = [path.join(buildDir, fileName), path.join(buildDir, "Release", fileName)];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Built library not found: ${fileName} (looked in ${candidates.join(", ")})`);
}

function copyToDist(builtPath, platformArchDir, fileName) {
  const destDir = path.join(distRoot, platformArchDir);
  fs.mkdirSync(destDir, { recursive: true });
  const destPath = path.join(destDir, fileName);
  fs.copyFileSync(builtPath, destPath);
  console.log(`  -> ${destPath}`);
  return destPath;
}

// x86_64 CPU tiers: a true SSE4.2-only floor plus an AVX2+FMA+F16C fast path. NOTE:
// plain GGML_NATIVE=OFF is NOT a true baseline by itself - ggml's own CMake defaults
// still turn on AVX/AVX2/FMA/F16C unless explicitly disabled. See the consuming app's
// docs/ROADMAP.local.md "Discovery" note - this also affects whisper-rs-based builds,
// tracked separately there.
const CPU_BASELINE_FLAGS = [
  "-DGGML_NATIVE=OFF",
  "-DGGML_AVX=OFF",
  "-DGGML_AVX2=OFF",
  "-DGGML_FMA=OFF",
  "-DGGML_F16C=OFF",
  "-DGGML_SSE42=ON",
];
const CPU_AVX2_FLAGS = ["-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON"]; // AVX/FMA/F16C follow via ggml's INS_ENB default
const VULKAN_FLAGS = ["-DGGML_VULKAN=ON"];

function buildWindowsVariants() {
  const platformArchDir = "windows-x64";
  console.log(`\n=== Building VoxBridge engines for ${platformArchDir} ===`);

  for (const target of requestedFamilies) {
    const family = target === "llm" ? "voxbridge_llm" : "voxbridge_engine";
    copyToDist(buildVariant(target, "cpu-baseline", CPU_BASELINE_FLAGS, "dll"), platformArchDir, `${family}_cpu-baseline.dll`);
    copyToDist(buildVariant(target, "cpu-avx2", CPU_AVX2_FLAGS, "dll"), platformArchDir, `${family}_cpu-avx2.dll`);

    if (process.env.VULKAN_SDK) {
      copyToDist(buildVariant(target, "vulkan", VULKAN_FLAGS, "dll"), platformArchDir, `${family}_vulkan.dll`);
    } else {
      console.warn(`  VULKAN_SDK not set - skipping ${target} vulkan engine variant`);
    }
  }
}

function buildLinuxVariants() {
  const arch = process.arch === "arm64" ? "arm64" : "x64";
  const platformArchDir = `linux-${arch}`;
  console.log(`\n=== Building VoxBridge engines for ${platformArchDir} (unverified this session - no Linux test machine) ===`);

  if (arch !== "x64") {
    // arm64 Linux desktops are rare and ggml has no ISA-tiering concept there (same
    // situation as macOS Apple Silicon) - just ship one native-ish CPU build.
    copyToDist(
      buildVariant("whisper", "cpu-baseline", ["-DGGML_NATIVE=OFF"], "so"),
      platformArchDir,
      "voxbridge_engine_cpu-baseline.so"
    );
    return;
  }

  for (const target of requestedFamilies) {
    const family = target === "llm" ? "voxbridge_llm" : "voxbridge_engine";
    copyToDist(buildVariant(target, "cpu-baseline", CPU_BASELINE_FLAGS, "so"), platformArchDir, `${family}_cpu-baseline.so`);
    copyToDist(buildVariant(target, "cpu-avx2", CPU_AVX2_FLAGS, "so"), platformArchDir, `${family}_cpu-avx2.so`);

    if (commandExists("glslc") || process.env.VULKAN_SDK) {
      copyToDist(buildVariant(target, "vulkan", VULKAN_FLAGS, "so"), platformArchDir, `${family}_vulkan.so`);
    } else {
      console.warn(`  Vulkan SDK/glslc not found - skipping ${target} vulkan engine variant`);
    }
  }
}

// --- macOS: NOT IMPLEMENTED. Skeleton only. ---
//
// No consuming app in this codebase has macOS platform support yet (see the voquill
// app's docs/ROADMAP.local.md for the full explanation - no .dmg has ever been
// released, no src-tauri/src/platform/macos exists there). Building engine variants for
// macOS now would be scaffolding for a platform nothing can launch on yet, so this is
// intentionally left unimplemented - scoped here so a future session (ideally on an
// actual Mac, since none of this has been buildable or testable without one) can pick
// it up with full context.
//
// Intended variant set, once a consuming app has macOS platform support:
//   - Apple Silicon (arm64): ONE cpu variant (no AVX-style ISA tiering on ARM - ggml's
//     `GGML_CPU_ARM_ARCH` is a single -march= string, not a matrix of boolean flags like
//     x86; NEON/dotprod are just always present on M-series chips) + ONE metal variant
//     (`-DGGML_METAL=ON` - ggml has native Metal support already, ARM Macs' GPU backend
//     of choice; do NOT attempt Vulkan-via-MoltenVK here, Metal is the correct native
//     path and far more stable in practice).
//   - Intel Mac (x64): cpu-baseline + cpu-avx2 (same x86_64 tiering as Windows/Linux,
//     Intel Macs broadly support AVX2) + metal (Metal also works on Intel Macs' AMD/Intel
//     GPUs, not just Apple Silicon).
//   - Build output extension: `.dylib`. Loader-side: macOS also needs an
//     `@rpath`/`install_name` fix-up (`-DCMAKE_INSTALL_RPATH=@loader_path` or similar) for
//     the dylib to find its own dependencies when loaded via dlopen from an app bundle -
//     not a concern on Windows/Linux, but relevant here.
function buildMacosVariants() {
  console.warn(
    "\n=== macOS engine variants: NOT IMPLEMENTED ===\n" +
      "See the comment above this function in voxbridge/scripts/build-engines.mjs. Skipping."
  );
}

function main() {
  fs.mkdirSync(distRoot, { recursive: true });

  if (process.platform === "win32") {
    buildWindowsVariants();
  } else if (process.platform === "linux") {
    buildLinuxVariants();
  } else if (process.platform === "darwin") {
    buildMacosVariants();
  } else {
    console.error(`Unsupported platform for engine builds: ${process.platform}`);
    process.exit(1);
  }

  console.log(`\nVoxBridge engine build complete. Output: ${distRoot}`);
}

main();
