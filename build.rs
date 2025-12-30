use std::env;
use std::process::Command;

fn main() {
    // Check if WHY_GIT_SHA is already set (e.g., by nix build)
    let git_sha = if let Ok(sha) = env::var("WHY_GIT_SHA") {
        if !sha.is_empty() && sha != "unknown" {
            sha
        } else {
            get_git_sha()
        }
    } else {
        get_git_sha()
    };

    println!("cargo:rustc-env=WHY_GIT_SHA={git_sha}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-env-changed=WHY_GIT_SHA");
}

fn get_git_sha() -> String {
    let mut git_sha = "unknown".to_string();
    let mut is_dirty = false;

    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            git_sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
    }

    if git_sha != "unknown" {
        if let Ok(status) = Command::new("git").args(["diff", "--quiet"]).status() {
            is_dirty = !status.success();
        }
    }

    if is_dirty {
        git_sha.push_str("-dirty");
    }

    git_sha
}
