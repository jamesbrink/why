use std::process::Command;

fn main() {
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

    println!("cargo:rustc-env=WHY_GIT_SHA={git_sha}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
