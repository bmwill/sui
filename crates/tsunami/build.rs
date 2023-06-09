use std::env;
use std::process::Command;
use std::str;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let rustc_version = rustc_version().unwrap();
    println!("cargo:rustc-env=RUSTC_VERSION={rustc_version}");
}

fn rustc_version() -> Option<String> {
    let rustc = env::var_os("RUSTC")?;

    let output = Command::new(rustc).arg("--version").output().ok()?;

    let version = str::from_utf8(&output.stdout).ok()?;

    let mut pieces = version.split(' ');
    if pieces.next() != Some("rustc") {
        return None;
    }

    let next = pieces.next()?;

    assert!(next.starts_with("1."));

    Some(next.into())
}
