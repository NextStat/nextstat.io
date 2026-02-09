fn main() {
    // On macOS, Python extension modules are typically linked with unresolved Python
    // symbols and rely on the interpreter to provide them at load time.
    //
    // `-undefined dynamic_lookup` matches what tools like maturin set up.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
