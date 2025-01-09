Just a wrapper around the C code used by Python Optimal Transport. No more, no less.

## Building

- This crate uses [bindgen](https://crates.io/crates/bindgen) for FFI. For this, you might
need to install LLVM and set some environment variables. See [here](https://rust-lang.github.io/rust-bindgen/requirements.html).

## See Also

There is also the crate rust-optimal-transport, which also wraps the same library, however
it has other dependencies that require OpenBLAS, which, to my knowledge, cannot be disabled
without making the library unusable (for some reason, the library doesn't compile when all
feature flags are disabled).
