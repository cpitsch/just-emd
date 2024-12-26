Just a wrapper around the C code used by Python Optimal Transport. No more, no less.

## See Also

There is also the crate rust-optimal-transport, which also wraps the same library, however
it has other dependencies that require OpenBLAS, which, to my knowledge, cannot be disabled
without making the library unusable (for some reason, the library doesn't compile when all
feature flags are disabled).
