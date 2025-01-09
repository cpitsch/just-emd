Just a wrapper around the C code used by Python Optimal Transport. No more, no less.

## Examples
```rust
use just_emd::{emd, EmdSolver};
use ndarray::array;

fn main() {
    let mut source = array![0.3, 0.4, 0.2];
    let mut target = array![0.2, 0.8, 0.0];

    // Absolute difference as cost
    let mut costs = array![
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ];

    let res = EmdSolver::new(&mut source, &mut target, &mut costs)
        .iterations(10000)
        .solve()
        .unwrap();

    println!("EMD: {}", res.emd); // EMD: 0.31999999999999995

    // Alternatively, the emd function can be used:
    let res_2 = emd(&mut source, &mut target, &mut costs, 10000).unwrap();

    assert_eq!(res.emd, res_2.emd);
}
```




## Building
- This crate uses [cc](https://crates.io/crates/cc) for C compilation and [bindgen](https://crates.io/crates/bindgen)
for FFI. For this, you might need to install LLVM and set some environment variables. See
[here](https://rust-lang.github.io/rust-bindgen/requirements.html).

## See Also

There is also the crate [rust-optimal-transport](https://crates.io/crates/rust-optimal-transport),
which also wraps the same library, however it has other dependencies that require OpenBLAS,
which, to my knowledge, cannot be disabled without making the library unusable (for some reason,
the library doesn't compile when all feature flags are disabled).
