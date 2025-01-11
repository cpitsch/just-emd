Just a wrapper around the C++ code used by Python Optimal Transport for the EMD. No more, no less.

- Uses the same C++ code as [rust-optimal-transport](https://crates.io/crates/rust-optimal-transport),
but uses different crates for creating the FFI. For more info, see _[See Also](#see-also)_.
    - C++ code is taken from the rust-optimal-transport [repository](https://github.com/kachark/rust-optimal-transport/tree/main/src/exact/fast_transport).

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

    // Create a solver instance to compute the EMD
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
for FFI. For this, you might need to install LLVM and set some environment variables.
For more information, see the [bindgen user guide](https://rust-lang.github.io/rust-bindgen/requirements.html).

## See Also

- [Rust-optimal-transport](https://crates.io/crates/rust-optimal-transport)
    - Wraps the same C++ library used for the EMD
    - Also contains many more optimal transport variants which are supported by Python Optimal Transport
    - However, some of these variants require OpenBLAS, which is complicated to set up, and
    when disabling these features flags, the library does not compile (on windows).
