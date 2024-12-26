use ndarray::{Array1, Array2, Axis};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[allow(non_snake_case)]
/// Largely inspired by the wrapper from rust_optimal_transport
pub fn c_emd_wrapper(
    a: &mut Array1<f64>,
    b: &mut Array1<f64>,
    M: &mut Array2<f64>,
    max_iter: i32,
) -> (Array2<f64>, f64, Array1<f64>, Array1<f64>, i32) {
    let n_1 = M.len_of(Axis(0));
    let n_2 = M.len_of(Axis(1));

    let mut cost = 0f64;
    let mut alpha = Array1::<f64>::zeros(n_1);
    let mut beta = Array1::<f64>::zeros(n_2);
    let mut G = Array2::<f64>::zeros((n_1, n_2));

    if a.is_empty() {
        *a = Array1::from_elem(n_1, 1f64 / n_1 as f64);
    }

    if b.is_empty() {
        *b = Array1::from_elem(n_2, 1f64 / n_2 as f64);
    }

    unsafe {
        let code = EMD_wrap(
            n_1 as i32,
            n_2 as i32,
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            M.as_mut_ptr(),
            G.as_mut_ptr(),
            alpha.as_mut_ptr(),
            beta.as_mut_ptr(),
            &mut cost,
            max_iter,
        );

        (G, cost, alpha, beta, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    /// From the examples in the python-optimal-transport docs
    fn test_c_wrapper() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];

        let mut costs = array![[0.0, 1.0], [1.0, 0.0]];

        let (ot_matrix, cost, _, _, _result) = c_emd_wrapper(&mut a, &mut b, &mut costs, 10000);

        assert_eq!(cost, 0.0);
        assert_eq!(ot_matrix, array![[0.5, 0.0], [0.0, 0.5]]);
    }
}
