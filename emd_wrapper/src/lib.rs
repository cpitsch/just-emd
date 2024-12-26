use ndarray::{Array1, Array2, Axis};
use thiserror::Error;
use wrap::c_emd_wrapper;

mod wrap;

#[derive(Error, Debug, PartialEq)]
pub enum EmdError {
    #[error("Dimensions of arguments do not match: Source distribution {0} and target distribution {1} do not match cost matrix dimensions {2}x{3}")]
    WeightDimensionError(usize, usize, usize, usize),
    #[error(transparent)]
    FastTransportError(#[from] FastTransportError),
    #[error("Number of iterations ({0}) must be > 0")]
    InvalidIterations(i32),
}

#[derive(Error, Debug, PartialEq)]
pub enum FastTransportError {
    #[error("Network simplex problem is infeasible")]
    Infeasible,
    #[error("Network simplex problem is unbounded")]
    Unbounded,
    #[error("Max iterations reached")]
    MaxIterReached,
}

impl From<i32> for FastTransportError {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Infeasible,
            1 => panic!("Cannot create FastTransportErrorCode for optimal solution"),
            2 => Self::Unbounded,
            3 => Self::MaxIterReached,
            _ => panic!("Invalid result code from FastTransport"),
        }
    }
}

#[derive(Debug)]
pub struct EmdResult {
    pub flow_matrix: Array2<f64>,
    pub emd: f64,
}

pub struct EmdSolver<'a> {
    source: &'a mut Array1<f64>,
    target: &'a mut Array1<f64>,
    costs: &'a mut Array2<f64>,
    iterations: i32,
}

impl<'a> EmdSolver<'a> {
    pub fn new(
        source: &'a mut Array1<f64>,
        target: &'a mut Array1<f64>,
        costs: &'a mut Array2<f64>,
    ) -> Self {
        Self {
            source,
            target,
            costs,
            iterations: 10000,
        }
    }

    pub fn iterations(mut self, iterations: i32) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn solve(&mut self) -> Result<EmdResult, EmdError> {
        emd(self.source, self.target, self.costs, self.iterations)
    }
}

pub fn emd(
    source_weights: &mut Array1<f64>,
    target_weights: &mut Array1<f64>,
    costs: &mut Array2<f64>,
    iterations: i32,
) -> Result<EmdResult, EmdError> {
    if iterations <= 0 {
        return Err(EmdError::InvalidIterations(iterations));
    }
    check_emd_input_shapes(source_weights, target_weights, costs)?;

    // From python optimal transport
    *target_weights *= source_weights.sum() / target_weights.sum();

    let (flow_matrix, cost, _a, _b, code) =
        c_emd_wrapper(source_weights, target_weights, costs, iterations);
    if code == 1 {
        Ok(EmdResult {
            flow_matrix,
            emd: cost,
        })
    } else {
        Err(FastTransportError::from(code))?
    }
}

fn check_emd_input_shapes(
    a: &Array1<f64>,
    b: &Array1<f64>,
    costs: &Array2<f64>,
) -> Result<(), EmdError> {
    let costs_dim_1 = costs.len_of(Axis(0));
    let costs_dim_2 = costs.len_of(Axis(1));

    let a_dim = a.len();
    let b_dim = b.len();

    if costs_dim_1 != a_dim || costs_dim_2 != b_dim {
        Err(EmdError::WeightDimensionError(
            a_dim,
            b_dim,
            costs_dim_1,
            costs_dim_2,
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    /// From the examples in the python-optimal-transport docs
    fn test_ot_simple_example() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];

        let mut costs = array![[0.0, 1.0], [1.0, 0.0]];

        let result = emd(&mut a, &mut b, &mut costs, 10000).unwrap();

        assert_eq!(result.emd, 0.0);
        assert_eq!(result.flow_matrix, array![[0.5, 0.0], [0.0, 0.5]]);
    }

    #[test]
    /// From the examples in the python-optimal-transport docs
    fn test_ot_simple_example_builder() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];

        let mut costs = array![[0.0, 1.0], [1.0, 0.0]];

        let result = EmdSolver::new(&mut a, &mut b, &mut costs)
            .iterations(1000)
            .solve()
            .unwrap();

        assert_eq!(result.emd, 0.0);
        assert_eq!(result.flow_matrix, array![[0.5, 0.0], [0.0, 0.5]]);
    }

    #[test]
    fn test_incorrect_dimensions_error() {
        let mut a: Array1<f64> = array![0.1, 0.3, 0.6];
        let mut b: Array1<f64> = array![1.0];

        let mut costs: Array2<f64> = Array2::from_elem((1, 3), 0.0); // Wrong order!

        let res = emd(&mut a, &mut b, &mut costs, 10000);

        assert!(res.is_err_and(|err| err
            == EmdError::WeightDimensionError(
                a.len(),
                b.len(),
                costs.shape()[0],
                costs.shape()[1]
            )));
    }

    #[test]
    fn test_max_iter() {
        // Random example that needs more than one iter; Found by trial and error
        let mut a = array![0.1, 0.1, 0.8];
        let mut b = array![0.5, 0.5];
        let mut costs = array![[0.3, 1.0], [1.5, 0.25], [0.1, 3.0]];

        let res = emd(&mut a, &mut b, &mut costs, 1);
        assert!(res.is_err_and(|err| matches!(
            err,
            EmdError::FastTransportError(FastTransportError::MaxIterReached)
        )));
    }
}
