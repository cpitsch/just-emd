#![doc = include_str!("../README.md")]

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;
use wrap::c_emd_wrapper;

mod wrap;

#[derive(Error, Debug, PartialEq)]
/// An error that is encountered in the computation of the EMD.
pub enum EmdError {
    /// A mismatch is detected between the dimensions of the populations and the cost matrix.
    /// Expected 0x1, found 2x3.
    #[error("Dimensions of arguments do not match: Source distribution {0} and target distribution {1} do not match cost matrix dimensions {2}x{3}")]
    WeightDimensionError(usize, usize, usize, usize),
    /// An error is raised by the fast_transport library. See [FastTransportError].
    #[error(transparent)]
    FastTransportError(#[from] FastTransportError),
    /// An invalid (negative or zero) number of iterations was supplied.
    #[error("Number of iterations ({0}) must be > 0")]
    InvalidIterations(i32),
}

#[derive(Error, Debug, PartialEq)]
/// An error that is raised by the fast_transport library. Based on the error code
/// given with the solution.
pub enum FastTransportError {
    /// The Optimal Transport instance is infeasible.
    #[error("Network simplex problem is infeasible")]
    Infeasible,
    /// The Optimal Transport instance is unbounded.
    #[error("Network simplex problem is unbounded")]
    Unbounded,
    /// The maximum number of iterations was reached without reaching the optimal solution.
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

#[derive(Debug, Clone, PartialEq)]
/// The result of the EMD computation.
pub struct EmdResult {
    /// The optimal transport solution matrix
    pub flow_matrix: Array2<f64>,
    /// The Earth Mover's Distance computed between the two populations
    pub emd: f64,
}

/// A struct used to solve a particular EMD instance.
///
/// # Examples
///
/// ```
/// use just_emd::EmdSolver;
/// use ndarray::array;
///
/// let mut source = array![0.3, 0.4, 0.2];
/// let mut target = array![0.2, 0.8, 0.0];
///
/// let mut costs = array![
///     [0.0, 1.0, 2.0],
///     [1.0, 0.0, 1.0],
///     [2.0, 1.0, 0.0]
/// ];
///
/// let res = EmdSolver::new(&mut source, &mut target, &mut costs)
///     .iterations(10000)
///     .solve()
///     .unwrap();
///
/// assert!(0.32 - res.emd <= 1e-5);
/// ```
///
/// # See Also
/// - The EMD can also be computed directly using [emd]
pub struct EmdSolver<'a> {
    /// The relative frequencies of the items in the source population.
    source: &'a mut Array1<f64>,
    /// The relative frequencies of the items in the target population.
    target: &'a mut Array1<f64>,
    /// The costs between the items of the two populations.
    costs: &'a mut Array2<f64>,
    /// The maximum number of iterations to perform in the network simplex algorithm. By default,
    /// this is 100000 iterations.
    iterations: i32,
}

impl<'a> EmdSolver<'a> {
    /// Create a new `EmdSolver`
    ///
    /// * `source` - The relative frequencies of the items in the source population.
    /// * `target` - The relative frequencies of the items in the target population.
    /// * `costs` - The cost matrix, giving a cost to matching a unit of the source item to
    /// a unit of the target item. Must have shape |`source`|x|`target`|.
    pub fn new(
        source: &'a mut Array1<f64>,
        target: &'a mut Array1<f64>,
        costs: &'a mut Array2<f64>,
    ) -> Self {
        Self {
            source,
            target,
            costs,
            iterations: 100000,
        }
    }

    /// Adjust the maximum number of iterations that are performed in the network simplex
    /// algorithm. By default, 100000 iterations are performed.
    pub fn iterations(mut self, iterations: i32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Solve the EMD instance.
    pub fn solve(&mut self) -> Result<EmdResult, EmdError> {
        emd(self.source, self.target, self.costs, self.iterations)
    }
}

/// Compute the Earth Mover's Distance between two populations
///
/// * `source_weights` - The relative frequencies of the items in the source population.
/// * `target_weights` - The relative frequencies of the items in the target population.
/// * `costs` - The cost matrix, giving a cost to matching a unit of the source item to
/// a unit of the target item. Must have shape |`source_weights`|x|`target_weights`|.
/// * `iterations` - The maximum number of iterations to perform in the network simplex algorithm.
///
/// # Examples
///
/// ```
/// use just_emd::emd;
/// use ndarray::array;
///
/// let mut source = array![0.3, 0.4, 0.2];
/// let mut target = array![0.2, 0.8, 0.0];
///
/// // Absolute difference as cost function
/// let mut costs = array![
///     [0.0, 1.0, 2.0],
///     [1.0, 0.0, 1.0],
///     [2.0, 1.0, 0.0]
/// ];
///
/// let res = emd(&mut source, &mut target, &mut costs, 10000).unwrap();
/// assert!(0.32 - res.emd <= 1e-5);
/// ```
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

/// Check that the dimensions of both populations match the dimensions of the cost matrix.
///
/// The length of `a` should match the number of rows in the costs matrix, and the
/// length of `b` should match the number of rows.
///
/// If this does not hold, an [EmdError] is returned.
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
