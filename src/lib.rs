#![feature(generic_const_exprs)]
// #![allow(incomplete_features)]

use std::iter;

use ode_solvers::dopri5::*;
use ode_solvers::*;
use pyo3::prelude::*;
mod system;

/// A Python module implemented in Rust.
#[pymodule]
mod plansim {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pymodule_export]
    use super::FloatClass;
}

macro_rules! create_interface {
    ($name: ident, $size: literal) => {
        #[pyclass]
        pub struct $name {
            inner: GenericSolver<$size>,
        }
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(data: [f64; $size]) -> Self {
                Self {
                    inner: GenericSolver::<$size>::new(data),
                }
            }
        }
    };
}

struct GenericSolver<const N: usize> {
    data: system::NBodies<N>,
}

impl<const N: usize> GenericSolver<N>
where
    [(); N * 6]:,
{
    pub fn new(masses: [f64; N]) -> Self {
        Self {
            data: system::NBodies { Ms: masses },
        }
    }

    pub fn solve(
        &mut self,
        rs: [f64; N],
        r_angs: [f64; N],
        vs: [f64; N],
        v_angs: [f64; N],
    ) -> anyhow::Result<dop_shared::Stats> {
        let mut accum_vec: Vec<f64> = Vec::from([0.0; N * 6]);
        for i in 0..N {
            accum_vec[i] = rs[i] * r_angs[i].cos();
            accum_vec[i + 1] = rs[i] * r_angs[i].sin();
            accum_vec[i + 2] = 0.0;
            accum_vec[i + 3] = vs[i] * v_angs[i].cos();
            accum_vec[i + 4] = vs[i] * v_angs[i].sin();
            accum_vec[i + 5] = 0.0;
        }
        let y0 = system::State::<N>::from_row_slice(&accum_vec);
        let period = std::f64::consts::TAU;

        // Create a stepper and run the integration.
        let mut stepper = Dopri5::new(
            self.data.clone(),
            0.0,
            5.0 * period,
            60.0,
            y0,
            1.0e-10,
            1.0e-10,
        );
        Ok(stepper.integrate()?)
    }
}

macro_rules! generate_interfaces {
    ($($size:expr),*) => {
        $(
            create_interface!(OrbitalSolver$size, $size);
        )*
    };
}

generate_interfaces!(2..10);
