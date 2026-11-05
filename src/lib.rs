#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::iter;

use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::dop_shared::SolverResult;
use ode_solvers::dopri5::*;
use ode_solvers::*;
use paste::paste;
use pyo3::prelude::*;
use seq_macro::seq;
mod system;

/// A Python module implemented in Rust.
#[pymodule]
mod plansim {
    use paste::paste;
    use pyo3::prelude::*;
    use seq_macro::seq;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        seq!(N in 2..=10 {
            paste! {
                m.add_class::<super::[<OrbitalSolver N>]>()?;
            }
        });
        Ok(())
    }
}

struct GenericSolver<const N: usize>
where
    [(); N * 6]:,
{
    data: system::NBodies<N>,
    stepper: Option<Dopri5<f64, system::State<N>, system::NBodies<N>>>,
}

impl<const N: usize> GenericSolver<N>
where
    [(); N * 6]:,
{
    pub fn new(masses: [f64; N]) -> Self {
        Self {
            data: system::NBodies { Ms: masses },
            stepper: None,
        }
    }

    pub fn solve(
        &mut self,
        rs: [f64; N],
        r_angs: [f64; N],
        vs: [f64; N],
        v_angs: [f64; N],
        tspan: (f64, f64),
    ) -> anyhow::Result<()> {
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

        // Create a stepper and run the integration.
        let mut stepper = Dopri5::new(
            self.data.clone(),
            tspan.0,
            tspan.1,
            60.0,
            y0,
            1.0e-10,
            1.0e-10,
        );
        if stepper.integrate()?.accepted_steps > 0 {
            self.stepper = Some(stepper);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Integration failed"))
        }
    }

    pub fn results(&self) -> Option<(&Vec<f64>, &Vec<system::State<N>>)> {
        self.stepper.as_ref().map(|stepper| stepper.results().get())
    }
}

seq!(N in 2..=10 {
    paste! {
    #[pyclass]
    pub struct [<OrbitalSolver N>] {
        inner: GenericSolver<N>,
    }
    #[pymethods]
    impl [<OrbitalSolver N>] {
        #[new]
        pub fn new(data: [f64; N]) -> Self {
            Self {
                inner: GenericSolver::<N>::new(data),
            }
        }

        pub fn solve(
                &mut self,
                rs: [f64; N],
                r_angs: [f64; N],
                vs: [f64; N],
                v_angs: [f64; N],
                tspan: (f64, f64),
                tpoints: Vec<f64>,
            ) -> Result<Vec<Vec<f64>>, PyErr> {
                self.inner.solve(rs, r_angs, vs, v_angs, tspan).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let (vec1, vec2) = self.inner.results().ok_or(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No results available"))?;
                let mut continuous_output_model = ContinuousOutputModel::default();
                let stepper = self.inner.stepper.as_ref().ok_or(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No stepper available"))?;
                let res = stepper.integrate_with_continuous_output_model(&mut continuous_output_model).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                continuous_output_model.evaluate(1.2);

                Ok(tpoints.iter().map(|&t| {
                    let state = continuous_output_model.evaluate(t);
                    state.as_slice().to_vec()
                }).collect())
            }
    }
}
});
