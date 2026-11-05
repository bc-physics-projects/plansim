#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use nalgebra::DVector;
use ode_solvers::System;
use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::dopri5::*;
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

struct FlatNBodies<const N: usize>
where
    [(); N * 6]:,
{
    inner: system::NBodies<N>,
}

impl<const N: usize> System<f64, DVector<f64>> for FlatNBodies<N>
where
    [(); N * 6]:,
{
    fn system(&self, t: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        let state = system::State::<N>::from_column_slice(y.as_slice());
        let mut derivative = system::State::<N>::zeros();

        <system::NBodies<N> as System<f64, system::State<N>>>::system(
            &self.inner,
            t,
            &state,
            &mut derivative,
        );

        dy.as_mut_slice().copy_from_slice(derivative.as_slice());
    }
}

struct GenericSolver<const N: usize>
where
    [(); N * 6]:,
{
    data: system::NBodies<N>,
    stepper: Option<Dopri5<f64, DVector<f64>, FlatNBodies<N>>>,
    output: Option<ContinuousOutputModel<f64, DVector<f64>>>,
}

impl<const N: usize> GenericSolver<N>
where
    [(); N * 6]:,
{
    pub fn new(masses: [f64; N]) -> Self {
        Self {
            data: system::NBodies { Ms: masses },
            stepper: None,
            output: None,
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
            let base = i * 6;
            accum_vec[base] = rs[i] * r_angs[i].cos();
            accum_vec[base + 1] = rs[i] * r_angs[i].sin();
            accum_vec[base + 2] = 0.0;
            accum_vec[base + 3] = vs[i] * v_angs[i].cos();
            accum_vec[base + 4] = vs[i] * v_angs[i].sin();
            accum_vec[base + 5] = 0.0;
        }
        let y0 = DVector::from_vec(accum_vec);

        let mut stepper = Dopri5::new(
            FlatNBodies {
                inner: self.data.clone(),
            },
            tspan.0,
            tspan.1,
            60.0,
            y0,
            1.0e-10,
            1.0e-10,
        );

        let mut continuous_output_model = ContinuousOutputModel::default();
        let res = stepper
            .integrate_with_continuous_output_model(&mut continuous_output_model)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if res.accepted_steps > 0 {
            self.stepper = Some(stepper);
            self.output = Some(continuous_output_model);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Integration failed"))
        }
    }

    pub fn results(&self, ts: &[f64]) -> Option<Vec<Vec<f64>>> {
        self.output.as_ref().map(|out| {
            ts.iter()
                .map(|&t| {
                    out.evaluate(t)
                        .as_slice()
                        .to_vec()
                        .into_iter()
                        .map(|x| x.as_slice().to_vec())
                })
                .flatten()
                .collect()
        })
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
                self.inner
    .results(&tpoints)
    .ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No results available")
    })
            }
    }
}
});
