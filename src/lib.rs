#![feature(portable_simd)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use differential_equations::prelude::*;
use paste::paste;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use seq_macro::seq;
use std::sync::{OnceLock, atomic::AtomicBool};
mod system;

static CTRL_C_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();
static HANDLE_EXIT: AtomicBool = AtomicBool::new(false);

fn install_ctrlc_handler() -> PyResult<()> {
    let result = CTRL_C_HANDLER.get_or_init(|| {
        ctrlc::set_handler(move || {
            if HANDLE_EXIT.load(std::sync::atomic::Ordering::SeqCst) {
                std::process::exit(130);
            }
        })
        .map_err(|e| e.to_string())
    });

    match result {
        Ok(()) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.clone())),
    }
}

macro_rules! define_solvers {
    ($(($min:literal, $max:literal)),+ $(,)?) => {
        $(
        seq!(N in $min..=$max {
            paste! {
            #[pyclass(unsendable)]
            pub struct [<OrbitalSolver N>] {
                inner: GenericSolver<N>,
            }
            #[pymethods]
            impl [<OrbitalSolver N>] {
                #[new]
                #[pyo3(signature = (masses, *, kill_sigint=false, max_steps=None, abs_tol=None, rel_tol=None))]
                pub fn new(masses: [f64; N], kill_sigint: bool, max_steps: Option<usize>, abs_tol: Option<f64>, rel_tol: Option<f64>) -> Self {
                    HANDLE_EXIT.store(kill_sigint, std::sync::atomic::Ordering::SeqCst);
                    Self {
                    inner: GenericSolver::<N>::new(masses, max_steps, abs_tol, rel_tol),
                    }
                }

                /// Return a Polars DataFrame with time column `t` and zero-based body
                /// columns `"0"`, `"1"`, ... containing x/y/z/vx/vy/vz Struct fields.
                /// Access a component with df["0"].struct.field("x").
                #[allow(clippy::too_many_arguments)]
                pub fn solve(
                        &mut self,
                        py: Python<'_>,
                        xs: [f64; N],
                        ys: [f64; N],
                        zs: [f64; N],
                        vxs: [f64; N],
                        vys: [f64; N],
                        vzs: [f64; N],
                        tspan: (f64, f64),
                        tspace: f64,
                    ) -> PyResult<PyDataFrame> {
                        py.detach(|| self.inner.solve(xs, ys, zs, vxs, vys, vzs, tspan, tspace))
                            .map(PyDataFrame)
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
                    }
            }
        }
        });
        )+

        /// A Python module implemented in Rust.
        #[pymodule]
        mod plansim {
            use paste::paste;
            use pyo3::prelude::*;
            use seq_macro::seq;

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                crate::install_ctrlc_handler()?;
                $(seq!(N in $min..=$max {
                    paste! {
                        m.add_class::<super::[<OrbitalSolver N>]>()?;
                    }
                });)+
                Ok(())
            }
        }
    };
}

define_solvers!((15, 15), (90, 100));

struct GenericSolver<const N: usize> {
    data: system::NBodies<N>,
    max_steps: Option<usize>,
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
}

impl<const N: usize> GenericSolver<N> {
    pub fn new(
        mus: [f64; N],
        max_steps: Option<usize>,
        abs_tol: Option<f64>,
        rel_tol: Option<f64>,
    ) -> Self {
        Self {
            data: system::NBodies { mus },
            max_steps,
            abs_tol,
            rel_tol,
        }
    }

    pub fn solve(
        &mut self,
        xs: [f64; N],
        ys: [f64; N],
        zs: [f64; N],
        vxs: [f64; N],
        vys: [f64; N],
        vzs: [f64; N],
        tspan: (f64, f64),
        tspace: f64,
    ) -> anyhow::Result<DataFrame>
    where
        [(); 6 * N]:,
    {
        anyhow::ensure!(
            tspace.is_finite() && tspace > 0.0,
            "tspace must be finite and positive"
        );
        anyhow::ensure!(
            tspan.0.is_finite() && tspan.1.is_finite() && tspan.1 > tspan.0,
            "tspan must contain finite, increasing times"
        );
        let mut accum_vec = vec![0.0; N * 6];
        for (component, values) in [xs, ys, zs, vxs, vys, vzs].iter().enumerate() {
            accum_vec[component * N..(component + 1) * N].copy_from_slice(values);
        }
        let y0 = system::State::<N>::from_vec(accum_vec);

        let duration = tspan.1 - tspan.0;
        let stepper = ExplicitRungeKutta::dop853()
            .rtol(self.rel_tol.unwrap_or(1e-10))
            .atol(self.abs_tol.unwrap_or(1e-12))
            .max_steps(
                self.max_steps
                    .unwrap_or((10_000.0 + 1_000.0 * duration).ceil() as usize),
            );
        let system = system::NBodies::<N> { mus: self.data.mus };
        let solution = IVP::ode(&system, tspan.0, tspan.1, y0)
            .method(stepper)
            .even(tspace)
            .solve()?;
        let frame = solution.to_polars()?;
        let mut columns = vec![frame.column("t")?.clone()];
        for body in 0..N {
            let fields = ["x", "y", "z", "vx", "vy", "vz"]
                .iter()
                .enumerate()
                .map(|(component, name)| {
                    let mut field = frame
                        .column(&format!("y{}", component * N + body))?
                        .as_materialized_series()
                        .clone();
                    field.rename((*name).into());
                    Ok(field)
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            columns.push(
                StructChunked::from_series(body.to_string().into(), frame.height(), fields.iter())?
                    .into_series()
                    .into(),
            );
        }
        Ok(DataFrame::new(frame.height(), columns)?)
    }
}
