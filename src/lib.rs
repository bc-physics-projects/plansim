use differential_equations::prelude::*;
use nalgebra::DVector;
use paste::paste;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use seq_macro::seq;
mod system;

macro_rules! define_solvers {
    ($(($min:literal, $max:literal)),+ $(,)?) => {
        $(
        seq!(N in $min..=$max {
            paste! {
            #[pyclass]
            pub struct [<OrbitalSolver N>] {
                inner: GenericSolver<N>,
            }
            #[pymethods]
            impl [<OrbitalSolver N>] {
                #[new]
                pub fn new(masses: [f64; N]) -> Self {
                    Self {
                        inner: GenericSolver::<N>::new(masses),
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
}

impl<const N: usize> GenericSolver<N> {
    pub fn new(mus: [f64; N]) -> Self {
        Self {
            data: system::NBodies { mus },
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
    ) -> anyhow::Result<DataFrame> {
        anyhow::ensure!(
            tspace.is_finite() && tspace > 0.0,
            "tspace must be finite and positive"
        );
        anyhow::ensure!(
            tspan.0.is_finite() && tspan.1.is_finite() && tspan.1 > tspan.0,
            "tspan must contain finite, increasing times"
        );
        let mut accum_vec = vec![0.0; N * 6];
        for i in 0..N {
            let base = i * 6;
            accum_vec[base..base + 6]
                .copy_from_slice(&[xs[i], ys[i], zs[i], vxs[i], vys[i], vzs[i]]);
        }
        let y0 = DVector::from_vec(accum_vec);

        let stepper = ExplicitRungeKutta::dop853()
            .rtol(1e-8)
            .atol(1e-6)
            .max_steps(usize::MAX);
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
                        .column(&format!("y{}", body * 6 + component))?
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
