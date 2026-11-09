use differential_equations::prelude::*;
use nalgebra::DVector;
use paste::paste;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
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
                #[allow(clippy::too_many_arguments)] // Preserve the public Python API.
                pub fn solve(
                        &mut self,
                        py: Python<'_>,
                        rs: [f64; N],
                        r_angs: [f64; N],
                        vs: [f64; N],
                        v_angs: [f64; N],
                        tspan: (f64, f64),
                        tspace: f64,
                    ) -> PyResult<Py<PyAny>> {
                        let (tpoints, data) = py.detach(|| self.inner.solve(rs, r_angs, vs, v_angs, tspan, tspace))
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                        results_dataframe::<N>(py, tpoints, data)
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

            /// Formats the sum of two numbers as string.
            #[pyfunction]
            fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
                Ok((a + b).to_string())
            }

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
        rs: [f64; N],
        r_angs: [f64; N],
        vs: [f64; N],
        v_angs: [f64; N],
        tspan: (f64, f64),
        tspace: f64,
    ) -> anyhow::Result<(Vec<f64>, Vec<DVector<f64>>)> {
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
            let (rsin, rcos) = r_angs[i].sin_cos();
            let (vsin, vcos) = v_angs[i].sin_cos();
            accum_vec[base] = rs[i] * rcos;
            accum_vec[base + 1] = rs[i] * rsin;
            accum_vec[base + 2] = 0.0;
            accum_vec[base + 3] = vs[i] * vcos;
            accum_vec[base + 4] = vs[i] * vsin;
            accum_vec[base + 5] = 0.0;
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
        Ok((solution.t, solution.y))
    }
}

/// One row per time sample, with a Struct column per zero-based body number.
fn results_dataframe<const N: usize>(
    py: Python<'_>,
    times: Vec<f64>,
    data: Vec<DVector<f64>>,
) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let dataframe = polars.getattr("DataFrame")?;
    let series = polars.getattr("Series")?;
    let frombuffer = py.import("numpy")?.getattr("frombuffer")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", polars.getattr("Float64")?)?;
    let columns = PyDict::new(py);
    columns.set_item("t", series.call(("t", times), Some(&kwargs))?)?;
    for body in 0..N {
        let fields = PyDict::new(py);
        for (component, name) in ["x", "y", "z", "vx", "vy", "vz"].iter().enumerate() {
            // Write directly into a Python-owned buffer: no per-sample Python floats.
            // NumPy retains the bytes owner; Polars retains/copies the array as needed.
            let bytes = PyBytes::new_with(py, data.len() * 8, |buffer| {
                for (row, chunk) in data.iter().zip(buffer.as_chunks_mut::<8>().0) {
                    chunk.copy_from_slice(&row[body * 6 + component].to_ne_bytes());
                }
                Ok(())
            })?;
            let values = frombuffer.call1((bytes, "float64"))?;
            fields.set_item(*name, series.call((*name, values), Some(&kwargs))?)?;
        }
        let name = body.to_string();
        let body_column = dataframe
            .call1((fields,))?
            .call_method1("to_struct", (&name,))?;
        columns.set_item(name, body_column)?;
    }
    Ok(dataframe.call1((columns,))?.unbind())
}
