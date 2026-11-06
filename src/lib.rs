use differential_equations::prelude::*;
use nalgebra::DVector;
use paste::paste;
use pyo3::prelude::*;
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
            #[pyclass(extends=OrbitalSolverResults, subclass)]
            pub struct [<OrbitalSolverResults N>] {
                data: Vec<Vec<f64>>,
                times: Vec<f64>,
            }
            #[pymethods]
            impl [<OrbitalSolver N>] {
                #[new]
                pub fn new(masses: [f64; N]) -> Self {
                    Self {
                        inner: GenericSolver::<N>::new(masses),
                    }
                }

                pub fn solve(
                        &mut self,
                        py: Python<'_>,
                        rs: [f64; N],
                        r_angs: [f64; N],
                        vs: [f64; N],
                        v_angs: [f64; N],
                        tspan: (f64, f64),
                        tspace: f64,
                    ) -> Result<Py<[<OrbitalSolverResults N>]>, PyErr> {
                        let (tpoints, data) = self.inner.solve(rs, r_angs, vs, v_angs, tspan, tspace)
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                            Py::new(
                                py,
                                (
                                    [<OrbitalSolverResults N>]::new(data, tpoints),
                                    OrbitalSolverResults {},
                                ),
                            )
                    }
            }

            impl [<OrbitalSolverResults N>] {
                fn new(data: Vec<Vec<f64>>, times: Vec<f64>) -> Self {
                    assert_eq!(data.first().unwrap().len(), N*6);
                    Self { data, times }
                }
            }

            #[pymethods]
            impl [<OrbitalSolverResults N>] {
                fn rows(&self) -> PyResult<&Vec<Vec<f64>>> {
                    Ok(&self.data)
                }
                fn cols(&self) -> PyResult<Vec<Vec<&f64>>> {
                    let mut cols: Vec<Vec<&f64>> = vec![Vec::new(); N*6];
                    for row in &self.data {
                        assert_eq!(row.len(), N*6);
                        for (j, val) in row.iter().enumerate() {
                            cols[j].push(val);
                        }
                    }
                    Ok(cols)
                }
                fn get_x(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[index*6]).collect())
                }
                fn get_y(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[(index*6)+1]).collect())
                }
                fn get_z(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[(index*6)+2]).collect())
                }
                fn get_vx(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[(index*6)+3]).collect())
                }
                fn get_vy(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[(index*6)+4]).collect())
                }
                fn get_vz(&self, index: usize) -> PyResult<Vec<&f64>> {
                    if index >= N {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
                    }
                    Ok(self.data.iter().map(|row| &row[(index*6)+5]).collect())
                }
                fn get_t(&self) -> PyResult<&Vec<f64>> {
                    Ok(&self.times)
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
                // m.add_class::<super::OrbitalSolver>()?;
                m.add_class::<super::OrbitalSolverResults>()?;
                $(seq!(N in $min..=$max {
                    paste! {
                        m.add_class::<super::[<OrbitalSolver N>]>()?;
                        m.add_class::<super::[<OrbitalSolverResults N>]>()?;
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
    ) -> anyhow::Result<(Vec<f64>, Vec<Vec<f64>>)> {
        anyhow::ensure!(
            tspace.is_finite() && tspace > 0.0,
            "tspace must be finite and positive"
        );
        let mut accum_vec = vec![0.0; N * 6];
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

        let stepper = ExplicitRungeKutta::dop853()
            .rtol(1e-8)
            .atol(1e-6)
            .max_steps(usize::MAX);
        let system = system::NBodies::<N> { mus: self.data.mus };
        let solution = IVP::ode(&system, tspan.0, tspan.1, y0)
            .method(stepper)
            .even(tspace)
            .solve()?;
        Ok((
            solution.t,
            solution
                .y
                .into_iter()
                .map(|state| state.as_slice().to_vec())
                .collect(),
        ))
    }
}

#[pyclass(subclass)]
pub struct OrbitalSolverResults {}

#[pymethods]
impl OrbitalSolverResults {
    pub fn rows(&self) -> PyResult<&Vec<Vec<f64>>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "rows() method not implemented",
        ))
    }

    pub fn cols(&self) -> PyResult<Vec<Vec<&f64>>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cols() method not implemented",
        ))
    }

    pub fn get_x(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_x() method not implemented",
        ))
    }
    pub fn get_y(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_y() method not implemented",
        ))
    }
    pub fn get_z(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_z() method not implemented",
        ))
    }
    pub fn get_vx(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_vx() method not implemented",
        ))
    }
    pub fn get_vy(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_vy() method not implemented",
        ))
    }
    pub fn get_vz(&self, index: usize) -> PyResult<Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_vz() method not implemented",
        ))
    }
    pub fn get_t(&self) -> PyResult<&Vec<f64>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_t() method not implemented",
        ))
    }
}
