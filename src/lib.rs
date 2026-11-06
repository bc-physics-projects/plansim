#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use nalgebra::DVector;
use ode_solvers::System;
use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::dop853::*;
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
                        self.inner.solve(rs, r_angs, vs, v_angs, tspan, tspace).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                        let (tpoints, data) = self.inner
                        .results()
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No results available")
                        })?;
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
    stepper: Option<Dop853<f64, DVector<f64>, FlatNBodies<N>>>,
    output: Option<ContinuousOutputModel<f64, DVector<f64>>>,
}

impl<const N: usize> GenericSolver<N>
where
    [(); N * 6]:,
{
    pub fn new(mus: [f64; N]) -> Self {
        Self {
            data: system::NBodies { mus: mus },
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
        tspace: f64,
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

        let mut stepper = Dop853::new(
            FlatNBodies {
                inner: self.data.clone(),
            },
            tspan.0,
            tspan.1,
            tspace,
            y0,
            1.0e-10,
            1.0e-10,
        );

        // let mut continuous_output_model = ContinuousOutputModel::default();
        // let res = stepper
        //     .integrate_with_continuous_output_model(&mut continuous_output_model)
        //     .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let res = stepper.integrate()?;
        if res.accepted_steps > 0 {
            self.stepper = Some(stepper);
            // self.output = Some(continuous_output_model);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Integration failed"))
        }
    }

    pub fn results(&self) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        self.stepper.as_ref().map(|stepper| {
            let y = stepper.results();
            let (ts, states) = y.get();
            let x: Vec<Vec<f64>> = states.iter().map(|s| s.as_slice().to_vec()).collect();
            (ts.to_vec(), x)
        })
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
