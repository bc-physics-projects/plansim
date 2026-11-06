use differential_equations::prelude::*;
use nalgebra::{DVector, Vector3};

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub mus: [f64; N],
}

impl<const N: usize> ODE<f64, DVector<f64>> for NBodies<N> {
    fn diff(&self, _t: f64, y: &DVector<f64>, dydt: &mut DVector<f64>) {
        let mut acc = Vector3::<f64>::zeros();
        let mut r_vec = Vector3::<f64>::zeros();

        for i in 0..N {
            acc.fill(0.0);

            for j in 0..N {
                if i != j {
                    for k in 0..3 {
                        r_vec[k] = y[j * 6 + k] - y[i * 6 + k];
                    }
                    let r = r_vec.norm();
                    acc += r_vec * (self.mus[j] / r.powi(3));
                }
            }

            for k in 0..3 {
                dydt[i * 6 + k] = y[i * 6 + k + 3];
                dydt[i * 6 + k + 3] = acc[k];
            }
        }
    }
}
