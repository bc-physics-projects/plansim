use nalgebra::SMatrix;
use ode_solvers::*;

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub mus: [f64; N],
}

pub type State<const N: usize> = SMatrix<f64, 6, N>;
pub type Time = f64;

impl<const N: usize> System<f64, State<N>> for NBodies<N> {
    fn system(&self, _t: Time, y: &State<N>, dy: &mut State<N>) {
        let mut acc = Vector3::<f64>::zeros();
        let mut r_vec = Vector3::<f64>::zeros();

        for i in 0..N {
            acc.fill(0.0);

            for j in 0..N {
                if i != j {
                    for k in 0..3 {
                        r_vec[k] = y[(k, j)] - y[(k, i)];
                    }
                    let r = r_vec.norm();
                    acc += r_vec * (self.mus[j] / r.powi(3));
                }
            }

            for k in 0..3 {
                dy[(k, i)] = y[(k + 3, i)];
                dy[(k + 3, i)] = acc[k];
            }
        }
    }
}
