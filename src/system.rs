use nalgebra::SMatrix;
use ode_solvers::*;

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub Ms: [f64; N],
}

pub type State<const N: usize> = SMatrix<f64, 6, N>;
pub type Time = f64;

impl<const N: usize> System<f64, State<N>> for NBodies<N> {
    fn system(&self, _t: Time, y: &State<N>, dy: &mut State<N>) {
        for i in 0..N {
            let mut acc = Vector3::zeros();

            for j in 0..N {
                if i != j {
                    let r_vec = Vector3::new(
                        y[(0, j)] - y[(0, i)],
                        y[(1, j)] - y[(1, i)],
                        y[(2, j)] - y[(2, i)],
                    );
                    let r = r_vec.norm();
                    acc += r_vec * (self.Ms[j] / r.powi(3));
                }
            }

            for k in 0..3 {
                dy[(k, i)] = y[(k + 3, i)];
                dy[(k + 3, i)] = acc[k];
            }
        }
    }
}
