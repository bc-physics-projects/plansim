use ode_solvers::*;

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub Ms: [f64; N],
}

pub type State<const N: usize> = SVector<f64, { N * 6 }>;
pub type Time = f64;

impl<const N: usize> System<f64, State<N>> for NBodies<N> {
    fn system(&self, _t: Time, y: &State<N>, dy: &mut State<N>) {
        for i in 0..N {
            let mut acc = Vector3::zeros();
            for j in 0..N {
                if i != j {
                    let r_vec = y.fixed_rows::<3>(j * 6) - y.fixed_rows::<3>(i * 6);
                    let r = r_vec.norm();
                    acc += self.Ms[j] * r_vec / r.powi(3);
                }
            }
            dy.fixed_rows_mut::<3>(i * 6)
                .copy_from(&y.fixed_rows::<3>(i * 6 + 3));
            dy.fixed_rows_mut::<3>(i * 6 + 3).copy_from(&acc);
        }
    }
}
