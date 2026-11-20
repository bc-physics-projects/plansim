use differential_equations::prelude::*;
use nalgebra::{DVector, SVector};

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub mus: [f64; N],
}

pub type State<const N: usize> = SVector<f64, { 6 * N }>;

impl<const N: usize> ODE<f64, State<N>> for NBodies<N> {
    fn diff(&self, _t: f64, y: &State<N>, dydt: &mut State<N>) {
        let y = y.as_slice();
        let dydt = dydt.as_mut_slice();
        // SoA: x, y, z, vx, vy, vz, each with N contiguous values.
        let (positions, velocities) = y.split_at(3 * N);
        let (rates, accelerations) = dydt.split_at_mut(3 * N);
        rates.copy_from_slice(velocities);
        accelerations.fill(0.0);
        let (xs, rest) = positions.split_at(N);
        let (ys, zs) = rest.split_at(N);
        let (axs, rest) = accelerations.split_at_mut(N);
        let (ays, azs) = rest.split_at_mut(N);
        // Explicit four-lane batches avoid relying on auto-vectorization of
        // pair loops of any size. Scalar tails and ordered sums preserve pair order.
        use std::simd::{StdFloat, f64x4};
        for i in 0..N {
            let mut ax_i = axs[i];
            let mut ay_i = ays[i];
            let mut az_i = azs[i];
            let mut j = i + 1;
            while j + 4 <= N {
                let dx = f64x4::from_slice(&xs[j..]) - f64x4::splat(xs[i]);
                let dy = f64x4::from_slice(&ys[j..]) - f64x4::splat(ys[i]);
                let dz = f64x4::from_slice(&zs[j..]) - f64x4::splat(zs[i]);
                let r2 = dx * dx + dy * dy + dz * dz;
                let inv = f64x4::splat(1.0) / (r2 * r2.sqrt());
                let fi = f64x4::from_slice(&self.mus[j..]) * inv;
                let fj = f64x4::splat(self.mus[i]) * inv;
                let fx = (dx * fi).to_array();
                let fy = (dy * fi).to_array();
                let fz = (dz * fi).to_array();
                (f64x4::from_slice(&axs[j..]) - dx * fj).copy_to_slice(&mut axs[j..j + 4]);
                (f64x4::from_slice(&ays[j..]) - dy * fj).copy_to_slice(&mut ays[j..j + 4]);
                (f64x4::from_slice(&azs[j..]) - dz * fj).copy_to_slice(&mut azs[j..j + 4]);
                for lane in 0..4 {
                    ax_i += fx[lane];
                    ay_i += fy[lane];
                    az_i += fz[lane];
                }
                j += 4;
            }
            for j in j..N {
                let dx = xs[j] - xs[i];
                let dy = ys[j] - ys[i];
                let dz = zs[j] - zs[i];
                let r2 = dx * dx + dy * dy + dz * dz;
                let inv = 1.0 / (r2 * r2.sqrt());
                let fi = self.mus[j] * inv;
                let fj = self.mus[i] * inv;
                ax_i += dx * fi;
                ay_i += dy * fi;
                az_i += dz * fi;
                axs[j] -= dx * fj;
                ays[j] -= dy * fj;
                azs[j] -= dz * fj;
            }
            axs[i] = ax_i;
            ays[i] = ay_i;
            azs[i] = az_i;
        }
    }
}

#[cfg(feature = "rectangular-kernel")]
impl<const N: usize> ODE<f64, State<N>> for NBodies<N> {
    fn diff(&self, _t: f64, y: &State<N>, dydt: &mut State<N>) {
        let y = y.as_slice();
        let dydt = dydt.as_mut_slice();
        // SoA: x, y, z, vx, vy, vz, each with N contiguous values.
        let (positions, velocities) = y.split_at(3 * N);
        let (rates, accelerations) = dydt.split_at_mut(3 * N);
        rates.copy_from_slice(velocities);
        let (xs, rest) = positions.split_at(N);
        let (ys, zs) = rest.split_at(N);
        let (axs, rest) = accelerations.split_at_mut(N);
        let (ays, azs) = rest.split_at_mut(N);
        // Each row accumulates independently in SIMD registers. Mask the self
        // lane before division so its zero separation never produces NaN.
        use std::simd::{Mask, Select, StdFloat, f64x4, num::SimdFloat};
        for i in 0..N {
            let mut ax = f64x4::splat(0.0);
            let mut ay = f64x4::splat(0.0);
            let mut az = f64x4::splat(0.0);
            let mut j = 0;
            while j + 4 <= N {
                let dx = f64x4::from_slice(&xs[j..]) - f64x4::splat(xs[i]);
                let dy = f64x4::from_slice(&ys[j..]) - f64x4::splat(ys[i]);
                let dz = f64x4::from_slice(&zs[j..]) - f64x4::splat(zs[i]);
                let active = Mask::<i64, 4>::from_array(std::array::from_fn(|lane| j + lane != i));
                let r2 = active.select(dx * dx + dy * dy + dz * dz, f64x4::splat(1.0));
                let mu = active.select(f64x4::from_slice(&self.mus[j..]), f64x4::splat(0.0));
                let force = mu / (r2 * r2.sqrt());
                ax += dx * force;
                ay += dy * force;
                az += dz * force;
                j += 4;
            }
            let mut ax_i = ax.reduce_sum();
            let mut ay_i = ay.reduce_sum();
            let mut az_i = az.reduce_sum();
            for j in j..N {
                if i == j {
                    continue;
                }
                let dx = xs[j] - xs[i];
                let dy = ys[j] - ys[i];
                let dz = zs[j] - zs[i];
                let r2 = dx * dx + dy * dy + dz * dz;
                let force = self.mus[j] / (r2 * r2.sqrt());
                ax_i += dx * force;
                ay_i += dy * force;
                az_i += dz * force;
            }
            axs[i] = ax_i;
            ays[i] = ay_i;
            azs[i] = az_i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<const N: usize>()
    where
        [(); 6 * N]:,
    {
        let b = NBodies {
            mus: std::array::from_fn::<_, N, _>(|i| (i * i) as f64),
        };
        let y: Vec<f64> = (0..6 * N)
            .map(|i| {
                let x = (i % N + 1) as f64;
                match i / N {
                    0 => x * 2.,
                    1 => x.sin(),
                    2 => x.cos(),
                    _ => x * 0.1,
                }
            })
            .collect();
        let y = State::<N>::from_row_slice(&y);
        let mut a = State::<N>::from_element(f64::NAN);
        b.diff(0.0, &y, &mut a);
        for i in 0..N {
            for k in 0..3 {
                assert_eq!(a[k * N + i], y[(k + 3) * N + i]);
                let mut expected = 0.;
                for j in 0..N {
                    if i == j {
                        continue;
                    }
                    let r2: f64 = (0..3).map(|c| (y[c * N + j] - y[c * N + i]).powi(2)).sum();
                    expected += (y[k * N + j] - y[k * N + i]) * b.mus[j] / r2.sqrt().powi(3);
                }
                assert!((a[(k + 3) * N + i] - expected).abs() <= 1e-12 * expected.abs().max(1.));
            }
        }
        for k in 3..6 {
            let total: f64 = (0..N).map(|i| b.mus[i] * a[k * N + i]).sum();
            let scale: f64 = (0..N).map(|i| (b.mus[i] * a[k * N + i]).abs()).sum();
            assert!(total.abs() <= 1e-12 * scale.max(1.));
        }
    }

    #[test]
    fn kernel_matches_direct_force() {
        check::<1>();
        check::<2>();
        check::<4>();
        check::<5>();
        check::<9>();
        check::<15>();
        check::<16>();
        check::<17>();
        check::<90>();
        check::<91>();
        check::<99>();
        check::<100>();
    }
}
