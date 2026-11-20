use differential_equations::prelude::*;
use nalgebra::DVector;

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub mus: [f64; N],
}

pub type State<const N: usize> = DVector<f64>;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_direct_force_in_three_dimensions() {
        check_forces::<1>();
        check_forces::<2>();
        check_forces::<4>();
        check_forces::<5>();
        check_forces::<9>();
        check_forces::<10>();
        check_forces::<12>();
        check_forces::<15>();
        check_forces::<16>();
        check_forces::<17>();
        check_forces::<31>();
        check_forces::<32>();
        check_forces::<33>();
        check_forces::<90>();
        check_forces::<91>();
        check_forces::<99>();
        check_forces::<100>();
    }

    fn check_forces<const N: usize>() {
        let system = NBodies {
            mus: std::array::from_fn::<_, N, _>(|i| (i * i) as f64),
        };
        let y = DVector::from_fn(N * 6, |i, _| {
            let body = (i % N + 1) as f64;
            match i / N {
                0 => body * 2.0,
                1 => body.sin(),
                2 => body.cos(),
                component => body * 0.1 * component as f64,
            }
        });
        let mut actual = DVector::from_element(N * 6, f64::NAN);
        system.diff(0.0, &y, &mut actual);
        for i in 0..N {
            for k in 0..3 {
                assert_eq!(actual[k * N + i], y[(k + 3) * N + i]);
                let mut expected = 0.0;
                for j in 0..N {
                    if i == j {
                        continue;
                    }
                    let r2: f64 = (0..3)
                        .map(|axis| (y[axis * N + j] - y[axis * N + i]).powi(2))
                        .sum();
                    expected += (y[k * N + j] - y[k * N + i]) * system.mus[j] / r2.sqrt().powi(3);
                }
                assert!(
                    (actual[(k + 3) * N + i] - expected).abs() <= 1e-12 * expected.abs().max(1.0)
                );
            }
        }
        // Internal forces conserve total momentum (mu is proportional to mass).
        for k in 3..6 {
            let total: f64 = (0..N).map(|i| system.mus[i] * actual[k * N + i]).sum();
            let scale: f64 = (0..N)
                .map(|i| (system.mus[i] * actual[k * N + i]).abs())
                .sum();
            assert!(total.abs() <= 1e-12 * scale.max(1.0));
        }
    }
}
