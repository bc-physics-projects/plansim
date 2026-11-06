use differential_equations::prelude::*;
use nalgebra::DVector;

#[derive(Debug, Clone)]
pub struct NBodies<const N: usize> {
    pub mus: [f64; N],
}

impl<const N: usize> ODE<f64, DVector<f64>> for NBodies<N> {
    fn diff(&self, _t: f64, y: &DVector<f64>, dydt: &mut DVector<f64>) {
        let y = y.as_slice();
        let dydt = dydt.as_mut_slice();
        for i in 0..N {
            let base = i * 6;
            dydt[base..base + 3].copy_from_slice(&y[base + 3..base + 6]);
            dydt[base + 3..base + 6].fill(0.0);
        }
        // Each pair shares one distance and inverse cube, even for unequal masses.
        for i in 0..N {
            let a = i * 6;
            for j in i + 1..N {
                let b = j * 6;
                let dx = y[b] - y[a];
                let dy = y[b + 1] - y[a + 1];
                let dz = y[b + 2] - y[a + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let inverse_cube = 1.0 / (r2 * r2.sqrt());
                let fi = self.mus[j] * inverse_cube;
                let fj = self.mus[i] * inverse_cube;
                dydt[a + 3] += dx * fi;
                dydt[a + 4] += dy * fi;
                dydt[a + 5] += dz * fi;
                dydt[b + 3] -= dx * fj;
                dydt[b + 4] -= dy * fj;
                dydt[b + 5] -= dz * fj;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_direct_force_in_three_dimensions() {
        const N: usize = 15;
        let system = NBodies {
            mus: std::array::from_fn::<_, N, _>(|i| (i * i) as f64),
        };
        let y = DVector::from_fn(N * 6, |i, _| {
            let body = (i / 6 + 1) as f64;
            match i % 6 {
                0 => body * 2.0,
                1 => body.sin(),
                2 => body.cos(),
                _ => body * 0.1,
            }
        });
        let mut actual = DVector::zeros(N * 6);
        system.diff(0.0, &y, &mut actual);
        for i in 0..N {
            for k in 0..3 {
                assert_eq!(actual[i * 6 + k], y[i * 6 + k + 3]);
                let mut expected = 0.0;
                for j in 0..N {
                    if i == j {
                        continue;
                    }
                    let r2: f64 = (0..3)
                        .map(|axis| (y[j * 6 + axis] - y[i * 6 + axis]).powi(2))
                        .sum();
                    expected += (y[j * 6 + k] - y[i * 6 + k]) * system.mus[j] / r2.sqrt().powi(3);
                }
                assert!(
                    (actual[i * 6 + k + 3] - expected).abs() <= 1e-12 * expected.abs().max(1.0)
                );
            }
        }
        // Internal forces conserve total momentum (mu is proportional to mass).
        for k in 3..6 {
            let total: f64 = (0..N).map(|i| system.mus[i] * actual[i * 6 + k]).sum();
            assert!(total.abs() < 1e-10);
        }
    }
}
