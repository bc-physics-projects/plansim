"""Run with unittest after building/installing the plansim extension."""

import unittest

import polars as pl

import plansim
from plansim.synthetic_solar_system_15 import DAY, initial_conditions


class DataFrameResultsTests(unittest.TestCase):
    def test_solar_daily_and_second_sampling_agree(self):
        mus, states = initial_conditions()
        solver = plansim.OrbitalSolver15(mus)
        daily = solver.solve(*states, (0.0, 1.0), 1.0)
        seconds = solver.solve(*states, (0.0, 1.0), 1.0 / DAY)
        self.assertEqual(daily.height, 2)
        # Repeated fractional-day steps may leave a tiny final interval;
        # the integrator then appends the exact endpoint as one extra row.
        self.assertIn(seconds.height, (86401, 86402))
        self.assertAlmostEqual(seconds["t"][-1], 1.0)
        self.assertAlmostEqual(seconds["t"][1], 1.0 / DAY)
        for body in range(15):
            for field in ("x", "y", "z", "vx", "vy", "vz"):
                a = daily[str(body)].struct.field(field)
                b = seconds[str(body)].struct.field(field)
                self.assertTrue(b.is_finite().all())
                self.assertAlmostEqual(a[-1], b[-1], delta=max(1e-8 * abs(a[-1]), 1e-6))
                self.assertEqual(a[0], b[0])

    def test_invalid_time_arguments(self):
        solver = plansim.OrbitalSolver15([0.0] * 15)
        arrays = ([float(i + 1) for i in range(15)],) + ([0.0] * 15,) * 5
        for span, spacing in (
            ((0.0, 1.0), 0.0),
            ((1.0, 0.0), 1.0),
            ((0.0, float("inf")), 1.0),
        ):
            with self.subTest(span=span, spacing=spacing), self.assertRaises(
                RuntimeError
            ):
                solver.solve(*arrays, span, spacing)

    def test_nested_state_and_sample_times(self):
        # With zero gravity, separated bodies move at constant velocity.
        for count in (15, 90, 100):
            with self.subTest(count=count):
                solver = getattr(plansim, f"OrbitalSolver{count}")([0.0] * count)
                result = solver.solve(
                    [float(i + 1) for i in range(count)],
                    [3.0] * count,
                    [5.0] * count,
                    [2.0] * count,
                    [-4.0] * count,
                    [6.0] * count,
                    (0.0, 1.0),
                    0.25,
                )
                self.assertIsInstance(result, pl.DataFrame)
                self.assertEqual(result.columns, ["t"] + list(map(str, range(count))))
                self.assertEqual(result["t"].to_list(), [0.0, 0.25, 0.5, 0.75, 1.0])
                self.assertEqual(result.schema["t"], pl.Float64)
                for body in range(count):
                    state = result[str(body)]
                    self.assertEqual(
                        state.struct.fields, ["x", "y", "z", "vx", "vy", "vz"]
                    )
                    for field in state.struct.fields:
                        self.assertEqual(state.struct.field(field).dtype, pl.Float64)
                    for t, x in zip(result["t"], state.struct.field("x")):
                        self.assertAlmostEqual(x, body + 1 + 2 * t)
                    self.assertEqual(
                        state.struct.field("vx").to_list(), [2.0] * result.height
                    )
                    for field, initial, velocity in (
                        ("y", 3.0, -4.0), ("z", 5.0, 6.0)
                    ):
                        for t, value in zip(result["t"], state.struct.field(field)):
                            self.assertAlmostEqual(value, initial + velocity * t)
                    for field, velocity in (("vy", -4.0), ("vz", 6.0)):
                        self.assertEqual(
                            state.struct.field(field).to_list(), [velocity] * result.height
                        )
        self.assertFalse(hasattr(plansim, "OrbitalSolverResults"))
        self.assertFalse(hasattr(plansim, "OrbitalSolverResults15"))


if __name__ == "__main__":
    unittest.main()
