"""Run with unittest after building/installing the plansim extension."""

import unittest

import polars as pl
import plansim


class DataFrameResultsTests(unittest.TestCase):
    def test_nested_state_and_sample_times(self):
        # With zero gravity, separated bodies move at constant velocity.
        for count in (15, 90, 100):
            with self.subTest(count=count):
                solver = getattr(plansim, f"OrbitalSolver{count}")([0.0] * count)
                result = solver.solve(
                    [float(i + 1) for i in range(count)],
                    [0.0] * count,
                    [2.0] * count,
                    [0.0] * count,
                    (0.0, 1.0),
                    0.25,
                )
                self.assertIsInstance(result, pl.DataFrame)
                self.assertEqual(result.columns, ["t"] + list(map(str, range(count))))
                self.assertEqual(result["t"].to_list(), [0.0, 0.25, 0.5, 0.75, 1.0])
                self.assertEqual(result.schema["t"], pl.Float64)
                for body in range(count):
                    state = result[str(body)]
                    self.assertEqual(state.struct.fields, ["x", "y", "z", "vx", "vy", "vz"])
                    for field in state.struct.fields:
                        self.assertEqual(state.struct.field(field).dtype, pl.Float64)
                    for t, x in zip(result["t"], state.struct.field("x")):
                        self.assertAlmostEqual(x, body + 1 + 2 * t)
                    self.assertEqual(state.struct.field("vx").to_list(), [2.0] * result.height)
                    for field in ("y", "z", "vy", "vz"):
                        self.assertEqual(state.struct.field(field).to_list(), [0.0] * result.height)
        self.assertFalse(hasattr(plansim, "OrbitalSolverResults"))
        self.assertFalse(hasattr(plansim, "OrbitalSolverResults15"))


if __name__ == "__main__":
    unittest.main()
