"""Plot Horizons 15-body solver times for increasing simulation durations."""

from concurrent.futures import ThreadPoolExecutor
from itertools import count
from time import perf_counter

import matplotlib.pyplot as plt

from horizons_solar_system_15 import initial_conditions, plansim


START_DAYS = 1_000
DAY_STEP = 5_000
STOP_SECONDS = 30


def time_solve(mus, init_data, days):
    solver = plansim.OrbitalSolver15(mus)
    start = perf_counter()
    result = solver.solve(*init_data, (0.0, days), 1.0)
    elapsed = perf_counter() - start
    if result["t"][-1] != days:
        raise RuntimeError(f"Solver stopped before reaching {days:,} days.")
    return elapsed


def main():
    mus, init_data = initial_conditions()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Horizons 15-body benchmark")
    (line,) = ax.plot([], [], marker="o")
    ax.set(xlabel="Simulation duration (days)", ylabel="Solver time (seconds)")
    ax.grid(alpha=0.25)
    plt.show(block=False)

    durations = []
    timings = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for days in count(START_DAYS, DAY_STEP):
            print(f"Running {days:,} days...", flush=True)
            ax.set_title(f"Running {days:,} days")
            fig.canvas.draw_idle()
            run = executor.submit(time_solve, mus, init_data, days)
            while not run.done() and plt.fignum_exists(fig.number):
                plt.pause(0.1)
            if not plt.fignum_exists(fig.number):
                return

            elapsed = run.result()
            durations.append(days)
            timings.append(elapsed)
            print(f"{days:,} days took {elapsed:.3f} seconds.", flush=True)
            line.set_data(durations, timings)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            if elapsed > STOP_SECONDS:
                ax.set_title(f"Cutoff exceeded at {days:,} days")
                break

    plt.show()


if __name__ == "__main__":
    main()
