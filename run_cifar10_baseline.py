"""CIFAR-10 FedAvg baseline, runnable on a desktop CPU or GPU with no Colab.

Run it in one of these ways (from the repository folder):
    python run_cifar10_baseline.py
    Jupyter: open this file, or in a notebook cell run:  %run run_cifar10_baseline.py

It does, in order:
    1. checks Python and installs the requirements if torch is missing
    2. runs the unit tests
    3. downloads CIFAR-10 (about 170 MB, once) and runs the baseline
    4. prints the summary and where the results are saved

Runs on a CPU in roughly 15 to 60 minutes depending on the machine (an estimate, not a measurement).
A GPU is used automatically if torch can see one.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "experiment_cifar10.yaml"
ALPHA = "0.1"  # Dirichlet concentration: 0.1 strongly non-IID, 0.5 moderate, 10 near-IID


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_python() -> None:
    step("1. Python and requirements")
    if sys.version_info < (3, 9):
        raise SystemExit(f"Python 3.9+ is needed; this is {sys.version.split()[0]}.")
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        print("torch and torchvision already installed")
    except ImportError:
        print("Installing requirements (torch can take a few minutes)...", flush=True)
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:], result.stderr[-2000:])
            raise SystemExit("pip install failed; see the output above.")
        print("requirements installed")


def run_tests() -> None:
    step("2. Unit tests")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
                            capture_output=True, text=True, cwd=ROOT)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit("Tests failed, so the baseline was not run. Fix the tests or send me this output.")


def run_baseline() -> Path:
    step("3. CIFAR-10 baseline (download on first run, then training)")
    sys.path.insert(0, str(ROOT / "experiments" / "baseline"))
    import run_baseline as rb  # experiments/baseline/run_baseline.py

    saved_argv = sys.argv
    sys.argv = ["run_cifar10_baseline.py", "--config", str(CONFIG), "--alpha", ALPHA]
    try:
        args = rb.parse_args()
    finally:
        sys.argv = saved_argv
    return rb.run(rb.build_config(args))


def show_results(run_dir: Path) -> None:
    import json

    step("4. Results")
    summary = json.loads((run_dir / "summary.json").read_text())
    energy = json.loads((run_dir / "energy.json").read_text()) if (run_dir / "energy.json").exists() else {}
    print(f"Final accuracy : {summary['final_accuracy']:.4f} (best {summary['best_accuracy']:.4f})")
    print(f"Simulated federation : {summary['sim_total_latency_s'] / 60:.1f} min, "
          f"{summary['sim_total_energy_kwh'] * 1000:.2f} Wh, {summary['sim_total_carbon_gco2e']:.2f} gCO2e")
    if energy:
        print(f"This machine's energy : {energy['energy_kwh'] * 1000:.2f} Wh, "
              f"CPU [{energy['cpu_source']}], GPU [{energy['gpu_source']}]")
    print(f"\nResults saved in: {run_dir}")


def main() -> None:
    check_python()
    run_tests()
    run_dir = run_baseline()
    show_results(run_dir)


if __name__ == "__main__":
    main()
