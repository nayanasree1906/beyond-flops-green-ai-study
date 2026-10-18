import time
import subprocess
import torch
import torchvision.models as models
from thop import profile
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

INPUT = torch.randn(1, 3, 224, 224)

WARMUP_RUNS = 20
TEST_RUNS = 500
TRIALS = 5

ENERGY_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Energy"


# ============================================================
# ENERGY READING
# ============================================================

def read_energy_counter():
    command = [
        "powershell",
        "-Command",
        f'Get-Counter "{ENERGY_COUNTER}" | '
        "Select-Object -ExpandProperty CounterSamples | "
        "Select-Object -ExpandProperty CookedValue"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return float(result.stdout.strip())


def pwh_to_joules(pwh):
    """
    Convert picowatt-hours to joules.

    1 pWh = 1e-12 Wh
    1 Wh = 3600 J
    """

    return pwh * 1e-12 * 3600


# ============================================================
# MODELS
# ============================================================

model_dict = {
    "MobileNetV2": models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    ),

    "ResNet50": models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    ),

    "EfficientNet-B0": models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    ),

    "ShuffleNetV2": models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
    )
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN EXPERIMENT
# ============================================================

for model_name, model in model_dict.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    model = model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    flops, _ = profile(
        model,
        inputs=(INPUT,),
        verbose=False
    )

    print(f"Parameters : {parameters:,}")
    print(f"FLOPs      : {flops:,.0f}")

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print("Warming up...")

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(INPUT)

    # --------------------------------------------------------
    # TRIALS
    # --------------------------------------------------------

    for trial in range(1, TRIALS + 1):

        print(f"\nTrial {trial}/{TRIALS}")

        # Small pause between trials
        time.sleep(3)

        # ----------------------------------------------------
        # BASELINE ENERGY
        # ----------------------------------------------------

        baseline_start_energy = read_energy_counter()
        baseline_start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(TEST_RUNS):
                pass

        baseline_end_time = time.perf_counter()
        baseline_end_energy = read_energy_counter()

        baseline_energy_pwh = (
            baseline_end_energy -
            baseline_start_energy
        )

        baseline_time = (
            baseline_end_time -
            baseline_start_time
        )

        # ----------------------------------------------------
        # MODEL RUN
        # ----------------------------------------------------

        energy_before = read_energy_counter()

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(TEST_RUNS):
                model(INPUT)

        end_time = time.perf_counter()

        energy_after = read_energy_counter()

        # ----------------------------------------------------
        # CALCULATIONS
        # ----------------------------------------------------

        total_time = end_time - start_time

        avg_time_ms = (
            total_time / TEST_RUNS
        ) * 1000

        raw_energy_pwh = (
            energy_after -
            energy_before
        )

        # Background energy during equivalent baseline period
        baseline_power_pwh_per_sec = (
            baseline_energy_pwh / baseline_time
            if baseline_time > 0
            else 0
        )

        estimated_background_energy_pwh = (
            baseline_power_pwh_per_sec *
            total_time
        )

        net_energy_pwh = (
            raw_energy_pwh -
            estimated_background_energy_pwh
        )

        raw_energy_joules = pwh_to_joules(
            raw_energy_pwh
        )

        net_energy_joules = pwh_to_joules(
            net_energy_pwh
        )

        energy_per_inference_joules = (
            net_energy_joules / TEST_RUNS
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "Trial": trial,
            "Parameters": parameters,
            "FLOPs": flops,
            "Inference_Count": TEST_RUNS,
            "Total_Time_s": total_time,
            "Avg_Inference_Time_ms": avg_time_ms,
            "Raw_Energy_pWh": raw_energy_pwh,
            "Background_Energy_pWh": estimated_background_energy_pwh,
            "Net_Energy_pWh": net_energy_pwh,
            "Raw_Energy_J": raw_energy_joules,
            "Net_Energy_J": net_energy_joules,
            "Energy_per_Inference_J": energy_per_inference_joules
        })

        print(
            f"Average inference time : "
            f"{avg_time_ms:.4f} ms"
        )

        print(
            f"Raw energy             : "
            f"{raw_energy_joules:.6f} J"
        )

        print(
            f"Net energy             : "
            f"{net_energy_joules:.6f} J"
        )

        print(
            f"Energy / inference     : "
            f"{energy_per_inference_joules:.9f} J"
        )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    "green_ai_final_raw_results.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

summary = (
    df.groupby("Model")
    .agg({
        "Parameters": "first",
        "FLOPs": "first",
        "Avg_Inference_Time_ms": ["mean", "std"],
        "Net_Energy_J": ["mean", "std"],
        "Energy_per_Inference_J": ["mean", "std"]
    })
)

summary.to_csv(
    "green_ai_final_summary.csv"
)

print("\n\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print("\nRaw results saved to:")
print("green_ai_final_raw_results.csv")

print("\nSummary saved to:")
print("green_ai_final_summary.csv")

print("\nSummary:")
print(summary)