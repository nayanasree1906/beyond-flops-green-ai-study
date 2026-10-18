import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FINAL GREEN AI ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1. RAW V3 EXPERIMENT DATA
# ------------------------------------------------------------

data = [
    # MobileNetV2
    ["MobileNetV2", 1, 3504872, 327486720, 23.4382718, 332.0667370008, 0.6641334740016],
    ["MobileNetV2", 2, 3504872, 327486720, 22.6717580, 297.5992970004, 0.5951985940008],
    ["MobileNetV2", 3, 3504872, 327486720, 23.2029724, 314.7708579984, 0.6295417159968],
    ["MobileNetV2", 4, 3504872, 327486720, 22.3204668, 307.8391839984, 0.6156783679968],
    ["MobileNetV2", 5, 3504872, 327486720, 25.0216388, 295.3993320000, 0.5907986640000],

    # ResNet50 — ONLY VALID ORIGINAL TRIALS 1-4
    ["ResNet50", 1, 25557032, 4133742592, 75.2525522, 807.9770860020, 1.6159541720040],
    ["ResNet50", 2, 25557032, 4133742592, 82.9531814, 882.0151040016, 1.7640302080032],
    ["ResNet50", 3, 25557032, 4133742592, 80.3111014, 871.6884750000, 1.7433769500000],
    ["ResNet50", 4, 25557032, 4133742592, 81.8572596, 935.9525239992, 1.8719050480000],

    # ResNet50 — REPLACEMENT TRIAL
    ["ResNet50", 6, 25557032, 4133742592, 88.2095, 992.7132070000, 1.9854264140000],

    # EfficientNet-B0
    ["EfficientNet-B0", 1, 5288548, 415145024, 42.6440276, 534.5012149992, 1.0690024299984],
    ["EfficientNet-B0", 2, 5288548, 415145024, 35.2640240, 432.9919079999, 0.8659838160000],
    ["EfficientNet-B0", 3, 5288548, 415145024, 35.3344864, 438.2901239976, 0.8765802479952],
    ["EfficientNet-B0", 4, 5288548, 415145024, 37.0710392, 438.5615739984, 0.8771231479968],
    ["EfficientNet-B0", 5, 5288548, 415145024, 31.4999008, 398.3096869992, 0.7966193739984],

    # ShuffleNetV2
    ["ShuffleNetV2", 1, 2278604, 152708792, 20.4723660, 280.2627309996, 0.5605254619992],
    ["ShuffleNetV2", 2, 2278604, 152708792, 22.8181080, 312.9234120000, 0.6258468240000],
    ["ShuffleNetV2", 3, 2278604, 152708792, 24.6851562, 330.9555000024, 0.6619110000048],
    ["ShuffleNetV2", 4, 2278604, 152708792, 25.5138594, 328.4491319999, 0.6568982640000],
    ["ShuffleNetV2", 5, 2278604, 152708792, 23.2632302, 326.1678540012, 0.6523357080024],
]

columns = [
    "Model",
    "Trial",
    "Parameters",
    "FLOPs",
    "Inference_Time_ms",
    "Energy_J",
    "Energy_per_Inference_J"
]

df = pd.DataFrame(data, columns=columns)

# ------------------------------------------------------------
# 2. CALCULATE FINAL SUMMARY
# ------------------------------------------------------------

summary = (
    df.groupby("Model")
    .agg(
        Parameters=("Parameters", "first"),
        FLOPs=("FLOPs", "first"),

        Mean_Inference_Time_ms=("Inference_Time_ms", "mean"),
        Std_Inference_Time_ms=("Inference_Time_ms", "std"),

        Mean_Energy_J=("Energy_J", "mean"),
        Std_Energy_J=("Energy_J", "std"),

        Mean_Energy_per_Inference_J=("Energy_per_Inference_J", "mean"),
        Std_Energy_per_Inference_J=("Energy_per_Inference_J", "std"),
    )
    .reset_index()
)

# Calculate average package power from energy/time
summary["Mean_Package_Power_W"] = (
    summary["Mean_Energy_J"] /
    (summary["Mean_Inference_Time_ms"] / 1000 * 500)
)

# ------------------------------------------------------------
# 3. SAVE FINAL CLEAN DATA
# ------------------------------------------------------------

df.to_csv("green_ai_FINAL_raw_data.csv", index=False)
summary.to_csv("green_ai_FINAL_summary.csv", index=False)

print("\n" + "=" * 70)
print("FINAL GREEN AI EXPERIMENT SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

# ------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# ------------------------------------------------------------

correlation_data = summary[
    [
        "FLOPs",
        "Parameters",
        "Mean_Inference_Time_ms",
        "Mean_Energy_per_Inference_J"
    ]
]

correlations = correlation_data.corr(method="pearson")

print("\n" + "=" * 70)
print("PEARSON CORRELATION MATRIX")
print("=" * 70)

print(correlations)

# Specific correlations
print("\n" + "=" * 70)
print("KEY CORRELATIONS")
print("=" * 70)

print(
    f"FLOPs vs Energy: "
    f"{correlations.loc['FLOPs', 'Mean_Energy_per_Inference_J']:.4f}"
)

print(
    f"Parameters vs Energy: "
    f"{correlations.loc['Parameters', 'Mean_Energy_per_Inference_J']:.4f}"
)

print(
    f"Inference Time vs Energy: "
    f"{correlations.loc['Mean_Inference_Time_ms', 'Mean_Energy_per_Inference_J']:.4f}"
)

# ------------------------------------------------------------
# 5. FINAL GRAPH FUNCTION
# ------------------------------------------------------------

def make_scatter_plot(
    x_column,
    y_column,
    xlabel,
    ylabel,
    title,
    filename,
    log_x=False
):

    plt.figure(figsize=(9, 6))

    x = summary[x_column].to_numpy()
    y = summary[y_column].to_numpy()
    labels = summary["Model"].to_numpy()

    # --------------------------------------------------------
    # Scatter points
    # --------------------------------------------------------

    plt.scatter(
        x,
        y,
        s=120
    )

    # --------------------------------------------------------
    # Model labels
    # --------------------------------------------------------

    for xi, yi, label in zip(x, y, labels):

        plt.annotate(
            label,
            (xi, yi),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10
        )

    # --------------------------------------------------------
    # Trendline
    # --------------------------------------------------------

    if log_x:

        # Fit trendline against log10(x)
        log_x_values = np.log10(x)

        coefficients = np.polyfit(
            log_x_values,
            y,
            1
        )

        x_line_log = np.linspace(
            log_x_values.min(),
            log_x_values.max(),
            200
        )

        y_line = (
            coefficients[0] * x_line_log
            + coefficients[1]
        )

        x_line = 10 ** x_line_log

        plt.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.5
        )

        # Logarithmic x-axis
        plt.xscale("log")

    else:

        # Normal linear trendline
        coefficients = np.polyfit(
            x,
            y,
            1
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200
        )

        y_line = (
            coefficients[0] * x_line
            + coefficients[1]
        )

        plt.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.5
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.xlabel(
        xlabel,
        fontsize=12
    )

    plt.ylabel(
        ylabel,
        fontsize=12
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ------------------------------------------------------------
# 6. CREATE FINAL RESEARCH GRAPHS
# ------------------------------------------------------------

# Graph 1 — FLOPs vs Energy
make_scatter_plot(
    "FLOPs",
    "Mean_Energy_per_Inference_J",
    "FLOPs",
    "Energy per inference (J)",
    "FLOPs vs Energy Consumption",
    "01_FLOPs_vs_Energy.png",
    log_x=True
)


# Graph 2 — Parameters vs Energy
make_scatter_plot(
    "Parameters",
    "Mean_Energy_per_Inference_J",
    "Number of parameters",
    "Energy per inference (J)",
    "Parameter Count vs Energy Consumption",
    "02_Parameters_vs_Energy.png",
    log_x=True
)


# Graph 3 — Inference Time vs Energy
make_scatter_plot(
    "Mean_Inference_Time_ms",
    "Mean_Energy_per_Inference_J",
    "Average inference time (ms)",
    "Energy per inference (J)",
    "Inference Time vs Energy Consumption",
    "03_Time_vs_Energy.png",
)


# ------------------------------------------------------------
# 7. FILES CREATED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("green_ai_FINAL_raw_data.csv")
print("green_ai_FINAL_summary.csv")
print("01_FLOPs_vs_Energy.png")
print("02_Parameters_vs_Energy.png")
print("03_Time_vs_Energy.png")

print("\nAnalysis complete!")