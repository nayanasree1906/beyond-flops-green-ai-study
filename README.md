# Beyond FLOPs: Green AI Model Performance and Energy Study

## About the Project

How much computational work does an AI model need, and how does that relate to its inference performance and energy consumption?

This project explores these questions by comparing four deep learning models under CPU-based inference: MobileNetV2, ResNet50, EfficientNet-B0, and ShuffleNetV2.

The idea behind this study is simple: FLOPs and parameter counts help describe a model's computational requirements, but they don't tell the whole story about its real-world performance or energy use.

## Models Compared

* MobileNetV2
* ResNet50
* EfficientNet-B0
* ShuffleNetV2

## Results

I compared the four models using parameter count, FLOPs, CPU inference
latency, and energy measurements.

| Model | Parameters | FLOPs | Mean Inference Time (ms) | Mean Energy per Inference (J) |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 5,288,548 | 415,145,024 | 36.36 | 0.897062 |
| MobileNetV2 | 3,504,872 | 327,486,720 | 23.33 | 0.619070 |
| ResNet50 | 25,557,032 | 4,133,742,592 | 81.72 | 1.796139 |
| ShuffleNetV2 | 2,278,604 | 152,708,792 | 23.35 | 0.631503 |

### Observations

- MobileNetV2 recorded the lowest mean inference time and mean energy
  per inference in these experiments.
- ShuffleNetV2 had the lowest parameter count and FLOPs.
- ResNet50 recorded the highest mean inference time and mean energy
  per inference.

These results reflect measurements from the tested environment.
Performance and energy consumption may differ across hardware,
software configurations, and experimental conditions.

## What I Measured

* **Model parameters:** The number of learnable parameters in each model.
* **FLOPs:** An estimate of the computational operations required for inference.
* **Inference latency:** The time taken for model inference on the CPU.
* **Energy measurements:** CPU energy-counter readings collected during experiments.

I also worked with repeated experimental runs and used Python to analyze the measurements and generate visualizations.

## Tools and Technologies

* Python
* PyTorch
* THOP
* Pandas and NumPy
* Matplotlib

## Repository Structure

* `final_green_ai_experiment.py` — Experiment and benchmarking script.
* `final_green_ai_analysis.py` — Analysis of experimental measurements and visualizations.
* `requirements.txt` — Python dependencies.
* `data/` — Data files.
* `docs/` — Project documentation.
* `graphs/` — Visualizations.

## What I'm Exploring

Through this project, I'm learning how to benchmark deep learning models, analyze experimental data, and look beyond theoretical computational complexity when studying AI efficiency.

## Limitations

The results come from experiments conducted in a specific hardware and software environment. Inference latency and CPU energy-counter measurements can vary with system conditions, so the findings should be interpreted within the context of the experimental setup.

This project is an exploration of model performance and energy measurement, not a claim that one model is universally the most energy-efficient.
