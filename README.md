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
