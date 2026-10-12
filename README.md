# Tensor-KAN

Tensor-KAN provides a PyTorch implementation of the Tensor
Kolmogorov--Arnold Neural Network (TKANN) for elliptic interface problems. The
initial reproducibility release contains the core model and optimizer code
together with one self-contained Gaussian-mixture interface example.

Only source code needed to train and evaluate this example is included. The
repository does not contain generated results, pretrained checkpoints, or
cluster-specific submission scripts.

## Highlights

- Coordinate-wise one-dimensional KAN encoders with a low-rank product-sum
  decoder.
- Interface-aware trial function
  \(u(\boldsymbol{x})=U(\boldsymbol{x})+s(\boldsymbol{x})V(\boldsymbol{x})\).
- Bulk, boundary, solution-jump, and flux-jump residuals.
- AdamW, L-BFGS, Levenberg--Marquardt (LM), and geodesic-damped LM (GDLM)
  training options.
- An optimized separable derivative path for TKANN.
- Double-precision and deterministic-seed controls for reproducible numerical
  experiments.

## Repository layout

    Tensor-KAN/
    ├── module/
    │   ├── gradient.py
    │   ├── geodesiclm_v1.py
    │   ├── kan.py
    │   ├── model.py
    │   ├── optimizer.py
    │   ├── sampler.py
    │   ├── tenkan_derivatives.py
    │   ├── utils.py
    │   └── visualize.py
    ├── ex2/
    │   ├── train.py
    │   ├── data_gen.py
    │   ├── loss_func.py
    │   └── evaluate.py
    ├── requirements.txt
    └── README.md

**ex2** is the Gaussian-mixture benchmark corresponding to Example 4.2 in the
manuscript. In the command-line interface, **TenKAN** denotes the TKANN model.

## Environment

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for
your CPU or CUDA platform first, and then install the remaining dependencies:

    python -m pip install -r requirements.txt

The code has been validated with Python 3.11.4, PyTorch 2.0.1+cu118,
NumPy 1.24.3, SciPy 1.10.1, Matplotlib 3.7.1, Numba 0.57.0,
scikit-optimize 0.9.0, and pyDOE 0.3.8.

## Quick start

Run commands from the repository root. The following small CPU-compatible
configuration is useful for checking the installation; it is not intended to
reproduce the reported accuracy:

    python ex2/train.py \
      --model-type TenKAN \
      --optimizer-type adam \
      --iterations 1 \
      --k 3 \
      --grid-size 3 \
      --spline-order 2 \
      --rank 2 \
      --kan-hidden 2 \
      --kan-depth 1 \
      --n-inner 32 \
      --n-boundary 16 \
      --n-interface 8 \
      --interface-candidates 10000 \
      --test-points 100 \
      --eval-batch-size 64 \
      --tenkan-derivatives separable \
      --no-plots

## Reproduce Example 2

The following command reproduces the representative \(K=3\) TKANN-GDLM
configuration:

    python ex2/train.py \
      --model-type TenKAN \
      --optimizer-type gdlm \
      --iterations 1600 \
      --k 3 \
      --grid-size 15 \
      --spline-order 6 \
      --rank 5 \
      --kan-hidden 5 \
      --kan-depth 2 \
      --tenkan-derivatives separable \
      --dtype float64 \
      --seed 1234 \
      --n-inner 2100 \
      --n-boundary 160 \
      --n-interface 160

CUDA is used automatically when available. By default, each run is written to
a timestamped directory under **ex2/results/**; generated result directories
are excluded from version control.

## Main options

| Option | Meaning | Default |
|---|---|---:|
| --model-type | Neural architecture (TenKAN selects TKANN) | TenKAN |
| --optimizer-type | adam, lbfgs, lm, or gdlm | gdlm |
| --iterations | Optimizer iteration budget | optimizer dependent |
| --k | Gaussian-mixture/interface complexity parameter | 5 |
| --grid-size | KAN spline-grid resolution \(M\) | 15 |
| --spline-order | Spline order | 6 |
| --rank | Separable rank \(p\) | 5 |
| --tenkan-derivatives | TKANN derivative implementation | separable |
| --n-inner | Number of interior collocation points | 2100 |
| --n-boundary | Number of boundary points | 160 |
| --n-interface | Number of interface points | 160 |
| --seed | Random seed | 1234 |
| --dtype | Floating-point precision | float64 |
| --output-root | Parent directory for generated results | ex2/results |

Use **python ex2/train.py --help** for the complete option list.

## Outputs

A standard run saves the trained state and numerical summaries, including
**model.pt**, **metrics.json**, **metrics.csv**, and **error_metrics.txt**.
Diagnostic figures are generated unless **--no-plots** is specified.

Exact wall-clock times can vary with hardware, CUDA, and library versions.
Reported experiments use seed 1234 and double precision unless stated
otherwise.

## Citation

If this implementation contributes to your work, please cite the associated
manuscript:

> *A Tensor Kolmogorov--Arnold Neural Network for Solving Elliptic Interface
> Problems* (manuscript under review).

The LM/GDLM implementation follows the geodesic-acceleration formulation
described by M. K. Transtrum and J. P. Sethna, *Improvements to the
Levenberg--Marquardt algorithm for nonlinear least-squares minimization*,
arXiv:1201.5885.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
