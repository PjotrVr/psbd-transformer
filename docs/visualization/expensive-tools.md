# The 3 expensive visualisation tools

BackdoorBench's analysis module ships 3 scripts whose cost is minutes to hours per checkpoint rather than seconds: the Hessian eigenvalue spectrum (`visual_hessian.py`), the 2-D loss landscape (`visual_landscape.py`) and feature visualisation (`visual_fv.py`). All 3 lean on packages this project does not carry (`pyhessian`, `mpi4py` with an unvendored clone of the loss-landscape repository, `omnixai`), and 2 of the 3 do not run upstream as shipped. This page records, for each tool, the statistic, the source paper, what the upstream script does, how the statistic maps onto a Vision Transformer, every deviation of the port and the cost measured on the Supek login node's A100. The ports live in `analysis/curvature.py`, `analysis/landscape.py` and `analysis/synthesis.py`, their figures in `visualization/spectra.py` and `visualization/surfaces.py`, and the 3 registry entries in `visualization/expensive_tools.py` under the names `hessian`, `landscape` and `feature_visualization`.

Every tool reads the loaded `VisualCase` of `cli.visualize`: the checkpoint's model, the 3 PSBD splits built by the standard permutation, its `args.json` metadata and the dataset's normalisation. The `--view` argument names the splits a tool reads. `paired` is the clean split and the backdoor split of the same rows, `clean_test` and `bd_test` are 1 of the 2, and `mixed` is half the rows of each in 1 batch. The backdoor split carries the attack-success label, so a loss on it is the attacker's objective, which is the quantity a loss surface or a curvature on triggered inputs is about. Each tool writes `results/<folder>/visual/<tool>.pdf`, `.png` and `.json`, the sidecar holding every plotted array, the settings and the git commit, and prints its wall time.

## Hessian spectrum

The statistic is the eigenvalue distribution of the loss Hessian over the network's parameters at 1 batch, summarised by its 2 largest eigenvalues and drawn as a smoothed spectral density. The source is Yao et al., "PyHessian: Neural Networks Through the Lens of the Hessian", 2020, whose library BackdoorBench imports. The Hessian of an 86 M parameter network has 7e15 entries and is never formed. Everything is computed from Hessian-vector products by double backward, Pearlmutter's trick as PyHessian applies it.

$$
\begin{aligned}
Hv &= \frac{\partial}{\partial \theta} \left( \left( \frac{\partial L}{\partial \theta} \right)^{T} v \right)
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $L$ | the mean cross-entropy over the batch |
| $\theta$ | every parameter that requires a gradient, flattened, 85 831 723 entries on ViT-B/16 |
| $H$ | the Hessian $\partial^{2} L / \partial \theta^{2}$ |
| $v$ | a vector in parameter space |

The top eigenpairs come from power iteration with deflation, PyHessian's `eigenvalues()` line for line. For the $j$-th eigenpair a Gaussian start vector is orthogonalised against the $j - 1$ eigenvectors already found before every product, the Rayleigh quotient is the estimate, and the loop stops when 2 consecutive quotients agree to a relative tolerance of $10^{-3}$, with at most 1000 iterations as `visual_hessian.py` asks for.

$$
\begin{aligned}
v_{k} &\leftarrow \operatorname{orthnormal}(v_{k}, \{u_{1}, \dots, u_{j-1}\}) \\
w &= H v_{k} \\
\lambda &= w^{T} v_{k} \\
v_{k+1} &= \frac{w}{\|w\| + 10^{-6}} \\
\text{stop} &\iff \frac{|\lambda_{\text{prev}} - \lambda|}{|\lambda_{\text{prev}}| + 10^{-6}} < \text{tol}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $u_{i}$ | the $i$-th eigenvector already found, deflated against |
| $\lambda$ | the Rayleigh quotient, the eigenvalue estimate |
| $\text{tol}$ | the relative tolerance, $10^{-3}$ |

The density comes from stochastic Lanczos quadrature, PyHessian's `density()`: a Rademacher start vector, 100 Lanczos steps with full reorthogonalisation against every earlier Lanczos vector, and the eigen-decomposition of the tridiagonal matrix $T$, whose eigenvalues are the quadrature nodes and whose eigenvectors' squared first components are the weights.

$$
\begin{aligned}
w &= H v_{i} \\
\alpha_{i} &= w^{T} v_{i} \\
w &\leftarrow w - \alpha_{i} v_{i} - \beta_{i} v_{i-1} \\
\beta_{i+1} &= \|w\| \\
v_{i+1} &= \operatorname{orthnormal}(w, \{v_{0}, \dots, v_{i}\}) \\
T &= \operatorname{tridiag}(\beta, \alpha, \beta) = Q \Lambda Q^{T} \\
\text{nodes} &= \operatorname{diag}(\Lambda), \qquad \text{weights} = (Q_{0,:})^{2}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $v_{i}$ | the $i$-th Lanczos vector, unit length and orthogonal to every earlier one |
| $\alpha_{i}$ | the diagonal of $T$, the Rayleigh quotient at $v_{i}$ |
| $\beta_{i}$ | the off-diagonal of $T$, the residual norm |
| $Q$ | the eigenvectors of $T$ as columns |

The curve is `visual_hessian.py`'s `density_generate`: each node becomes a Gaussian bump of its weight on a grid of 10 000 points from the smallest node minus 0.01 to the largest plus 0.01, with variance $10^{-5}$ times the grid width when that width exceeds 1, and the curve is divided by its rectangle-rule integral `np.sum(density) * dx` so it integrates to 1. The figure draws it on a log y axis over the node range padded by 1 on each side, as upstream does.

### What the upstream script does

`visual_hessian.py` loads the attack result, takes the first batch of 128 images of the chosen split (`bd_train` in the demo, the poisoned training set, which at a 10% poison rate is mostly clean images), builds `pyhessian.hessian(model, criterion, data=(x, y))`, which runs 1 forward and 1 backward with `create_graph=True` and keeps that graph for every later product, calls `eigenvalues(top_n=2, maxIter=1000)` and `density()`, and plots the density with the largest eigenvalue in the title. It monkey-patches `torch.eig` onto `torch.linalg.eig` for torch newer than 1.8. The script runs upstream once `pyhessian` is installed.

### The ViT mapping and the deviations

The Hessian is taken over every parameter of the `Sequential(Resize, network)` model that requires a gradient, which is every parameter of ViT-B/16 or Swin-S since the Resize front-end has none. The products run in float32 with no autocast, because the second derivative of a bfloat16 forward is noise, and under the math scaled-dot-product kernel forced through `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`, because neither the fused CUDA kernels nor the CPU flash kernel implements a double backward (`derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented`). torchvision's ViT reaches that kernel through `nn.MultiheadAttention`, Swin's `shifted_window_attention` uses an explicit softmax and works either way.

The batch is 32 rows rather than upstream's 128, evaluated as 2 micro-batches of 16 through PyHessian's own multi-batch path (`dataloader_hv_product`, the row-weighted sum of per-chunk products divided by the row count). The retained double-backward graph of a 32-image batch of ViT-B/16 at 224 peaks at 16.3 GiB and a 64-image batch at 30.9 GiB, both over the login node's 15 GB budget, while a micro-batch of 16 peaks at about 9 GB. The loss is a mean over rows, so the micro-batch accumulation reproduces the 32-batch Hessian exactly, which `tests/test_analysis_curvature.py` checks to $10^{-10}$ on a net small enough to form the matrix. The price is that the forward and the first backward are recomputed for every product instead of reused from a kept graph, about 1.2 s per product against 0.55 s.

The Lanczos basis lives on the CPU. Full reorthogonalisation needs every earlier Lanczos vector, 100 of 86 M floats is 34 GB, and the GPU cannot hold it. Each new residual is moved to the CPU, orthogonalised there by the same sequential Gram-Schmidt PyHessian runs on the GPU, and moved back. The tridiagonal $T$ is diagonalised by `torch.linalg.eigh` in float64 rather than PyHessian's general `torch.linalg.eig` in float32, which on a symmetric tridiagonal matrix is the same decomposition with the real parts already taken. The vectors are held flat rather than as PyHessian's per-parameter lists, so a dot product is 1 call. PyHessian's 2 quirks in `eigenvalues()` are kept because the numbers upstream reports carry them: the eigenvalue recorded at the stop is the earlier of the 2 agreeing quotients, and the eigenvector is the normalised product of the final iteration. Its `+1e-6` in every normalisation is kept too, which limits the accuracy of the Lanczos nodes to about $10^{-3}$ of the spectral width, measured on the test net.

Both the top eigenvalues and the density are computed for every split the view names, and the figure draws 1 curve per split on shared axes with each split's top 2 eigenvalues in its legend entry, where upstream draws 1 split per figure with the largest eigenvalue in the title. The x axis is the union of the node ranges padded by 1. The sidecar records the top eigenvalues, the 100 nodes and weights and the 10 000-point curve per split.

## Loss landscape

The statistic is the loss on a 2-D slice through parameter space around the trained weights, along 2 random directions rescaled so that every unit of a direction has the norm of the matching unit of the weights. The source is Li et al., "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018, Section 4, Eq. 3, and the authors' `loss-landscape` repository, which BackdoorBench's script imports.

$$
\begin{aligned}
f(\alpha, \beta) &= L\left(\theta^{*} + \alpha \delta + \beta \eta\right) \\
\delta_{i,j} &\leftarrow \frac{\delta_{i,j}}{\|\delta_{i,j}\|} \, \|\theta_{i,j}\|
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $\theta^{*}$ | the trained parameters |
| $\delta, \eta$ | the 2 random Gaussian directions, 1 tensor per parameter |
| $\alpha, \beta$ | the grid coordinates, $[-1, 1]$ |
| $\delta_{i,j}$ | the $j$-th filter of the $i$-th layer of $\delta$ |
| $\theta_{i,j}$ | the matching filter of $\theta^{*}$ |
| $L$ | the mean loss over the evaluation set |

The repository's `normalize_direction` with `norm='filter'` iterates a parameter tensor along its first axis and rescales each slice by `w.norm() / (d.norm() + 1e-10)`. Its `normalize_directions_for_weights` with `ignore='biasbn'` sets the direction of every parameter of rank 1 or 0 to 0. The cosine between the 2 directions is `projection.cal_angle`, the dot product of the flattened directions over their norms, which the script prints. At every grid point `evaluation.eval_loss` sets the weights, runs the loader under `no_grad` and returns the row-weighted mean loss and the accuracy.

### What the upstream script does

`visual_landscape.py` asserts that a clone of `github.com/tomgoldstein/loss-landscape` sits at `./visualization/loss-landscape`, which BackdoorBench does not vendor, imports `mpi4pytorch` and `h5py` from it, and distributes the grid over MPI ranks, `mpirun -n 8` in the demo. It reads the poisoned training set (`bd_train`) or the clean one, draws the 2 directions into an HDF5 file, evaluates a 51 by 51 grid over $[-1, 1]$ (the demo's `--x=-1:1:51 --y=-1:1:51`, the argument default is 30 by 30) across the full set, and plots the 3-D surface with `plot_surface` under `coolwarm`, leaving the contour of `plot_2D.plot_2d_contour` commented out. Before any of that it reads `args.dir_gen` to decide whether to use the top 2 Hessian eigenvectors as directions, and `get_args()` in `visual_utils.py` defines no such argument, so the script raises `AttributeError` at that line on every invocation. The script does not run upstream as shipped, on 2 counts.

### The ViT mapping and the deviations

The paper's unit is a convolution filter. On a transformer the unit is a row of a weight matrix, every `Linear` and the fused qkv `in_proj_weight` included, because the repository's loop over the first axis of a rank-2 tensor is a loop over its rows and over the rank-4 `conv_proj` kernel is a loop over filters. The rank-3 class token and position embedding have a first axis of length 1, so the rule rescales each as 1 unit to its own norm. Every bias and every LayerNorm affine parameter has rank 1 and takes a direction of 0. `tests/test_analysis_landscape.py` pins the invariant on the synthetic ViT, which carries every rank from 1 to 4: every row of a direction has the norm of the corresponding parameter row and every rank-1 parameter's direction is 0.

The 2 directions are drawn on the CPU in the parameters' dtype from seeds `--seed` and `--seed + 1` through Lightning's `seed_everything`, so the same seed gives the same slice on any machine. They are cached under `results/<folder>/visual/landscape_directions.pt` (688 MB for ViT-B/16, `results/**` is gitignored) so the clean surface, the triggered surface and any later run of the same checkpoint share their coordinates. The cosine is recorded in the sidecar.

The grid is 21 by 21 over $[-1, 1]$ and the set is the first 512 rows of each split the view names, rather than the 51 by 51 grid over the full training set. Every grid point is a full pass, so the cost is 441 passes over 512 images per split, about 2 minutes per split under bfloat16 autocast on the A100, where upstream's grid over the full set would take 4 to 6 GPU hours on ViT-B/16 and is not run. `--samples` caps the row count below 512 for a smoke run. The rows are moved to the GPU once as tensors so the 441 passes rebuild no loader. The grid is serial, without the MPI reduction.

The weights are set from a stored copy of $\theta^{*}$ before every point and copied back from it after every point, in a `try/finally`, so the model leaves the surface bit for bit as it entered, an exception midway included, which the tests check by comparing every state-dict tensor with `torch.equal`. Upstream's `crunch` never restores the weights after the last point. Both the loss and the accuracy surfaces are recorded as upstream's `surf_file` records `train_loss` and `train_acc`, with the accuracy as a fraction in $[0, 1]$ rather than upstream's percentage. The forward passes honour `--no-bfloat16` as every other tool does, so the default surface is evaluated under bfloat16 autocast, and the surfaces with their alpha and beta grids are in the sidecar.

The figure draws 1 row per split, the 3-D surface under `coolwarm` on the left and the contour on the right, with the repository's contour levels 0.1 to 10 in steps of 0.5 where at least 2 of them fall inside the surface and 10 evenly spread levels otherwise, since a contour with fewer than 2 levels draws nothing.

## Feature visualisation

The statistic is the image that maximises 1 unit of the network, found by gradient ascent from noise. The source is Olah, Mordvintsev and Schubert, "Feature Visualization", Distill 2017, in the form OmniXAI's `FeatureVisualizer` runs it, whose Fourier parameterisation and colour decorrelation are lucid's as ported to torch by lucent.

$$
\begin{aligned}
x_{b} &= \operatorname{range}\left( \sigma\left( C \, \operatorname{irfft2}\left( S_{b} \odot s \right) / 4 \right) \right) \\
s &= \frac{1}{\max\left(f, \, 1 / \max(h, w)\right)^{d}} \\
\ell_{b} &= - a_{k_{b}}\left(T(x_{b})\right) + w_{1} \operatorname{mean}\left|T(x_{b})\right| + w_{tv} \operatorname{TV}\left(T(x_{b})\right) \\
\operatorname{TV}(x) &= \frac{1}{c h w} \left( \sum_{i,j} (x_{i+1,j} - x_{i,j})^{2} + \sum_{i,j} (x_{i,j+1} - x_{i,j})^{2} \right)
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $S_{b}$ | the half spectrum of image $b$, real and imaginary parts, the free variable, initialised at $\mathcal{N}(0, 0.01)$ |
| $f$ | the radial frequency of each spectrum entry |
| $s$ | the $1/f$ scaling, with $d = 1$ |
| $C$ | lucid's colour correlation matrix, the normalised square root of the ImageNet colour covariance |
| $\sigma$ | the sigmoid |
| $\operatorname{range}$ | OmniXAI's rescale of each image's own minimum and maximum onto $[0.05, 0.95]$ |
| $T$ | the random transform of the step |
| $a_{k}$ | the activation of unit $k$ |
| $w_{1}, w_{tv}$ | the L1 and total-variation weights, 0.15 and 0.25 |
| $c, h, w$ | the image's channels, height and width |

Adam at learning rate 0.05 runs 300 steps on the spectra of every image at once, 1 forward and 1 backward per step for the whole batch. OmniXAI's `_regularize` gives the L1 term as the mean absolute pixel value per image and the total variation as the sum of squared neighbour differences over $c h w$, and its `optimize` differentiates the per-image losses through `torch.unbind`, which is the gradient of their sum since each image's loss depends on its own spectrum alone.

### What the upstream script does

`visual_fv.py` resolves `--target_layer_name` to a module, `layer4.1.conv2` in the demo, builds `FeatureVisualizer` with the objective `{"type": "channel", "index": list(range(target_layer.out_channels))}`, passes BackdoorBench's train transform (`Resize`, `RandomCrop` with padding 4, a horizontal flip on CIFAR-10 only, `ToTensor`, `Normalize`) as `transformers`, and calls `explain(num_iterations=300, regularizers=[("l1", 0.15), ("l2", 0), ("tv", 0.25)], use_fft=True)`. It then draws the images 16 per row, titled `Kernel i`. 2 things stop it from running. `target_layer.out_channels` exists on a convolution and on nothing in a transformer, so the objective cannot be built for a ViT. And OmniXAI's optimiser calls `transformers.transform(images)` on the object it was handed, which is a torchvision `Compose` with no `transform` method (`hasattr(Compose([...]), "transform")` is `False` on torchvision 0.21), so the call raises `AttributeError` on any model, and the `ToTensor` inside that `Compose` would refuse a batched tensor after that. The script does not run upstream as shipped, on 2 counts.

### The ViT mapping and the deviations

Upstream's unit is a convolution channel, its objective the mean of that channel's spatial map. A transformer block emits `(batch, tokens, dim)`, so the unit here is 1 residual dimension of a block's output, read at the class token on ViT, the token the head classifies from, and as the mean over tokens on Swin, which has no class token. `--layer` picks the block, the last one by default, matching the demo's choice of the last convolution. Upstream synthesises 1 image for every channel of the layer. A ViT-B/16 block has 768 dimensions, so the tool restricts itself to the 16 with the largest trigger-activated change, `analysis.direction.trigger_activated_change` pooled the same way over the first 4 paired batches of the clean and backdoor loaders, since the dimensions the trigger moves most are the ones whose maximising input might show the trigger. The ranking is a paired statistic, so `--view` has no effect on this tool.

The image is synthesised at the dataset's native resolution, 32 by 32 on GTSRB, where the triggers live, and the model's own Resize front-end takes it to 224 as it does every input. The transform is what upstream's train transform would have done had it run, a translation by up to 4 pixels through zero padding and a random crop, applied as 1 shift per batch, plus lucent's random scale by a factor from 0.9 to 1.1, whose output size the Resize front-end absorbs. The dataset's normalisation is applied after the transform and before the model, the place the train transform put it, and the regularisers read the transformed image in $[0, 1]$ before that normalisation. The `l2` regulariser has weight 0 upstream and is not implemented. The model's parameters are frozen through `defences.inference.frozen_parameters` for the duration, so the backward stops at the activations, and the block is read through `analysis.features.captured_layers`. The final image is taken without any transform. The figure lays the 16 images out as a 4 by 4 grid with nearest-neighbour interpolation so each pixel of the 32 by 32 canvas stays a visible square, and titles each by its dimension index and its TAC. The sidecar holds the dimensions, their TAC, the unit activation at the first and last step and the images as 8-bit integers.

## Cost on the login node

Measured on the Supek login node's A100-PCIE-40GB with the GPU otherwise idle, on `vit_gtsrb_badnet_a2o_0_05` and `vit_gtsrb_benign` with `--samples 512`, the `paired` view and the default batch size of 64. The Hessian run is the only one in float32.

COST_TABLE_PLACEHOLDER

The Hessian-vector product costs about 1.2 s for 32 rows as 2 micro-batches of 16, and the CPU-side reorthogonalisation of the Lanczos basis about 2 minutes over the 100 steps at 64 threads. A loss-surface point costs a pass over 512 images at batch 64 under bfloat16 autocast. The feature-visualisation step costs 1 forward and 1 backward over 16 images.

RESULTS_PLACEHOLDER
