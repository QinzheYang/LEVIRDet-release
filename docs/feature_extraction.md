# Extract features using LEVIRDetNet

This API loads the backbone learned by the **30-class LEVIRDetNet** checkpoint
`epoch_117.pth`. It is intended for extracting image descriptors, inspecting
spatial features, and using them in downstream analysis. It does not run the
object detector, neck, detection head, or GSD branch.

Use the existing project Docker or Linux Conda environment and run from the
repository root. No additional packages or pretrained weights are downloaded.
Keep `levir_features.py` beside `configs/` and `mmdet/`; the helper automatically
locates `configs/levirdetnet/levirdetnet-30class.py`. Relative image/checkpoint
paths are resolved from your working directory. Use trusted local checkpoints.

## Minimal example

```python
from levir_features import load_model, load_image, vis

model = load_model("epoch_117.pth")
img = load_image("path/to/your/image.jpg")
feature = model(img)
vis(feature, "work_dirs/feature_pca.png", size=(1024, 1024))
```

All preprocessing and checkpoint loading live in `levir_features.py`.
CUDA is selected when available; CPU works but is slower. To choose explicitly:

```python
model = load_model("epoch_117.pth", device="cuda:0")
# Or: model = load_model("epoch_117.pth", device="cpu", image_size=512)
```

For a ready-to-run example that also saves the feature tensor:

```bash
python demo/feature_demo.py path/to/your/image.jpg --weights epoch_117.pth --out-dir work_dirs/features
```

This creates `feature_pca.png` and `features.pt`. Load the latter as a CPU tensor
with `torch.load("work_dirs/features/features.pt", map_location="cpu", weights_only=True)`.
In Docker, save under the writable `work_dirs` mount so results persist on the
host. The same Python code runs inside Docker and in an independently prepared
Conda environment.

## Which features are returned?

The released configuration uses **DINOv3 ViT-S+/16**, even though some historical
configuration filenames contain `vitb`. Its transformer has 384 channels and
16-pixel patches. The trained backbone also includes a spatial adapter (STA).

| Call | Output for a 1024 × 1024 image | Meaning |
| --- | --- | --- |
| `model(img)` | `[1, 384, 64, 64]` | Final normalized DINO patch-token grid, without CLS/storage tokens or STA fusion. |
| `model(img, output="cls")` | `[1, 384]` | Final normalized global CLS token. |
| `model(img, output="pyramid")` | Tuple: `[1,256,128,128]`, `[1,256,64,64]`, `[1,256,32,32]` | Complete trained `DINOv3STAs` backbone outputs, including STA fusion and projections, at strides 8/16/32. |

Example for the middle pyramid level:

```python
pyramid = model(img, output="pyramid")
vis(pyramid[1], "work_dirs/pyramid_level2.png", size=(1024, 1024))
```

Outputs remain on the model's device and require no gradients. Use `.cpu()`
before saving if you want portable tensors. This convenience wrapper is frozen
and inference-only; use the training configuration for fine-tuning.

The backbone weights are read strictly from the checkpoint's `state_dict`, with
all 254 backbone tensors checked for matching keys and shapes. The helper does
not substitute `ema_state_dict`: this follows the deployed state selection used
by the project's image demo. In this checkpoint's EMA save convention, the two
fields are not interchangeable. The checkpoint must match the 30-class detector;
a 159-class checkpoint or raw DINO pretraining checkpoint is rejected.

## Inputs and preprocessing

- `load_image(path)` returns an RGB PIL image and applies its EXIF orientation.
- `model` accepts that PIL image, a local path, a `uint8` RGB numpy array of shape
  `[H,W,3]`, or a RGB tensor of shape `[3,H,W]` / `[B,3,H,W]`.
- Tensor pixels must be `uint8` in 0–255 or floating point in 0–1. **Do not
  normalize inputs yourself.** Convert OpenCV BGR arrays to RGB first.
- Default preprocessing resizes to **1024 × 1024 without keeping aspect ratio**,
  as in the release test configuration. It normalizes RGB pixels in 0–255 using
  mean `[123.675,116.28,103.53]` and standard deviation `[58.395,57.12,57.375]`.
- PIL/array inputs use OpenCV bilinear resizing. Tensor inputs use PyTorch
  bilinear resizing; rounding can produce small differences between input types.
- `image_size` can be an integer or `(height,width)`; dimensions must be positive
  multiples of 32 so every pyramid level remains aligned. Larger inputs use more
  time and memory. The entire image is resized; this helper does not tile images.

## PCA visualization

`vis` projects the dense feature channels onto their first three principal
components, scales each component using its 2nd/98th percentiles, and saves an
RGB image. This provides a qualitative view of feature structure, inspired by
the [DINOv3 feature visualization workflow](https://github.com/facebookresearch/dinov3/tree/main/notebooks).
It is a simple whole-image PCA; it does not perform foreground masking.

PCA colors **are not semantic segmentation or detection labels**. Each image
gets its own PCA basis, so independently generated colors should not be compared
as if they encoded the same classes. Changing the image or its crop can change
the colors. Constant features produce a black visualization.

`vis` accepts one dense map, not a CLS vector or an entire batch. For batch output,
use `vis(feature[i])`; for a pyramid tuple, select a level first. It returns a PIL
image, which can be displayed directly in a notebook:

```python
preview = vis(feature, save_path=None, size=(1024, 1024))
display(preview)  # Jupyter
```

Feature extraction and visualization do not need a graphical desktop and work
on headless servers. `vis` writes an image; it does not automatically open an
external image viewer.
