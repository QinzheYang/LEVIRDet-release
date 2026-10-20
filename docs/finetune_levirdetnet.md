# Fine-tune LEVIRDetNet on your own classes

Start from the released **30-class `epoch_117.pth`** in the existing LEVIRDetNet
Docker or Linux Conda environment. Run the commands from the project root.
The transfer script prepares a new task; normal MMDetection training commands
remain unchanged.

## Quick start

Prepare your own bounding-box dataset in COCO format:

```text
data/my_dataset/
├── train/images/...
├── val/images/...
└── annotations/
    ├── train.json
    └── val.json
```

For a two-class task, both JSON files must define exactly `plane` and `ship` in
their `categories`. The names are case-sensitive; category IDs need not start at
zero or be consecutive. Each image's `file_name` is relative to its split's
image directory. Validation classes with no annotated objects must still appear
in `categories`.

```bash
python tools/transfer_ckpt.py --new-class plane,ship --data-root data/my_dataset
python tools/train.py work_dirs/finetune/finetune.py
```

Replace `plane,ship` with your own class names. Both `--new-class plane ship`
and `--new-class plane,ship` work. Quote names containing spaces.

The first command creates:

```text
work_dirs/finetune/
├── init.pth                 # New-task initialization, not a resumed training run
├── finetune.py              # Train with this generated config
├── transfer_report.json     # Class mapping and validation summary
└── annotations/
    ├── train.json           # Copies with categories ordered to match the head
    └── val.json
```

The original checkpoint, images, and annotations are not modified. Images are
not copied. All generated files and subsequent logs/checkpoints stay under the
new output directory. An existing output directory is never overwritten; choose
a different `--out-dir` when preparing another task.

For multiple GPUs, use the same generated configuration:

```bash
bash tools/dist_train.sh work_dirs/finetune/finetune.py 8
```

Replace `8` with your GPU count. Do not pass `--resume` for the first training
run: `load_from` loads the transferred model parameters and starts a fresh
optimizer and schedule. To continue this new run after interruption, use the
normal `--resume` option with the same config and work directory.

## Paths and common options

The default source checkpoint is `epoch_117.pth` in the working directory. Use
`--weights /path/to/epoch_117.pth` for another location. It must be the released
30-class complete detector, not the 159-class or DINO-only weights. Only load
checkpoints from a trusted source.

The defaults are a **48-epoch starting schedule**, batch size **2 per GPU**, and
AdamW learning rate **0.0001**. These are a starting point for a new dataset,
not a claim that they are optimal for every task. To set them when generating:

```bash
python tools/transfer_ckpt.py --new-class plane,ship --data-root /data/my_dataset \
  --epochs 24 --batch-size 2 --lr 0.0001 --out-dir work_dirs/plane_ship
python tools/train.py work_dirs/plane_ship/finetune.py
```

If your validation split is called `test`, use:

```bash
python tools/transfer_ckpt.py --new-class plane,ship --data-root /data/my_dataset \
  --val-ann annotations/test.json --val-images test/images
```

`--train-ann` and `--train-images` likewise change the training layout. Paths
are relative to `--data-root`, or may be absolute. The converter checks both
annotation files and their referenced image paths before creating outputs.

In Docker, run the converter **inside the same container environment used for
training**, using container paths such as `/data/my_dataset`. The project's
existing `work_dirs` bind mount must be writable. The generated config inherits
the project code at a relative location; rerun conversion if moving to a
different project/data layout. A different host path is not automatically a
valid path inside a container. No extra environment activation or separate
DINO/GSD initializer is needed in the prepared image.

You can adjust the generated config's top-level `max_epochs`, `batch_size`,
`base_lr`, and `num_workers`. Changing only a top-level data path does not
regenerate annotation copies; run the converter again for a different dataset.
For **different classes or a different class order**, always rerun conversion
into a new directory instead of editing only `num_classes`.

## What is transferred?

The converter reads `checkpoint['state_dict']`, following the released image
demo's deployed-weight convention. It does not substitute `ema_state_dict`.

- All class-independent parameters are retained: DINO/STA backbone, encoder,
  decoder, box regression, and the complete GSD branch.
- All seven classification branches (six decoder layers and the encoder head)
  and the denoising label embedding are adapted to the requested class order.
- **Exact-name matches** reuse their original rows; for example, `plane` uses
  source row 0 and `ship` uses source row 9.
- A genuinely new class receives new classifier weights, a low-probability
  classifier bias, and a new denoising embedding. It is not initialized from an
  unrelated source class. `--seed` makes this initialization repeatable.
- Old optimizer/scheduler state, epoch/iteration counters, EMA state and palette
  are discarded. The new checkpoint records the new classes and mapping.

For the released checkpoint, this adapts **15 tensors** and preserves the other
**1,369 entries**. The new detector has the same architecture apart from its
class count. Both the DINO backbone and GSD branch remain trainable, with smaller
learning rates (0.1× and 0.01× the base rate respectively).

The generated configuration uses fixed 1024×1024 resize, random horizontal
flipping, one warmup epoch, and cosine decay thereafter. It removes the original
192-epoch augmentation stages, assigner switching and EMA restart schedule. It
evaluates the new COCO classes each epoch and saves the best `coco/bbox_mAP`.

## Annotation semantics

The source JSON files must contain only the intended task's categories. The
script does not silently rename, merge or remove unrelated categories. Prepare
your task annotations first if starting from a larger dataset.

The two generated JSON copies reorder `categories` to exactly match the new
classifier rows. This matters because the project's COCO loader and evaluator
use JSON category order, not just `metainfo.classes`. Category IDs, image IDs,
annotation IDs, boxes and additional fields are preserved; the two splits can
use different category IDs.

By default, standard COCO `iscrowd=1` remains a crowd/ignored annotation. If those
flags were accidental and should represent ordinary objects, add
`--crowd-as-normal`. This sets `iscrowd=0` **only in the generated copies**.
Other flags such as `ignore` are preserved. The original JSON files stay intact.

## After training

Use the same generated config for testing and prediction:

```bash
python tools/test.py work_dirs/finetune/finetune.py work_dirs/finetune/epoch_48.pth
python demo/image_demo.py your_image.jpg work_dirs/finetune/finetune.py --weights work_dirs/finetune/epoch_48.pth --out-dir work_dirs/finetune/predictions
```

Replace `epoch_48.pth` with a checkpoint actually saved by your run, such as its
best-mAP checkpoint. A class created from random rows needs fine-tuning before
its predictions are meaningful.
