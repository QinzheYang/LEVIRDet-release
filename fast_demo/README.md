# LEVIRDetNet Fast Demo

A standalone desktop application for local GPU inference. It opens its own
window with native file and folder pickers; no browser window, command entry,
Conda environment, Python installation, or Node.js installation is needed for
normal use. The 30-class model is selected by default; the 159-class model is
available in the same window.

## First-time preparation

Use a Windows 10/11 x86-64 PC or a Linux x86-64 desktop with an NVIDIA GPU.
Install Docker and the NVIDIA driver once:

- Windows: install Docker Desktop, finish its setup, and enable its WSL 2
  backend. Follow [Docker's GPU setup guide](https://docs.docker.com/desktop/features/gpu/).
- Linux: install Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Your ordinary desktop user needs Docker access. A graphical desktop session,
  Bash 4+, curl, unzip, and standard system tools are required. An SSH-only
  server cannot display this application. Electron also needs the system's
  desktop libraries and a working Chromium sandbox.

The launcher checks Docker and GPU access. It can start an already installed
Docker Desktop on Windows, but does not install drivers, change system
permissions, or accept Docker's terms for you.

The three main downloads total about **13.13 GiB**, plus the project source and
a portable desktop runtime (about 151 MiB on Windows / 117 MiB on Linux).
Docker needs additional space to import the image. Choose a writable location
with ample free disk space, and keep the launcher window open during setup.

## Start without typing commands

1. Download the [Windows launcher](https://github.com/QinzheYang/LEVIRDet-release/raw/refs/heads/main/launch_fast_demo.cmd)
   or [Linux launcher](https://github.com/QinzheYang/LEVIRDet-release/raw/refs/heads/main/launch_fast_demo.sh).
   Save the actual file, not an HTML copy of a GitHub page. If the browser shows
   its contents, use **Save link as** on the download link.
2. Put the launcher in a writable folder. If you already have the complete
   release project, put it in the project root to reuse your files.
3. Windows: double-click `launch_fast_demo.cmd`. Linux: allow the `.sh` file to
   execute in its file properties and choose **Run as a program** in your file
   manager. Terminal users can instead run `bash launch_fast_demo.sh`.
4. The launcher finds or downloads the code, Docker image, both checkpoints,
   and desktop runtime. It verifies file sizes and SHA-256 checksums, imports
   the image if needed, and opens the **LEVIRDetNet** desktop window.
5. Select **Choose images** or **Choose folder**, choose **30 classes** or
   **159 classes**, then click **Confirm & run**. The left panel shows the
   original and the right panel shows detections. Use the arrows to page
   through both views together.

Folder selection includes supported images in subfolders. The display
threshold affects drawn boxes; prediction JSON files keep all saved detection
scores. Images are resized to 1024 × 1024 for inference and detections are
saved in original-image coordinates. Inference uses one image per batch,
matching the standard image demo.

Close the application after the run to stop its own background container.
Closing it during inference interrupts that run; files already written remain
available. The launcher does not stop unrelated Docker containers.

## Where files are stored

When only a launcher was downloaded, the project is created beside it as
`levirdetnet-release`. When launched inside an existing project, that project
is reused. The launcher does not overwrite an incomplete project automatically.

```text
levirdetnet-release/
├── launch_fast_demo.cmd
├── launch_fast_demo.sh
├── fast_demo/
├── docker/levir-train-cuda121-torch231.tar
├── epoch_117.pth
├── best_coco_weighted_bbox_mAP_epoch_110_159class.pth
├── .fast_demo_cache/              # Downloads and verification cache
├── .fast_demo/                    # Portable desktop runtime
└── fast_demo_outputs/
    └── <timestamp>_<model>class_<id>/
        ├── input/                # Copies of selected inputs
        ├── predictions/
        │   ├── vis/              # Visualization images
        │   ├── preds/            # Per-image prediction JSON
        │   ├── run_metadata.json # Class names and inference checks
        │   └── effective_config.py
        ├── manifest.json         # Original names → numbered input/output files
        └── inference.log
```

Use **Open folder** to see the saved run. Inputs are copied and numbered so
different folders can contain identical filenames without overwriting results.
`manifest.json` retains each original name and relative folder. Original images
are never changed. Browser-incompatible image formats receive a JPEG preview
inside the desktop window; original inputs and detector outputs stay on disk.

Images and results stay on your computer. The desktop window talks only to
the local inference service; it does not upload images to Google Drive or a
remote server. The service is published only on `127.0.0.1`, with a new session
key each time it starts. On Linux, the inference container runs with your user
and group IDs so that generated files remain writable by you. Its private
runtime home is stored under `fast_demo_outputs/.runtime-home`.

## If an automatic download fails

The launcher shows the reason and creates `LEVIRDetNet-setup-help.txt` beside
it, containing the download links, expected filenames, checksums, and exact
destination paths. Download the original files manually and run the same
launcher again:

| File | Manual download | Place inside the project |
| --- | --- | --- |
| Docker image | [Google Drive](https://drive.google.com/file/d/1h76qWD5WfomQUXoQWGxFZs6R_QCl64ei/view) | `docker/levir-train-cuda121-torch231.tar` |
| 30-class checkpoint | [Google Drive](https://drive.google.com/file/d/1G-WIA44hUVPjEjEPOHcm3OP2qh7jUIde/view) | `epoch_117.pth` |
| 159-class checkpoint | [Google Drive](https://drive.google.com/file/d/1lLIrXkpyDi0gJHCvjHKffYupr2Eq8UGG/view) | `best_coco_weighted_bbox_mAP_epoch_110_159class.pth` |

Google Drive may limit downloads or show a permission page. An HTML page is
never accepted as a checkpoint. If a file has the wrong size or checksum,
keep it elsewhere and put the correct file at the requested location. Download
the portable desktop runtime from the official Electron link printed in the
setup-help file if its automatic download fails.

All expected sizes, hashes, model/config associations, and runtime URLs are
stored in [assets.json](assets.json). The standalone inference checkpoints
already include DINO and GSD parameters; their separate initialization weights
and dataset annotation files are not required for this demo.

## For release maintainers

Publish both root launchers and the complete `fast_demo/` directory together
with the existing `demo/`, `mmdet/`, `configs/`, and other project source on the
repository's `main` branch. A standalone downloaded launcher obtains those
files from that branch. Do not commit runtime caches, downloaded checkpoints,
Docker archives, or users' inference outputs.

When replacing a checkpoint or runtime, update its size and SHA-256 in
`assets.json`; do not reuse an old checksum with new file contents. The two
Google Drive weight links are **159 classes first, 30 classes second** in the
order originally supplied; the manifest binds them to the verified filenames.

The desktop shell is Electron with context isolation and sandboxing enabled.
Inference uses the existing `demo/image_demo.py` without changing the model or
prediction code. Local protocol tests can be run in a Python environment with
Pillow: `python fast_demo/test_server.py`.
