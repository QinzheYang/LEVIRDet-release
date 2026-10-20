# LEVIRDetNet Fast Demo

A standalone graphical launcher and desktop application for local GPU
inference. Both open their own windows; normal use needs no browser, command
entry, Conda environment, Python installation, or Node.js installation. The
launcher prepares the resources, then opens the existing side-by-side
inference interface. The 30-class model remains the default; the 159-class
model is available in the same inference window.

## First-time preparation

Use a Windows 10/11 x86-64 PC or a Linux x86-64 desktop with an NVIDIA GPU.
Install the NVIDIA driver for your computer first. Opening a downloaded
launcher prepares only its interface files and portable Electron runtime.
Windows shows a small preparation window if those files are missing. It
reuses an existing complete launcher/runtime when available. The main
launcher then opens before any Docker installer, image, checkpoint, or full
project-code download starts.

After you click **Download**, it checks these container prerequisites:

- **Windows:** missing Docker Desktop or missing/outdated WSL 2 triggers an
  installation question. Approval downloads the official installer or runs
  Microsoft's WSL setup command. The Docker installer is checked for a valid
  Docker publisher signature before it runs. GPU support comes from Docker
  Desktop's WSL 2 integration; the Linux NVIDIA Container Toolkit package is
  not installed on Windows. See [Docker's GPU setup guide](https://docs.docker.com/desktop/features/gpu/).
- **Linux:** the launcher uses native Docker Engine and
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  With your approval, it can install missing components from official signed
  package repositories on Ubuntu 22.04 / 24.04 / 26.04 and Debian 12 / 13.
  Other distributions receive manual setup instructions. A graphical desktop
  and an ordinary user account are required. Bash 4+, curl, unzip, SHA-256
  utilities, and standard system tools are needed to prepare the desktop
  launcher. Electron also needs the system's desktop libraries and a
  working Chromium sandbox. An SSH-only server cannot display the demo.

All installation questions default to **No**. Declining or closing a question
stops that setup step and displays manual instructions. The launcher checks
again after installation and continues when the environment is ready.

You may still need to complete administrator authentication, accept Docker's
terms in Docker Desktop, restart Windows, or sign out and back in on Linux.
After a restart or new login, open the same launcher again. The launcher does
not install GPU drivers, automatically accept Docker's terms, or reboot your
computer.

Windows prerequisite installer downloads are cached in `.fast_demo_setup`
inside the selected download folder. Linux uses its system package manager.
Linux installation questions require a supported desktop dialog tool
(`zenity`, `kdialog`, or `xmessage`). If none is available, install one through
your distribution or follow the manual prerequisite instructions. Lack of a
response never counts as approval.

On Linux, changes to Docker's GPU configuration and service restarts require
approval. The existing daemon configuration is backed up before modification.
A restart can interrupt running containers. Adding your account to the
`docker` group is a separate question because that access grants root-level
control through Docker; it requires a new login to take effect. Remote,
rootless, and Docker Desktop contexts receive manual guidance instead of
automatic changes to the wrong daemon.

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
4. In the launcher, open **Settings** at the upper right to inspect the
   destination, available disk space, required space, and planned files.
   Click the displayed path to expand the resource and disk-space details.
   **Change path** opens a native folder picker. The selected folder becomes
   the project/resource root. Its setting is remembered for the next launch.
5. Click **Download** at the lower right. Review prerequisite installation
   questions and complete approved administrator or Docker license prompts.
   If a restart or new login is required, reopen the launcher afterward.
   Setup proceeds in this order: **prerequisites → Docker image download and
   import → 30-class checkpoint → 159-class checkpoint → full project code**.
   Existing valid resources are reused.
6. Watch the current item, downloaded/total bytes, speed, and estimated time
   remaining. Speed and ETA use actual transfer measurements; operations
   without a known size, such as Docker import or package-manager setup,
   show their stage rather than a made-up percentage. Use **Pause** and
   **Resume** during supported transfers; partial files are retained.
7. When resources are ready, click **Launch**. In the inference window,
   select **Choose images** or **Choose folder**, choose **30 classes** or
   **159 classes**, then click **Confirm & run**. The left panel shows the
   original and the right panel shows detections. Use the arrows to page
   through both views together.

The gear beside **Download / Launch** opens shortcuts for adding a desktop
shortcut, checking resource integrity, changing the download path, and
switching between English and Chinese. Integrity checking reads the Docker
archive and both checkpoints and compares their sizes and SHA-256 hashes;
for project code, it checks that the required entrypoint files exist. It does
not claim a cryptographic integrity check of every source file.

Path changes do not move, delete, or overwrite resources in the old location.
Choose an existing release folder to reuse its files, or a new folder for a
separate setup. Pause the current transfer before changing its destination.
Some startups show a small greeting with local time, a wish, and an optional
link to the project's GitHub homepage; dismissing it does not affect setup.

Folder selection includes supported images in subfolders. The display
threshold affects drawn boxes; prediction JSON files keep all saved detection
scores. Images are resized to 1024 × 1024 for inference and detections are
saved in original-image coordinates. Inference uses one image per batch,
matching the standard image demo.

Closing the inference window stops its own background container. Closing the
launcher while it owns an active inference session asks for confirmation and
then stops that session. Interrupting inference leaves files already written
available. Unrelated Docker containers are not stopped. Installation or import
steps must finish before the launcher can close.

The launcher keeps the reference artwork's 1639:907 proportions and fits
inside the display's work area, including Windows display scaling. It reports
that inference is open only after the native window confirms it is visible
and its interface has loaded. Startup failures show an error and record the
desktop log under `fast_demo_outputs/.runtime-home/`.

## Where files are stored

When only a launcher was downloaded, the default project/resource folder is
`levirdetnet-release` beside it. A launcher opened inside an existing release
defaults to that project. Change this location before downloading through
**Settings → Change path**. Existing files are preserved; conflicts while
completing project code require manual review.

The initial desktop bootstrap is stored in `.levir-launcher-v3` beside a
standalone launcher. The original launcher folder also holds
`.fast_demo_setup/launcher-settings.json` and its desktop profile. An existing
project can reuse its `.fast_demo` runtime. The selected resource folder has
the following layout:

```text
levirdetnet-release/
├── launch_fast_demo.cmd
├── launch_fast_demo.sh
├── fast_demo/
├── docker/levir-train-cuda121-torch231.tar
├── epoch_117.pth
├── best_coco_weighted_bbox_mAP_epoch_110_159class.pth
├── .fast_demo_cache/              # Downloads and verification cache
├── .fast_demo/                    # Portable runtime, if prepared here
├── .fast_demo_setup/              # Approved Windows prerequisite installer cache
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

## Download sources and automatic fallback

Docker and both checkpoints try Google Drive first. After a 10-second
connection grace period, a 30-second average speed below **1 MiB/s** switches
the transfer to the matching [Hugging Face mirror](https://huggingface.co/yangqinzhe/LEVIRDetNet/tree/main).
Quota pages, HTTP errors, and connection failures also trigger fallback.
The progress panel displays the current source. Switching sources keeps the
partial download and attempts HTTP Range resume; a server that does not
support Range causes a clean restart. Both sources use the same required
file size and SHA-256, and verification must pass before the file is used.
Pressing **Pause** pauses the whole transfer instead of switching sources.
Invalid partial files are retained with an `.invalid` suffix for inspection.
The desktop launcher uses Electron's native networking so it follows the
system network/proxy configuration. HTTPS certificate validation stays enabled.

The new CMD and shell bootstrap use `.levir-launcher-v3` when only a launcher
was downloaded, so an older cached downloader cannot silently skip fallback.
Existing project resources and the older Electron runtime remain reusable.
Project code and the portable Electron runtime still use their GitHub URLs.

## If an automatic download fails

The launcher's error panel shows the reason, a manual download link, and the
exact destination. It also writes `LEVIRDetNet-setup-help.txt` in the selected
download folder with links, filenames, sizes, checksums, and destinations.
If preparing the launcher itself fails before the main window opens, see
`LEVIRDetNet-launcher-help.txt` beside the original CMD or shell file instead.
Download the original files manually and click **Download** again:

| File | Manual download | Place inside the project |
| --- | --- | --- |
| Docker image | [Google Drive](https://drive.google.com/file/d/1h76qWD5WfomQUXoQWGxFZs6R_QCl64ei/view) · [Hugging Face](https://huggingface.co/yangqinzhe/LEVIRDetNet/resolve/main/docker/levir-train-cuda121-torch231.tar?download=true) | `docker/levir-train-cuda121-torch231.tar` |
| 30-class checkpoint | [Google Drive](https://drive.google.com/file/d/1G-WIA44hUVPjEjEPOHcm3OP2qh7jUIde/view) · [Hugging Face](https://huggingface.co/yangqinzhe/LEVIRDetNet/resolve/main/epoch_117.pth?download=true) | `epoch_117.pth` |
| 159-class checkpoint | [Google Drive](https://drive.google.com/file/d/1lLIrXkpyDi0gJHCvjHKffYupr2Eq8UGG/view) · [Hugging Face](https://huggingface.co/yangqinzhe/LEVIRDetNet/resolve/main/best_coco_weighted_bbox_mAP_epoch_110_159class.pth?download=true) | `best_coco_weighted_bbox_mAP_epoch_110_159class.pth` |

Google Drive may limit downloads or show a permission page. An HTML page is
never accepted as a checkpoint. If a file has the wrong size or checksum,
keep it elsewhere and put the correct file at the requested location. Download
the portable desktop runtime from the official Electron link printed in the
bootstrap help file if its automatic download fails. For a source-code
failure, download the [repository ZIP](https://github.com/QinzheYang/LEVIRDet-release/archive/refs/heads/main.zip)
and place its extracted contents directly in the selected project folder,
alongside the existing checkpoints and `docker/` directory.

All expected sizes, hashes, model/config associations, and runtime URLs are
stored in [assets.json](assets.json). The standalone inference checkpoints
already include DINO and GSD parameters; their separate initialization weights
and dataset annotation files are not required for this demo.

## For release maintainers

Publish both root launchers and the complete `fast_demo/` directory together
with the existing `demo/`, `mmdet/`, `configs/`, and other project source on the
repository's `main` branch. In particular, publish **all files under
`fast_demo/launcher/`**, including the prerequisite scripts and the three
artwork files in `launcher/assets/`, plus `fast_demo/assets.json`. A standalone
downloaded CMD or shell script fetches this minimal launcher first; it cannot
start until those files are publicly available. Full project code is fetched
later through the launcher. Keep the existing `fast_demo/desktop/`, `static/`,
and `server.py` files for the inference window.

Do not commit runtime caches, `.levir-launcher`, `.levir-launcher-v3`, `.fast_demo_setup`, downloaded
checkpoints, Docker archives, or users' inference outputs.

When replacing a checkpoint or runtime, update its size and SHA-256 in
`assets.json`; do not reuse an old checksum with new file contents. The two
Google Drive weight links are **159 classes first, 30 classes second** in the
order originally supplied; the manifest binds them to the verified filenames.

The desktop shell is Electron with context isolation and sandboxing enabled.
Inference uses the existing `demo/image_demo.py` without changing the model or
prediction code. Local protocol tests can be run in a Python environment with
Pillow: `python fast_demo/test_server.py`.
