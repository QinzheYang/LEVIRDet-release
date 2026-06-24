# LEVIRDetNet

LEVIRDetNet is a scale-hierarchy-aware detection foundation model for universal
remote sensing object detection.

## Core Components

1. **Online GSD Predictor**
   - Predicts visual Ground Sampling Distance from the input image.
   - Helps the detector adapt to scale variation across sensors and resolutions.

2. **GSD-guided Query Embedding and Selection**
   - Uses GSD-conditioned query modulation and dynamic query allocation.
   - Adapts the number and distribution of detection queries to image scale.

3. **Hierarchy-aware Head**
   - Learns from mixed-granularity category supervision.
   - Supports category transfer across parent, child, and fine-grained labels.

## Planned Release Contents

The model release is planned to include:

- Training code.
- Inference code.
- Evaluation scripts.
- Configuration files.
- Trained LEVIRDetNet checkpoints.
- Demo scripts and visualization utilities.

Artifacts will be released with the final paper.

