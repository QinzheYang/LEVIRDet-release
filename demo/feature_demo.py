"""Run from the repository root in the existing LEVIR Docker/Conda environment.

python demo/feature_demo.py path/to/image.jpg --weights epoch_117.pth
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levir_features import load_model, load_image, vis


def main():
    parser = argparse.ArgumentParser(description='Extract spatial features using LEVIRDetNet-30.')
    parser.add_argument('image', help='Local image path')
    parser.add_argument('--weights', default='epoch_117.pth')
    parser.add_argument('--device', default=None, help='Defaults to CUDA when available, otherwise CPU')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--out-dir', default='work_dirs/features')
    args = parser.parse_args()

    model = load_model(args.weights, device=args.device, image_size=args.image_size)
    img = load_image(args.image)
    feature = model(img)
    out = Path(args.out_dir)
    vis(feature, out / 'feature_pca.png', size=model.image_size)

    import torch
    torch.save(feature.cpu(), out / 'features.pt')
    print(f'Feature shape: {tuple(feature.shape)}')
    print(f'PCA image: {(out / "feature_pca.png").resolve()}')
    print(f'Feature tensor: {(out / "features.pt").resolve()}')


if __name__ == '__main__':
    main()
