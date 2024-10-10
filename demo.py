from argparse import ArgumentParser
import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import lpips
import torch
from torch.utils.data import DataLoader

from dataset import DIV2K
from models import SRResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
lpips_loss = lpips.LPIPS(net="alex").to("cpu")


def load_model(model_path: str, scale_factor: int) -> SRResNet:
    model = SRResNet(n_upsamples=scale_factor).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def postprocess_image(generated: torch.Tensor) -> np.ndarray:
    generated = generated.cpu().numpy()  # CHW to HWC
    x = generated
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    x = generated * STD[:, None, None] + MEAN[:, None, None]
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)  # Normalize to [0, 255]
    return x.transpose(1, 2, 0)


def metrics(target, generated):
    # Calculate PSNR
    psnr_val = cv2.PSNR(generated, target)

    # Calculate SSIM
    ssim_val = ssim(target, generated, multichannel=True, channel_axis=2)

    # Calculate LPIPS
    generated = generated.transpose(2, 0, 1)
    target = target.transpose(2, 0, 1)
    lpips_val = lpips_loss(
        torch.tensor(generated).cpu().unsqueeze(0),
        torch.tensor(target).cpu().unsqueeze(0),
    ).item()

    return psnr_val, ssim_val, lpips_val


def inference(model: SRResNet, val_loader: DataLoader) -> None:
    with torch.no_grad():
        for batch_i, data in enumerate(val_loader):
            low_res_batch, high_res_batch = data

            low_res_batch = low_res_batch.to(DEVICE)
            high_res_batch = high_res_batch.to(DEVICE)

            # Генерация высококачественных изображений
            generated_batch = model(low_res_batch)

            for i, generated in enumerate(generated_batch):
                high_res_image = high_res_batch[i]
                high_res_image = postprocess_image(high_res_image)

                low_res_image = low_res_batch[i]
                low_res_image = postprocess_image(low_res_image)
                img_resized = cv2.resize(
                    low_res_image,
                    (high_res_image.shape[0], high_res_image.shape[1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                output_image = postprocess_image(generated)
                output_path = f"./output/image_{batch_i}_{i}_triple.png"
                result = np.hstack((high_res_image, img_resized, output_image))
                metrics_ = metrics(high_res_image, img_resized)
                print(
                    "Metrics for linear",
                    f"psnr {metrics_[0]}, ssim {metrics_[1]}, lpips {metrics_[2]}",
                )
                metrics_ = metrics(high_res_image, output_image)
                print(
                    "Metrics for SRResNet",
                    f"psnr {metrics_[0]}, ssim {metrics_[1]}, lpips {metrics_[2]}",
                )
                cv2.imwrite(output_path, result)


def main(args):
    low_res_size = args.low_res
    high_res_size = args.high_res
    scale_factor = int(math.log2(high_res_size // low_res_size))
    model_path = args.ckpt_path
    model = load_model(model_path, scale_factor)

    div2k_val = DIV2K(
        args.data_path, low_res_size=low_res_size, high_res_size=high_res_size
    )
    val_loader = DataLoader(div2k_val, batch_size=1, shuffle=False)

    inference(model, val_loader)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--low_res", type=int, default=128)
    parser.add_argument("--high_res", type=int, default=256)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ckpt_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
