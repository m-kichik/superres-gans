In this repository we reproduce the HRResNet model [arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802).

**Train**
```bash
python train.py --low_res=128 --high_res=256 --train_data_path=path/to/train/data --val_data_path=path/to/val/data --exp_name{optionally}=exp_name
```
_exp_name_ is used to name saved weights and initialize wandb experiment name.

**Evaluate**
```bash
python train.py --low_res=128 --high_res=256 --val_data_path=path/to/val/data --ckpt_path=path/to/checkpoint
```
