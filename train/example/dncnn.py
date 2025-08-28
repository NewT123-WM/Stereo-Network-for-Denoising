import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb
import sys
sys.path.append("../..")

from utils.seeds import random_seed
from utils.load import load_data
from utils.train_utils import train_one_epoch, validate


from model.dncnn import DnCNN


parser = argparse.ArgumentParser(description="Train DnCNN for image denoising")
parser.add_argument('--train_dir', type=str, default='../../data/curve/train_patch.h5')
parser.add_argument('--val_dir', type=str, default='../../data/curve/val.h5')
parser.add_argument('--save_dir', type=str, default='../../weights/curve/dncnn/')
parser.add_argument('--plot_dir', type=str, default='../../plot_data/curve/dncnn/')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--print_step', type=int, default=50)
parser.add_argument('--seeds', type=int, default=100)
parser.add_argument('--cuda', type=int, default=1)
args = parser.parse_args()

# ======================== wandb init  =====================================
data_name = "Curve"
model_name = "DnCNN"
machine_name = "1344_4090"

wandb.init(
    project=data_name,    # 数据集名
    group=model_name,   # 模型名
    name=f"{model_name}-dncnn",
    tags=[machine_name, model_name],
    config={**vars(args)},
    save_code=True,
    notes=""
)
# ======================== wandb init  =====================================

random_seed(args.seeds)

torch.cuda.set_device(args.cuda)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

model = DnCNN().to(device)
# ======================== wandb watch model  =====================================
wandb.watch(model, log="all", log_freq=args.print_step)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, milestones=[int(args.epochs*0.3), int(args.epochs*0.6), int(args.epochs*0.9)], gamma=0.2)


os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# ========= Dataloaders =========
train_loader, val_loader = load_data(train_path=args.train_dir,
                                     val_path=args.val_dir,
                                     train_batch_size=args.batch_size,
                                     val_batch_size=1)

best_model_save_path = os.path.join(args.save_dir, 'best.pth')

metrics = {'train_loss': [], 'val_loss': [], 'train_snr': [], 'val_snr': []}

best_snr = float('-inf')

start_time = time.time()
since = time.time()
best_epoch = 0
for epoch in tqdm(range(1, args.epochs + 1), leave=False):
    train_loss, train_snr = train_one_epoch(model, device, train_loader, optimizer, criterion)
    val_loss, val_snr = validate(model, device, val_loader, criterion)
    scheduler.step()

    metrics['train_loss'].append(float(train_loss))
    metrics['train_snr'].append(float(train_snr))
    metrics['val_loss'].append(float(val_loss))
    metrics['val_snr'].append(float(val_snr))

    # ========================wandb记录 =====================================
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_snr': train_snr,
        'val_loss': val_loss,
        'val_snr': val_snr,
        'best_snr': best_snr
    })
    # ========================wandb记录 =====================================

    if val_snr > best_snr:
        best_snr = val_snr
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_save_path)
        # 使用wandb artifact管理权重
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(best_model_save_path)
        wandb.log_artifact(artifact)

    if epoch % args.print_step == 0:
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val SNR={val_snr:.2f} dB, ", end='  ')
        time_elapsed = time.time() - since
        print(f"Consuming time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        since = time.time()

        # ============================== 图像对比可视化 ===========================================
        model.eval()
        with torch.no_grad():
            # 从val_loader中取第一个batch
            val_inputs, val_labels = next(iter(val_loader))
            val_inputs = val_inputs.unsqueeze(1).to(device)  # [B, 1, H, W]
            val_labels = val_labels.unsqueeze(1).to(device)
            val_outputs = model(val_inputs)

            # 只显示前4张（你也可以自定义数量）
            n_show = min(4, val_inputs.shape[0])
            image_table = wandb.Table(columns=["noisy", "denoised", "clean"])
            for i in range(n_show):
                noisy = val_inputs[i].detach().cpu().numpy().squeeze()
                denoised = val_outputs[i].detach().cpu().numpy().squeeze()
                clean = val_labels[i].detach().cpu().numpy().squeeze()
                # 保证是[H, W]，如果有归一化可在这里反归一化。 如果需要彩色图像，只需保证维度是[H,W,3]即可
                image_table.add_data(
                    wandb.Image(noisy, caption="noisy"),
                    wandb.Image(denoised, caption="denoised"),
                    wandb.Image(clean, caption="clean"),
                )
            wandb.log({f"val_image_compare_epoch{epoch}": image_table})
        # ============================== 图像对比可视化 ===========================================

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

    with open(os.path.join(args.plot_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

# ========  wandb 结束 ===============
wandb.finish()

total_time = time.time() - start_time
print(f"Best Epoch:{best_epoch}, Best Val SNR:{best_snr:.2f} dB")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

