import os
import copy
import numpy as np
from PIL import Image
import zipfile

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.utils.prune as prune

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torch.nn.functional as F



# ======================================================
# DDP SETUP
# ======================================================
def setup_ddp():
    import os
    import torch.distributed as dist

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    return rank, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ======================================================
# CITYSCAPES LABEL MAPPING
# ======================================================
ID2TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}


def convert_label_to_train(label_path, save_path):
    label = np.array(Image.open(label_path))
    train = np.full(label.shape, 255, dtype=np.uint8)

    for label_id, train_id in ID2TRAINID.items():
        train[label == label_id] = train_id

    Image.fromarray(train).save(save_path)


def convert_split(gt_root, split):
    split_dir = os.path.join(gt_root, split)
    for city in tqdm(os.listdir(split_dir), desc=f"Converting {split}"):
        city_dir = os.path.join(split_dir, city)
        for file in os.listdir(city_dir):
            if file.endswith("_gtFine_labelIds.png"):
                label_path = os.path.join(city_dir, file)
                train_path = label_path.replace(
                    "_gtFine_labelIds.png",
                    "_gtFine_trainIds.png"
                )
                if not os.path.exists(train_path):
                    convert_label_to_train(label_path, train_path)

def unzip_if_needed(zip_path, extract_to):
    if os.path.exists(extract_to):
        print(f"{extract_to} already exists, skipping unzip.")
        return

    print(f"Extracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# ======================================================
# DATASET
# ======================================================
class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, size=(384, 768)):
        self.samples = []
        self.size = size

        for city in os.listdir(img_root):
            img_city_dir = os.path.join(img_root, city)
            mask_city_dir = os.path.join(mask_root, city)

            if not os.path.isdir(mask_city_dir):
                continue

            for fname in os.listdir(img_city_dir):
                if not fname.endswith("_leftImg8bit.png"):
                    continue

                img_path = os.path.join(img_city_dir, fname)
                mask_path = os.path.join(
                    mask_city_dir,
                    fname.replace("_leftImg8bit.png", "_gtFine_trainIds.png")
                )

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        self.img_transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        mask = Image.open(mask_path)
        mask = T.functional.resize(
            mask,
            self.size,
            interpolation=T.InterpolationMode.NEAREST
        )
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


#======================================================
#MODEL (UNET)
#======================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class UNetResNet50(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()

        backbone = models.resnet50(pretrained=pretrained)

        # Encoder
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 256
        self.layer2 = backbone.layer2  # 512
        self.layer3 = backbone.layer3  # 1024
        self.layer4 = backbone.layer4  # 2048

        # Decoder
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)

        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)

        out = F.interpolate(d2, size=input_size, mode="bilinear", align_corners=False)
        return self.final(out)

#class LightUNet(nn.Module):
#    def __init__(self, num_classes=19):
#        super().__init__()
#
#        # Encoder (molto leggero)
#        self.enc1 = ConvBlock(3, 32)
#        self.enc2 = ConvBlock(32, 64)
#        self.enc3 = ConvBlock(64, 128)
#
#        self.pool = nn.MaxPool2d(2)
#
#        # Bottleneck
#        self.bottleneck = ConvBlock(128, 256)
#
#        # Decoder
#        self.dec3 = DecoderBlock(256, 128, 128)
#        self.dec2 = DecoderBlock(128, 64, 64)
#        self.dec1 = DecoderBlock(64, 32, 32)
#
#        # Head
#        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
#
#    def forward(self, x):
#        input_size = x.shape[-2:]
#
#        e1 = self.enc1(x)
#        e2 = self.enc2(self.pool(e1))
#        e3 = self.enc3(self.pool(e2))
#
#        b = self.bottleneck(self.pool(e3))
#
#        d3 = self.dec3(b, e3)
#        d2 = self.dec2(d3, e2)
#        d1 = self.dec1(d2, e1)
#
#        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
#        return self.final(out)
#class ConvBlock(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super().__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True)
#        )
#
#    def forward(self, x):
#        return self.conv(x)
#class DecoderBlock(nn.Module):
#    def __init__(self, in_ch, skip_ch, out_ch):
#        super().__init__()
#        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
#
#    def forward(self, x, skip):
#        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
#        x = torch.cat([x, skip], dim=1)
#        return self.conv(x)


@torch.no_grad()
def infer_single_image(model, image, device):
    model.eval()

    image = image.to(device)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        output = model(image)
        ender.record()
        torch.cuda.synchronize()

        inf_time = starter.elapsed_time(ender) / 1000.0  # sec
    else:
        start = time.perf_counter()
        output = model(image)
        inf_time = time.perf_counter() - start

    pred = torch.argmax(output, dim=1).cpu()

    return pred, inf_time

import matplotlib.pyplot as plt

def plot_sample(image, gt, pred, inf_time, idx, save_dir="vis"):
    os.makedirs(save_dir, exist_ok=True)

    image = image.squeeze().permute(1, 2, 0)
    image = image * torch.tensor([0.229, 0.224, 0.225]) + \
            torch.tensor([0.485, 0.456, 0.406])
    image = image.clamp(0, 1)

    gt = gt.squeeze()
    pred = pred.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="tab20")
    axes[1].set_title("GT")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="tab20")
    axes[2].set_title(f"Pred\n{inf_time*1000:.1f} ms")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{idx}.png")
    plt.close()

# ======================================================
# TRAIN / VAL
# ======================================================
def train_one_epoch(model, loader, optimizer, criterion,device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def pixel_accuracy_val(model, loader,device):
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = torch.argmax(model(images), dim=1)
        valid = masks != 255

        correct += (preds[valid] == masks[valid]).sum()
        total += valid.sum()

    dist.all_reduce(correct)
    dist.all_reduce(total)

    return (correct / total).item()


@torch.no_grad()
def measure_inference_time(model, loader, device, num_batches=20):
    model.eval()
    times = []

    use_cuda = (device.type == "cuda")

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break

        images = images.to(device, non_blocking=use_cuda)

        if use_cuda:
            starter.record()
            _ = model(images)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
        else:
            start = time.perf_counter()
            _ = model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000

        times.append(elapsed_ms)

    return sum(times) / len(times)


def reset_to_initial_weights(model, initial_state):
    for name, module in model.module.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, "weight_mask"):
            assert name in initial_state, f"Missing initial weight for {name}"
            module.weight_orig.data = (
                initial_state[name] * module.weight_mask
            )

def compute_sparsity(model):
    zeros = 0
    total = 0
    for p in model.parameters():
        zeros += (p == 0).sum().item()
        total += p.numel()
    return zeros / total

# ======================================================
# MAIN
# ======================================================

def main():
    rank, local_rank = setup_ddp()

    # =========================
    # PATHS
    # =========================
    DATA_ROOT = "/srv/data/wilson/hpc4ai/home/fnonnis/segmentation/"

    IMG_ROOT_TRAIN = os.path.join(DATA_ROOT, "leftImg8bit", "train")
    IMG_ROOT_VAL   = os.path.join(DATA_ROOT, "leftImg8bit", "val")
    
    MASK_ROOT_TRAIN = os.path.join(DATA_ROOT, "gtFine", "train")
    MASK_ROOT_VAL   = os.path.join(DATA_ROOT, "gtFine", "val")


    dist.barrier()  # ⬅️ tutti aspettano che i file esistano

    # =========================
    # DATASET
    # =========================
    train_dataset = CityscapesDataset(
        img_root=IMG_ROOT_TRAIN,
        mask_root=MASK_ROOT_TRAIN
    )

    val_dataset = CityscapesDataset(
        img_root=IMG_ROOT_VAL,
        mask_root=MASK_ROOT_VAL
    )


    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        sampler=val_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device("cpu")
    print("load data")
    model = UNetResNet50().to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("Loaded model")
    # ======================================================
    # ITERATIVE LOTTERY TICKET
    # ======================================================
    IMP_ROUNDS = 8
    PRUNE_PER_ROUND = 0.2
    EPOCHS_PRE = 5
    EPOCHS_RE = 10
    initial_conv_weights = {}
    
    for name, m in model.module.named_modules():
        if isinstance(m, nn.Conv2d):
            initial_conv_weights[name] = m.weight.detach().clone()

    print("parametri copiati")
    parameters_to_prune = [
        (m, "weight") for m in model.module.modules()
        if isinstance(m, nn.Conv2d)
    ]
    
    results = []
    
    current_sparsity = 0.0
    
#    for epoch in range(EPOCHS_PRE):
#            train_sampler.set_epoch(epoch)
#            train_one_epoch(model, train_loader, optimizer, criterion, device)
#            print(f"epoch{epoch+1}")
#    torch.save(
#                model.module.state_dict(),
#                f"modello_iniziale.pth"
#            )
    load_model_path="/srv/data/wilson/hpc4ai/home/fnonnis/segmentation/modello_iniziale.pth"
    state_dict=torch.load(load_model_path, map_location=device)
    model.module.load_state_dict(state_dict)
    
    print("Loaded model")
    #ricarico i pesi fino al quinto giro 
    for giro in range(5):
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=PRUNE_PER_ROUND)
        load_model_path=f"/srv/data/wilson/hpc4ai/home/fnonnis/segmentation/lottery_ticket_round_{giro}.pth"
        state_dict = torch.load(load_model_path, map_location=device)
        model.module.load_state_dict(state_dict)
        print(f"caricato{load_model_path}")
        print(f"[rank {rank}] sparsity: {compute_sparsity(model)}")
        dist.barrier()
    for round_idx in range(IMP_ROUNDS):
        # ------------------
        # PRUNING (RANK 0)
        # ------------------
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=PRUNE_PER_ROUND)
        print("prunato")
        
        with torch.no_grad():
            for name, m in model.module.named_modules():
                if isinstance(m, nn.Conv2d) and hasattr(m, "weight_orig"):
                    m.weight_orig.copy_(initial_conv_weights[name])

        reset_to_initial_weights(model, initial_conv_weights)
        print(f"sparsity: {compute_sparsity(model)}")
        dist.barrier()
        # ------------------
        # RETRAIN
        # ------------------
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
        for epoch in range(EPOCHS_RE):
            train_sampler.set_epoch(epoch)
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"epoch numero {epoch}")
        print(f"fine Retrain per il giro {round_idx}") 
    
        # ------------------
        # EVALUATION
        # ------------------
        acc = pixel_accuracy_val(model, val_loader, device)
    
        if rank == 0:
            inf_time = measure_inference_time(
                model.module, val_loader, device
            )
    
            sparsity = compute_sparsity(model)
    
            results.append({
                "round": round_idx,
                "sparsity": sparsity,
                "accuracy": acc,
                "inference_ms": inf_time
            })

    
            torch.save(
                model.module.state_dict(),
                f"lottery_ticket_round_{round_idx+5}.pth"
            )
    
            print(
                f"[IMP {round_idx+5}] "
                f"Sparsity: {sparsity:.2f} | "
                f"Acc: {acc:.4f} | "
                f"Inference: {inf_time:.2f} ms"
            )
            
            
            

    dist.barrier()



if __name__ == "__main__":
    main()
