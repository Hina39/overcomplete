import pathlib

import numpy as np
import torch
from einops import rearrange
from overcomplete.models import DinoV2, ResNet
from overcomplete.optimization import NMF, ConvexNMF, SemiNMF
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader
transform_resnet = Compose(
    [transforms.Resize(
    (224, 224),
    interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ]
)

transform_dino = Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder(
    root=pathlib.Path("data/imagenet/train/"), transform=transform_resnet
)

model = ResNet(device='cuda')

z_dict = {}
dictionary_all = []
for class_name, class_idx in tqdm(dataset.class_to_idx.items()):
    # 指定クラスのサンプルのインデックスを収集する
    indices = [i for i, (_, target) in enumerate(dataset.samples) if target == class_idx]
    if not indices:
        continue  # サンプルがなければスキップ！
    
    # 該当クラスのSubsetを作成してデータローダーを用意
    subset = torch.utils.data.Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    images_list = []
    for images, _ in subset_loader:
        images_list.append(images)

    # バッチごとに連結してGPUへ
    images_class = torch.cat(images_list, dim=0).cuda()
    print(images_class.shape) # torch.Size([1300, 3, 224, 224])
    # forward_featuresで特徴抽出
    activations = model.forward_features(images_class)
    print(activations.shape) # torch.Size([1300, 2048, 7, 7])

    Activations = rearrange(activations, 'n c h w -> (n h w) c')
    print(f"クラス {class_name}: Activations shape = {Activations.shape} 😄")

    nmf = NMF(nb_concepts=10, solver='hals', device='cuda', verbose=True)
    Z, D = nmf.fit(torch.relu(Activations))

    print(Z.shape, D.shape) # torch.Size([15680, 10]) # torch.Size([10, 2048])

    z_dict[class_name] = Z.cpu().detach().numpy()
    dictionary_all.append(D)

np.savez("outputs/nmf/z_dict.npz", **z_dict) # 辞書を保存

dictionary_all_cat = torch.cat(dictionary_all, dim=0)  # 辞書は横方向に連結する場合
np.savez("outputs/nmf/dictionary.npz", dictionary_all_cat.cpu().detach().numpy())
print(dictionary_all_cat.shape)
print("Finish!🤩")



"""
ここから下はクラスごとの計算を考慮しない場合のコード
"""

# dataloader = DataLoader(
#     dataset,
#     batch_size=128,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True,
# )

# model = ResNet(device='cuda')
# # to_pil = transforms.ToPILImage()

# images = []
# for i, (image, _) in enumerate(dataloader):
#     if i == 10:
#         break
#     print(image.shape)
#     images.append(image)

# images = torch.cat(images, dim=0).cuda()  # concatenate along batch dimension
# print(images.shape)

# # ok then forward and flatten to get the tokens
# Activations = model.forward_features(images)
# print(Activations.shape)

# # DinoV2
# #  Activations = rearrange(Activations, 'n t d -> (n t) d')

# # ResNet
# Activations = rearrange(Activations, 'n c h w -> (n h w) c')
# print(Activations.shape)

# nmf = NMF(nb_concepts=10, solver='hals', device='cuda', verbose=True)
# Z, D = nmf.fit(torch.relu(Activations))

# print(Z.shape, D.shape) # torch.Size([15680, 10]) # torch.Size([10, 2048])