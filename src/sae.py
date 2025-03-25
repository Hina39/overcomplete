import pathlib

import numpy as np
import torch
from einops import rearrange

# https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/topk_sae.py#L11
from overcomplete.metrics import r2_score
from overcomplete.models import DinoV2, ResNet
from overcomplete.sae import TopKSAE, train_sae
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
    root=pathlib.Path("data/imagenet/train"), transform=transform_resnet
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

    sae = TopKSAE(Activations.shape[-1], nb_concepts=10, top_k=2, device='cuda')

    dataloader = torch.utils.data.DataLoader(TensorDataset(Activations), batch_size=1024, shuffle=True)
    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

    def criterion(x, x_hat, pre_codes, codes, dictionary):
        """saeの損失関数."""
        mse = (x - x_hat).square().mean()
        return mse

    logs = train_sae(sae, dataloader, criterion, optimizer, nb_epochs=20, device='cuda')
    # print(logs)

    sae = sae.eval()
    with torch.no_grad():
        # codesがZにあたる＝スパースな組み合わせ係数
        pre_codes, z_topk = sae.encode(Activations)
        # Zを保存
        # np.savez("z.npz", z_topk.cpu().detach().numpy())
        z_dict[class_name] = z_topk.cpu().detach().numpy() # クラスのラベルをキーにして保存
        print(z_topk.shape) # torch.Size([63700, 10])
        # print(z_topk)

        # ここの出力はよくわからん
        _, codes, recons = sae(Activations)
        print('R2', r2_score(Activations, recons).item())

    # 辞書はこのようにして獲得できる
    dictionary = sae.get_dictionary()
    dictionary_all.append(dictionary)
    # 辞書を保存　(GPU → CPU → NumPy変換)
    # np.savez("dictionary.npz", dictionary.cpu().detach().numpy())
    print(dictionary.shape) #(コンセプトの数,　特徴量の次元数)

np.savez("outputs/topk_sae/z_dict.npz", **z_dict) # 辞書を保存

dictionary_all_cat = torch.cat(dictionary_all, dim=0)  # 辞書は横方向に連結する場合
np.savez("outputs/topk_sae/dictionary.npz", dictionary_all_cat.cpu().detach().numpy())
print(dictionary_all_cat.shape)
print("Finish!🤩")


"""
ここから下はクラスごとの計算を考慮しない場合のコード
"""

#例えば、"00000"クラスの画像だけ使いたい場合
# target_class = "00000"
# target_idx = dataset.class_to_idx[target_class]
# dataset.samples = [s for s in dataset.samples if s[1] == target_idx]
# dataset.targets = [target_idx for _ in dataset.samples]

# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True,
# )

# images = []
# for i, (image, _) in enumerate(dataloader):
#     # print(image.shape)
#     images.append(image)


# images = torch.cat(images, dim=0).cuda()  # concatenate along batch dimension
# print(images.shape) # torch.Size([1300, 3, 224, 224])

# # ok then forward and flatten to get the tokens
# Activations = model.forward_features(images)

# # Activations = rearrange(Activations, 'n t d -> (n t) d')
# # print(Activations.shape) # torch.Size([81920, 384])

# Activations = rearrange(Activations, 'n c h w -> (n h w) c')
# print(Activations.shape) # torch.Size([63700, 2048])

# sae = TopKSAE(Activations.shape[-1], nb_concepts=10, top_k=2, device='cuda')

# """
# top_k : int, optional
#         Number of top activations to keep in the latent representation,
#         by default n_components // 10 (sparsity of 90%).
# """

# dataloader = torch.utils.data.DataLoader(TensorDataset(Activations), batch_size=1024, shuffle=True)
# optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

# def criterion(x, x_hat, pre_codes, codes, dictionary):
#   """
#   saeの損失関数
#   """
#   mse = (x - x_hat).square().mean()
#   return mse

# logs = train_sae(sae, dataloader, criterion, optimizer, nb_epochs=20, device='cuda')
# print(logs)

# sae = sae.eval()

# from overcomplete.metrics import r2_score
# with torch.no_grad():
#     # codesがZにあたる＝スパースな組み合わせ係数
#     pre_codes, z_topk = sae.encode(Activations)
#     # Zを保存
#     np.savez("z.npz", z_topk.cpu().detach().numpy())
#     print(z_topk.shape) # torch.Size([63700, 10])
#     print(z_topk)

#     # ここの出力はよくわからん
#     _, codes, recons = sae(Activations)
#     print('R2', r2_score(Activations, recons).item())

# # 辞書はこのようにして獲得できる
# dictionary = sae.get_dictionary()
# # 辞書を保存　(GPU → CPU → NumPy変換)
# np.savez("dictionary.npz", dictionary.cpu().detach().numpy())
# print(dictionary.shape) #(コンセプトの数,　特徴量の次元数)


# encode(x) を呼び出し、pre_codes（活性化関数適用前の出力）と codes（最終的な潜在表現）を取得。
# エンコーダから得られた潜在表現 z を用いて、辞書層で入力の再構成