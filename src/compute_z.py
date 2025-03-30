import pathlib
from multiprocessing import Pool

import numpy as np
import torch
from overcomplete.models import DinoV2, ResNet  # noqa: F401
from scipy.optimize import nnls
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from tqdm import tqdm

# def compute_Z_for_all_images(D_star, X):
#     """D_star: shape (2048, 10000)
#     X:      shape (N, 2048)  # N枚の画像を平均プーリングした特徴
#     出力:   Z: shape (N, 10000) # 各画像ごとに10次元の非負係数ベクトル
#     """
#     Z_list = []
#     for i in tqdm(range(X.shape[0])):
#         x_i = X[i]              # shape=(2048,)
#         z_i, _ = nnls(D_star, x_i)  # shape=(10000,)
#         Z_list.append(z_i)
#     Z = np.stack(Z_list, axis=0)   # shape=(N,10000)
#     return Z

def compute_single_nnls(args):
    """args: (D_star, x_i) のタプル
    d_star: shape (2048, 10000)
    x_i:    shape (2048,)  # 1枚の画像の特徴ベクトル
    出力:   z_i: shape (10000,) # 1枚の画像ごとに10000次元の非負係数ベクトル.
    """
    d_star, x_i = args
    z_i, _ = nnls(d_star, x_i)
    return z_i

def compute_z_for_all_images(d_star, x, num_processes=None):
    """args:
    d_star: shape (2048, 10000)
    X:      shape (N, 2048)  # N枚の画像を平均プーリングした特徴
    出力:   Z: shape (N, 10000) # 各画像ごとに10000次元の非負係数ベクトル.
    """
    # 各画像の特徴を (d_star, x_i) のタプルとしてまとめる
    args = [(d_star, x[i]) for i in range(x.shape[0])]
    
    # multiprocessing.Pool を用いた並列処理
    with Pool(processes=num_processes) as pool:
        # tqdmで進捗表示
        z_list = list(tqdm(pool.imap(compute_single_nnls, args), total=len(args)))
    
    z = np.stack(z_list, axis=0)   # shape=(N,10000)
    return z

def compute_z_for_single_image(d_star, x):
    """D_star: shape (2048, 10000)
    x:      shape (2048,) - 画像1枚の特徴ベクトル.
    """
    z, _ = nnls(d_star, x)  # z は (10000,) の非負解
    return z

# 既存のヘルパー関数
def torch_to_numpy(tensor):
    """PyTorchのテンソルをNumPy配列に変換します。
    GPU上のテンソルをCPUに移動し、NumPy配列に変換します。
    入力:
        tensor: torch.Tensor - PyTorchのテンソル
    出力:
        numpy.ndarray - NumPy配列.
    """
    try:
        return tensor.detach().cpu().numpy()
    except Exception:
        return np.array(tensor)

def select_patch_from_activations(activations, patch_index):
    """CNNの出力から指定のパッチを選択します.
    
    入力:
      activations: torch.Tensor, shape (N, C, H, W)
      patch_index: tuple of int, (i, j) で選択する空間位置
    出力:
      selected: torch.Tensor, shape (N, C) 
                指定したパッチの特徴（各画像につき1パッチ分の特徴）.
    """
    N, C, H, W = activations.shape # noqa: N806
    i, j = patch_index
    if not (0 <= i < H and 0 <= j < W):
        raise ValueError(f"指定したパッチインデックス {patch_index} は有効な範囲 (H, W)=({H},{W}) ではありません。")
    return activations[:, :, i, j]


# --- 使用例 ---
if __name__ == '__main__':

    # デバイス設定（GPUが使える場合は 'cuda'）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        root=pathlib.Path("data/imagenet/val/"), transform=transform_resnet
    )

    model = ResNet(device='cuda')
    model.eval().to(device)

    D = np.load("outputs/nmf/dictionary.npz")["arr_0"]
    print(D.shape)  # (10000, 2048)

    z_all = []
    for class_name, class_idx in tqdm(dataset.class_to_idx.items()):
        # 指定クラスのサンプルのインデックスを収集する
        indices = [i for i, (_, target) in enumerate(dataset.samples) if target == class_idx]
        if not indices:
            continue  # サンプルがなければスキップ！
        
        # 該当クラスのSubsetを作成してデータローダーを用意
        subset = torch.utils.data.Subset(dataset, indices[:20])
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

        activations_gap = activations.mean(dim=[2, 3])
        print(f"クラス {class_name}: activations_gap shape = {activations_gap.shape} 😄")

        # Activations = rearrange(activations, 'n c h w -> (n h w) c')
        # print(f"クラス {class_name}: Activations shape = {Activations.shape} 😄")

        activation_np = torch_to_numpy(activations_gap)  # GPUテンソルをCPU/NumPyに変換
        z = compute_z_for_all_images(D.T, activation_np)
        print("z shape:", z.shape)  # z shape: (5, 10000)
        z_all.append(z)
    z_all = np.concatenate(z_all, axis=0)
    np.savez("outputs/nmf/z_all_imagenet.npz", z=z_all)
    print("outputs/nmf/z_all_imagenet shape:", z_all.shape) 