import pathlib

import numpy as np
import torch
from einops import rearrange

# https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/topk_sae.py#L11
from overcomplete.metrics import r2_score
from overcomplete.models import DinoV2, ResNet
from overcomplete.sae import JumpSAE, train_sae
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from tqdm import tqdm
from scipy.optimize import nnls

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

    sae = JumpSAE(Activations.shape[-1], nb_concepts=10, bandwith=1e-2, kernel='silverman', device='cuda')

    dataloader = torch.utils.data.DataLoader(TensorDataset(Activations), batch_size=1024, shuffle=True)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-3)

    desired_sparsity = 0.10
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        # here we directly use the thresholds of the model to control the sparsity
        loss = (x - x_hat).square().mean()

        sparsity = (codes > 0).float().mean().detach()
        if sparsity > desired_sparsity:
            # if we are not sparse enough, increase the thresholds levels
            loss -= sae.thresholds.sum()

        return loss

    logs = train_sae(sae, dataloader, criterion, optimizer, nb_epochs=20, device='cuda')

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
        # _, codes, recons = sae(Activations)
        # print('R2', r2_score(Activations, recons).item())

    # 辞書はこのようにして獲得できる
    dictionary = sae.get_dictionary()
    dictionary_all.append(dictionary)
    # 辞書を保存　(GPU → CPU → NumPy変換)
    # np.savez("dictionary.npz", dictionary.cpu().detach().numpy())
    print(dictionary.shape) #(コンセプトの数,　特徴量の次元数)

    dictionary_np = dictionary.cpu().detach().numpy()
    reconstructed = []
    if len(activations.shape) == 4:
        orig_act = torch.mean(activations, dim=(2, 3))
    orig_act_np = orig_act.cpu().detach().numpy()
    for i in range(orig_act_np.shape[0]):
        # D_np.T の shape は (2048, 10)、act の shape は (2048,)
        u_i, _ = nnls(dictionary_np.T, orig_act_np[i])
        rec_i = np.dot(u_i, dictionary_np)  # 再構成: rec_i = u_i * D_np, shape = (2048,)
        reconstructed.append(rec_i)
    reconstructed = np.array(reconstructed)

    orig_act_tensor = torch.from_numpy(orig_act_np).to(device).float()
    reconstructed_tensor = torch.from_numpy(reconstructed).to(device).float()

    # # ここでunsqueezeして、conv2dが処理できる形に変換するぜ！
    orig_act_tensor = orig_act_tensor.unsqueeze(-1).unsqueeze(-1)         # [N, C, 1, 1]
    reconstructed_tensor = reconstructed_tensor.unsqueeze(-1).unsqueeze(-1)   # [N, C, 1, 1]

    preds_orig = torch.argmax(model.model.head.fc(orig_act_tensor), dim=1).cpu().numpy()
    preds_rec = torch.argmax(model.model.head.fc(reconstructed_tensor), dim=1).cpu().numpy()

    reconstruction_accuracy = np.mean(preds_orig == preds_rec)
    print("Reconstruction Accuracy (NNLS):", reconstruction_accuracy)

    # reconstruction_accuracy と class_name を log に保存するぜ！
    with open("outputs/jump_sae/Reconstruction_Accuracy_epoch_20_log_4_4.txt", "a") as f:
        f.write(f"{class_name}: Reconstruction Accuracy: {reconstruction_accuracy}\n")

np.savez("outputs/jump_sae/epoch_20_z_dict.npz", **z_dict) # 辞書を保存

dictionary_all_cat = torch.cat(dictionary_all, dim=0)  # 辞書は横方向に連結する場合
np.savez("outputs/jump_sae/epoch_20_dictionary.npz", dictionary_all_cat.cpu().detach().numpy())
print(dictionary_all_cat.shape)
print("Finish!🤩")

