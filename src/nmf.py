import pathlib

import numpy as np
import torch
from einops import rearrange
from overcomplete.models import DinoV2, ResNet
from overcomplete.optimization import NMF, ConvexNMF, SemiNMF
from scipy.optimize import nnls  # NNLSによる係数抽出のため
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
# print("Model Keys:")
# print(list(model.model.state_dict().keys()))
# print(model.model) # torch.Size([1000, 2048])

z_dict = {}
dictionary_all = []
all_acc = []
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
    Z, D = nmf.fit(torch.relu(Activations), max_iter=1000)

    print(Z.shape, D.shape) # torch.Size([15680, 10]) # torch.Size([10, 2048])

    D_np = D.cpu().detach().numpy()  # D_np: shape (10, 2048)

    reconstructed = []
    if len(activations.shape) == 4:
        orig_act = torch.mean(activations, dim=(2, 3))
    orig_act_np = orig_act.cpu().detach().numpy()
    for i in range(orig_act_np.shape[0]):
        # D_np.T の shape は (2048, 10)、act の shape は (2048,)
        u_i, _ = nnls(D_np.T, orig_act_np[i])
        rec_i = np.dot(u_i, D_np)  # 再構成: rec_i = u_i * D_np, shape = (2048,)
        reconstructed.append(rec_i)
    reconstructed = np.array(reconstructed)

    # latent_to_logitを使って予測を取得して、再構成前後でどれだけ一致するかチェックするぜ！
    # numpy配列をtorch.Tensorに変換してデバイスに移すぜ！
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
    with open("outputs/nmf/Reconstruction_Accuracy_log_4_4.txt", "a") as f:
        f.write(f"{class_name}: Reconstruction Accuracy: {reconstruction_accuracy}\n")

    z_dict[class_name] = Z.cpu().detach().numpy()
    dictionary_all.append(D)
    all_acc.append(reconstruction_accuracy)

np.savez("outputs/nmf/z_dict_ResNet.npz", **z_dict) # 辞書を保存

dictionary_all_cat = torch.cat(dictionary_all, dim=0)  # 辞書は横方向に連結する場合
np.savez("outputs/nmf/dictionary_ResNet.npz", dictionary_all_cat.cpu().detach().numpy())
print(dictionary_all_cat.shape)
all_acc_mean = sum(np.array(all_acc)) / 1000
np.savez("output/nmf/reconstruction_accuracies_ResNet.npz", accuracies=all_acc_mean)
print("Reconstruction accuracies:", all_acc_mean)
print("Finish!🤩")