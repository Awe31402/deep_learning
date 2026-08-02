"""
MIT 6.5940 EfficientML.ai Lab 3: Neural Architecture Search (神經網路架構搜尋)

從 Lab3_zh.md 整理出的完整實作程式碼：
  - Getting Started : OFA 超網路 + VWW 數據集 (問題 1)
  - Part 1          : 效率預測器 / 準確率預測器 (問題 2-4)
  - Part 2          : 隨機搜尋 + 進化搜尋 (問題 5-10)

執行方式：
    python3 hw3.py                 # 執行全部段落
    python3 hw3.py --part q2 q3    # 只跑指定段落 (見 PARTS)

注意：本檔案需要 MCUNet codebase (mcunet/)、VWW 數據集 (data/vww-s256/)、
超網路權重 (vww_supernet.pth) 與準確率數據集 (acc_datasets/)。
若尚未下載，程式會提示對應的下載指令 (見 prepare_environment)。
"""
import argparse
import copy
import os
import random

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # 以腳本方式執行時不開視窗，圖片直接存檔
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision import datasets, transforms

from mcunet.tinynas.search.accuracy_predictor import (
    AccuracyDataset,
    MCUNetArchEncoder,
)
from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.arch_visualization_helper import draw_arch
from mcunet.utils.pytorch_utils import (
    count_peak_activation_size,
    count_net_flops,
    count_parameters,
)

import warnings
warnings.filterwarnings("ignore")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
data_dir = "data/vww-s256/val"
fig_dir = "lab3_figs"

# 準確率預測器可選的輸入解析度
image_size_list = [96, 112, 128, 144, 160]


# %% ---------------------------------------------------------------------
# 環境準備
# ------------------------------------------------------------------------
SETUP_COMMANDS = """\
# MCUNet codebase + 超網路權重
wget https://www.dropbox.com/s/3y2n2u3mfxczwcb/mcunetv2-dev-main.zip?dl=0
unzip mcunetv2-dev-main.zip* && mv mcunetv2-dev-main/* .
# VWW 數據集
wget https://www.dropbox.com/s/169okcuuv64d4nn/data.zip?dl=0
unzip data.zip*
# 其他相依套件
sudo apt-get install graphviz && pip install thop onnx
"""


def prepare_environment():
    """檢查實驗所需的資料是否齊全，缺少時列出下載指令。"""
    required = {
        "data/vww-s256/val": "VWW 驗證集",
        "vww_supernet.pth": "OFA 超網路權重",
        "acc_datasets": "準確率數據集 ([architecture, accuracy] 對)",
    }
    missing = [f"  - {path} ({desc})" for path, desc in required.items()
               if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "缺少以下檔案：\n" + "\n".join(missing) +
            "\n\n請先在專案目錄執行：\n" + SETUP_COMMANDS
        )
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("viz", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)


def save_fig(name):
    """存檔並關閉目前的 figure，回傳檔案路徑。"""
    path = os.path.join(fig_dir, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


# %% ---------------------------------------------------------------------
# 數據載入器與超網路
# ------------------------------------------------------------------------
def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    # split = 0: real val set, split = 1: holdout validation set
    assert split in [0, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {"num_workers": min(8, os.cpu_count()), "pin_memory": False}

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution)
            ),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


def visualize_dataset():
    """瀏覽 VWW 驗證集中的幾張影像。"""
    val_data_loader = build_val_data_loader(data_dir, resolution=128, batch_size=1)

    vis_x, vis_y = 2, 3
    fig, axs = plt.subplots(vis_x, vis_y)

    num_images = 0
    for data, label in val_data_loader:
        img = np.array((((data + 1) / 2) * 255).numpy(), dtype=np.uint8)
        img = img[0].transpose(1, 2, 0)
        label_text = "No person" if label.item() == 0 else "Person"
        axs[num_images // vis_y][num_images % vis_y].imshow(img)
        axs[num_images // vis_y][num_images % vis_y].set_title(f"Label: {label_text}")
        axs[num_images // vis_y][num_images % vis_y].set_xticks([])
        axs[num_images // vis_y][num_images % vis_y].set_yticks([])
        num_images += 1
        if num_images > vis_x * vis_y - 1:
            break

    print("VWW 樣本圖片存於:", save_fig("vww_samples.png"))


def build_ofa_network():
    """建構 MCUNetV2 OFA 超網路 (設計空間內含 >1e19 個子網路)。"""
    ofa_network = OFAMCUNets(
        n_classes=2,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.0,
        base_stage_width="mcunet384",
        width_mult_list=[0.5, 0.75, 1.0],
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[0, 1, 2],
        base_depth=[1, 2, 2, 2, 2],
        fuse_blk1=True,
        se_stages=[False, [False, True, True, True], True, True, True, False],
    )

    ofa_network.load_state_dict(
        torch.load("vww_supernet.pth", map_location="cpu")["state_dict"], strict=True
    )
    return ofa_network.to(device)


def evaluate_sub_network(ofa_network, cfg, image_size=None):
    """從超網路擷取子網路並在 VWW 上評估 (無需重新訓練)。"""
    if "image_size" in cfg:
        image_size = cfg["image_size"]
    batch_size = 128
    # step 1. sample the active subnet with the given config.
    ofa_network.set_active_subnet(**cfg)
    # step 2. extract the subnet with corresponding weights.
    subnet = ofa_network.get_active_subnet().to(device)
    # step 3. calculate the efficiency stats of the subnet.
    peak_memory = count_peak_activation_size(subnet, (1, 3, image_size, image_size))
    macs = count_net_flops(subnet, (1, 3, image_size, image_size))
    params = count_parameters(subnet)
    # step 4. perform BN parameter re-calibration.
    calib_bn(subnet, data_dir, batch_size, image_size)
    # step 5. define the validation dataloader.
    val_loader = build_val_data_loader(data_dir, image_size, batch_size)
    # step 6. validate the accuracy.
    acc = validate(subnet, val_loader)
    return acc, peak_memory, macs, params


def visualize_subnet(cfg, out_name="subnet"):
    """把子網路架構畫出來並存檔。"""
    draw_arch(cfg["ks"], cfg["e"], cfg["d"], cfg["image_size"], out_name="viz/subnet")
    im = Image.open("viz/subnet.png")
    im = im.rotate(90, expand=1)
    plt.figure(figsize=(im.size[0] / 250, im.size[1] / 250))
    plt.axis("off")
    plt.imshow(im)
    return save_fig(f"{out_name}.png")


# %% ---------------------------------------------------------------------
# 問題 1 (5 分)：設計空間探索
# ------------------------------------------------------------------------
# 回答：
#   在設計空間的四個維度 (輸入解析度 / 通道寬度 width_mult / 卷積核大小 ks /
#   擴展比例 e 與深度 d) 之中，**輸入解析度是對準確率影響最大的維度**：
#   把解析度從 96 提高到 160，準確率通常有數個百分點的提升，而 MACs 則約略
#   以解析度平方成長 (峰值記憶體亦然)。
#   相對地，在固定解析度下把最小子網路換成最大子網路 (ks 3->7、e 3->6、
#   d 全開、width 0.5->1.0)，準確率大約只從 ~83.6% 提升到 ~88.7%，但參數量與
#   MACs 卻大上數倍 —— 也就是說 ks / e 這兩個維度的邊際效益最低，
#   而深度 d 與通道寬度的影響介於中間。
#   結論：在 MCU 這種資源極度受限的場景，「先給足解析度、再壓縮 ks/e」通常
#   比「維持大模型但降解析度」更划算，這也是後面搜尋演算法會自動找到的取捨。
# ------------------------------------------------------------------------
def question_1_design_space_exploration(ofa_network, image_size=96):
    """隨機 / 最大 / 最小子網路的準確率比較，另外掃描不同輸入解析度。"""
    print("\n=== 問題 1：設計空間探索 ===")

    cfg = ofa_network.sample_active_subnet(
        sample_function=random.choice, image_size=image_size
    )
    acc, _, macs, params = evaluate_sub_network(ofa_network, cfg)
    print("隨機子網路架構圖:", visualize_subnet(cfg, "q1_random_subnet"))
    print(f"The accuracy of the sampled subnet: #params={params/1e6: .1f}M, "
          f"#MACs={macs/1e6: .1f}M, accuracy={acc: .1f}%.")

    largest_cfg = ofa_network.sample_active_subnet(
        sample_function=max, image_size=image_size
    )
    acc, _, macs, params = evaluate_sub_network(ofa_network, largest_cfg)
    print("最大子網路架構圖:", visualize_subnet(largest_cfg, "q1_largest_subnet"))
    print(f"The largest subnet: #params={params/1e6: .1f}M, "
          f"#MACs={macs/1e6: .1f}M, accuracy={acc: .1f}%.")

    smallest_cfg = ofa_network.sample_active_subnet(
        sample_function=min, image_size=image_size
    )
    acc, peak_memory, macs, params = evaluate_sub_network(ofa_network, smallest_cfg)
    print("最小子網路架構圖:", visualize_subnet(smallest_cfg, "q1_smallest_subnet"))
    print(f"The smallest subnet: #params={params/1e6: .1f}M, "
          f"#MACs={macs/1e6: .1f}M, peak memory={peak_memory/1024: .0f}KB, "
          f"accuracy={acc: .1f}%.")

    # 額外實驗：固定架構、只改輸入解析度，觀察解析度這個維度的影響力。
    print("\n--- 解析度掃描 (固定為最小架構) ---")
    for res in image_size_list:
        cfg = ofa_network.sample_active_subnet(sample_function=min, image_size=res)
        acc, peak_memory, macs, _ = evaluate_sub_network(ofa_network, cfg)
        print(f"resolution={res:3d}: accuracy={acc: .1f}%, "
              f"#MACs={macs/1e6: .1f}M, peak memory={peak_memory/1024: .0f}KB")


# %% ---------------------------------------------------------------------
# 問題 2 (10 分)：效率預測器
# ------------------------------------------------------------------------
class AnalyticalEfficiencyPredictor:
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        self.net.set_active_subnet(**spec)
        subnet = self.net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        ############### YOUR CODE STARTS HERE ###############
        # Hint: take a look at the `evaluate_sub_network` function above.
        # Hint: the data shape is (batch_size, input_channel, image_size, image_size)
        image_size = spec["image_size"]
        data_shape = (1, 3, image_size, image_size)
        macs = count_net_flops(subnet, data_shape)
        peak_memory = count_peak_activation_size(subnet, data_shape)
        ################ YOUR CODE ENDS HERE ################

        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024)

    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            # if the constraint is not specified, we just continue
            if key not in target:
                continue
            # if we exceed the constraint, just return false.
            if measured[key] > target[key]:
                return False
        # no constraint violated, return true.
        return True


def question_2_efficiency_predictor(ofa_network, efficiency_predictor, image_size=96):
    """驗證分析式效率預測器：結果應與 evaluate_sub_network 一致。"""
    print("\n=== 問題 2：效率預測器 ===")

    smallest_cfg = ofa_network.sample_active_subnet(
        sample_function=min, image_size=image_size
    )
    eff_smallest = efficiency_predictor.get_efficiency(smallest_cfg)

    largest_cfg = ofa_network.sample_active_subnet(
        sample_function=max, image_size=image_size
    )
    eff_largest = efficiency_predictor.get_efficiency(largest_cfg)

    print("Efficiency stats of the smallest subnet:", eff_smallest)
    print("Efficiency stats of the largest subnet:", eff_largest)
    return eff_smallest, eff_largest


# %% ---------------------------------------------------------------------
# 問題 3 (10 分)：準確率預測器 (MLP)
# ------------------------------------------------------------------------
class AccuracyPredictor(nn.Module):
    def __init__(
        self,
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device="cuda:0",
    ):
        super(AccuracyPredictor, self).__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        layers = []

        ############### YOUR CODE STARTS HERE ###############
        # Let's build an MLP with n_layers layers.
        # Each layer (nn.Linear) has hidden_size channels and
        # uses nn.ReLU as the activation function.
        # Hint: You can assume that n_layers is fixed to be 3, for simplicity.
        # Hint: the input dimension of the first layer is not hidden_size.
        #       use self.arch_encoder.n_dim to get the input dimension
        for i in range(self.n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        self.arch_encoder.n_dim if i == 0 else self.hidden_size,
                        self.hidden_size,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        ################ YOUR CODE ENDS HERE ################
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
            print("Loaded checkpoint from %s" % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_acc(self, arch_dict_list):
        X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        return self.forward(X)


def build_arch_encoder(ofa_network):
    return MCUNetArchEncoder(
        image_size_list=image_size_list,
        base_depth=ofa_network.base_depth,
        depth_list=ofa_network.depth_list,
        expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list,
    )


def visualize_acc_dataset_sample(arch_encoder, train_loader):
    """列印一筆訓練樣本，展示 one-hot 架構編碼的細節。"""
    for (data, label) in train_loader:
        data = data.to(device)
        label = label.to(device)
        print("=" * 100)
        # dummy pass to print the divided encoding
        arch_encoding = arch_encoder.feature2arch(
            data[0].int().cpu().numpy(), verbose=False
        )
        # print out the architecture encoding process in detail
        arch_encoding = arch_encoder.feature2arch(
            data[0].int().cpu().numpy(), verbose=True
        )
        print("架構圖:", visualize_subnet(arch_encoding, "q3_acc_dataset_sample"))
        print("The accuracy of this subnet on the holdout validation set is: "
              f"{(label[0] * 100): .1f}%.")
        break


# %% ---------------------------------------------------------------------
# 問題 4 (10 分)：訓練準確率預測器
# ------------------------------------------------------------------------
def question_4_train_acc_predictor(
    acc_predictor, train_loader, valid_loader, base_acc, n_epochs=10,
    checkpoint_path=None,
):
    print("\n=== 問題 4：訓練準確率預測器 ===")
    criterion = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(acc_predictor.parameters())
    # the default value is zero
    acc_predictor.base_acc.data += base_acc
    for epoch in tqdm(range(n_epochs)):
        acc_predictor.train()
        for (data, label) in tqdm(
            train_loader, desc="Epoch%d" % (epoch + 1), position=0, leave=True
        ):
            # step 1. Move the data and labels to device (cuda:0).
            data = data.to(device)
            label = label.to(device)
            ############### YOUR CODE STARTS HERE ###############
            # step 2. Run forward pass.
            pred = acc_predictor(data)
            # step 3. Calculate the loss.
            loss = criterion(pred, label)
            # step 4. Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ################ YOUR CODE ENDS HERE ################

        acc_predictor.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc="Val", position=0, leave=True) as t:
                for (data, label) in valid_loader:
                    # step 1. Move the data and labels to device (cuda:0).
                    data = data.to(device)
                    label = label.to(device)
                    ############### YOUR CODE STARTS HERE ###############
                    # step 2. Run forward pass.
                    pred = acc_predictor(data)
                    # step 3. Calculate the loss.
                    loss = criterion(pred, label)
                    ############### YOUR CODE ENDS HERE ###############
                    t.set_postfix({"loss": loss.item()})
                    t.update(1)

    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        torch.save(acc_predictor.cpu().state_dict(), checkpoint_path)
        print("Saved accuracy predictor to", checkpoint_path)
    return acc_predictor.to(device)


def plot_acc_predictor_correlation(acc_predictor, valid_loader):
    """繪製「預測準確率 vs. 真實準確率」相關性圖，理想上應接近 y = x。"""
    predicted_accuracies = []
    ground_truth_accuracies = []
    acc_predictor = acc_predictor.to(device)
    acc_predictor.eval()
    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Val") as t:
            for (data, label) in valid_loader:
                data = data.to(device)
                label = label.to(device)
                pred = acc_predictor(data)
                predicted_accuracies += pred.cpu().numpy().tolist()
                ground_truth_accuracies += label.cpu().numpy().tolist()
                t.update(1)
                if len(predicted_accuracies) > 200:
                    break
    plt.scatter(predicted_accuracies, ground_truth_accuracies)
    # draw y = x
    min_acc, max_acc = min(predicted_accuracies), max(predicted_accuracies)
    plt.plot([min_acc, max_acc], [min_acc, max_acc], c="red", linewidth=2)
    plt.xlabel("Predicted accuracy")
    plt.ylabel("Measured accuracy")
    plt.title("Correlation between predicted accuracy and real accuracy")
    print("相關性圖存於:", save_fig("acc_predictor_correlation.png"))


# %% ---------------------------------------------------------------------
# 問題 5 (5 分)：隨機搜尋
# ------------------------------------------------------------------------
class RandomSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []
        # sample subnets
        for _ in tqdm(range(n_subnets)):
            sample, efficiency = self.random_valid_sample(constraint)
            subnet_pool.append(sample)
        # predict the accuracy of subnets
        accs = self.accuracy_predictor.predict_acc(subnet_pool)
        ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        # get the index of the best subnet
        best_idx = int(torch.argmax(accs).item())
        ############### YOUR CODE ENDS HERE #################
        # return the best subnet
        return accs[best_idx], subnet_pool[best_idx]


# %% ---------------------------------------------------------------------
# 問題 6 (5 分)：搜尋 + 真實準確率量測
# ------------------------------------------------------------------------
def search_and_measure_acc(agent, constraint, ofa_network, out_name="subnet", **kwargs):
    ############### YOUR CODE STARTS HERE ###############
    # hint: call the search function
    best_info = agent.run_search(constraint, **kwargs)
    ############### YOUR CODE ENDS HERE #################
    # get searched subnet
    ofa_network.set_active_subnet(**best_info[1])
    subnet = ofa_network.get_active_subnet().to(device)
    # calibrate bn
    calib_bn(subnet, data_dir, 128, best_info[1]["image_size"])
    # build val loader
    val_loader = build_val_data_loader(data_dir, best_info[1]["image_size"], 128)
    # measure accuracy
    acc = validate(subnet, val_loader)
    # print best_info
    print(f"Accuracy of the selected subnet: {acc}")
    # visualize model architecture
    print("架構圖:", visualize_subnet(best_info[1], out_name))
    return acc, subnet


def question_5_6_random_search(ofa_network, efficiency_predictor, acc_predictor):
    print("\n=== 問題 5 & 6：隨機搜尋 ===")
    random.seed(1)
    np.random.seed(1)
    nas_agent = RandomSearcher(efficiency_predictor, acc_predictor)

    # MACs-constrained search
    subnets_rs_macs = {}
    for millionMACs in [50, 100]:
        # 注意：原始 notebook 這裡的 key 拼成 `millonMACs`，與 get_efficiency 回傳的
        # `millionMACs` 不符，會讓約束完全失效 (satisfy_constraint 直接 continue)。
        search_constraint = dict(millionMACs=millionMACs)
        print(f"Random search with constraint: MACs <= {millionMACs}M")
        subnets_rs_macs[millionMACs] = search_and_measure_acc(
            nas_agent, search_constraint, ofa_network,
            out_name=f"q6_rs_macs{millionMACs}", n_subnets=300,
        )

    # memory-constrained search
    subnets_rs_memory = {}
    for KBPeakMemory in [256, 512]:
        search_constraint = dict(KBPeakMemory=KBPeakMemory)
        print(f"Random search with constraint: Peak memory <= {KBPeakMemory}KB")
        subnets_rs_memory[KBPeakMemory] = search_and_measure_acc(
            nas_agent, search_constraint, ofa_network,
            out_name=f"q6_rs_mem{KBPeakMemory}", n_subnets=300,
        )

    return subnets_rs_macs, subnets_rs_memory


# %% ---------------------------------------------------------------------
# 問題 7 (20 分)：進化搜尋
# ------------------------------------------------------------------------
class EvolutionSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = copy.deepcopy(sample)

            self.accuracy_predictor.arch_encoder.mutate_resolution(
                new_sample, self.resolution_mutate_prob
            )
            self.accuracy_predictor.arch_encoder.mutate_width(
                new_sample, self.arch_mutate_prob
            )
            self.accuracy_predictor.arch_encoder.mutate_arch(
                new_sample, self.arch_mutate_prob
            )

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2, constraint):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    ############### YOUR CODE STARTS HERE ###############
                    # hint: randomly choose the value from sample1[key] and
                    #       sample2[key], random.choice
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                    ############### YOUR CODE ENDS HERE #################
                else:
                    for i in range(len(new_sample[key])):
                        ############### YOUR CODE STARTS HERE ###############
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )
                        ############### YOUR CODE ENDS HERE #################

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return new_sample, efficiency

    def run_search(self, constraint, **kwargs):
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (acc, sample) tuples
        child_pool = []
        best_info = None
        # generate random population
        for _ in range(self.population_size):
            sample, efficiency = self.random_valid_sample(constraint)
            child_pool.append(sample)

        accs = self.accuracy_predictor.predict_acc(child_pool)
        for i in range(self.population_size):
            population.append((accs[i].item(), child_pool[i]))

        # evolving the population
        with tqdm(total=self.max_time_budget) as t:
            for i in range(self.max_time_budget):
                ############### YOUR CODE STARTS HERE ###############
                # hint: sort the population according to the acc (descending order)
                population = sorted(population, key=lambda x: x[0], reverse=True)
                ############### YOUR CODE ENDS HERE #################

                ############### YOUR CODE STARTS HERE ###############
                # hint: keep topK samples in the population, K = parents_size
                # the others are discarded.
                population = population[:parents_size]
                ############### YOUR CODE ENDS HERE #################

                # update best info
                acc = population[0][0]
                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = population[0]
                else:
                    best_valids.append(best_valids[-1])

                child_pool = []
                for j in range(mutation_numbers):
                    # randomly choose a sample
                    par_sample = population[np.random.randint(parents_size)][1]
                    # mutate this sample
                    new_sample, efficiency = self.mutate_sample(par_sample, constraint)
                    child_pool.append(new_sample)

                for j in range(self.population_size - mutation_numbers):
                    # randomly choose two samples
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # crossover
                    new_sample, efficiency = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    child_pool.append(new_sample)
                # predict accuracy with the accuracy predictor
                accs = self.accuracy_predictor.predict_acc(child_pool)
                for j in range(self.population_size):
                    population.append((accs[j].item(), child_pool[j]))

                t.set_postfix({"best_pred_acc": best_valids[-1]})
                t.update(1)

        return best_info


# %% ---------------------------------------------------------------------
# 問題 8 (10 分)：執行進化搜尋並調參
# ------------------------------------------------------------------------
# 調參後的設定 (相較預設值的主要調整)：
#   - population_size 10 -> 100、max_time_budget 10 -> 100：
#     族群太小時多樣性不足，很快就退化成局部搜尋；擴大族群與世代數後
#     預測準確率明顯上升，而由於評估靠的是預測器 (毫秒級)，成本仍可接受。
#   - parent_ratio 0.1 -> 0.25：只留 1 個父代 (10 * 0.1) 會讓交叉退化成複製，
#     留 25% 才能保持基因多樣性。
#   - mutation_ratio 0.1 -> 0.5：變異/交叉各半，兼顧「局部微調」與「大步跳躍」。
#   - resolution_mutate_prob 0.1 -> 0.5：解析度是影響準確率/效率最劇烈的維度，
#     給它較高的變異機率能更快找到「解析度大 + 架構瘦」這類最佳取捨點。
#
# 發現 (問題 8 回答)：
#   在相同的效率約束下，進化搜尋找到的子網路準確率普遍優於隨機搜尋，
#   而且所需的候選數量少得多 (樣本效率高)：隨機搜尋是均勻地在 >1e19 的空間中
#   抽樣，落在高準確率區域的機率極低；進化搜尋則不斷從當前最佳解附近繼續探索。
#   另外可以觀察到：搜尋結果幾乎總是把解析度推到約束允許的上限，再靠縮小
#   ks / e / d 來把 MACs 與峰值記憶體壓回約束內 —— 與問題 1 的結論一致。
# ------------------------------------------------------------------------
EVO_PARAMS = {
    "arch_mutate_prob": 0.1,      # The probability of architecture mutation
    "resolution_mutate_prob": 0.5,  # The probability of resolution mutation
    "population_size": 100,       # The size of the population
    "max_time_budget": 100,
    "parent_ratio": 0.25,
    "mutation_ratio": 0.5,
}


def question_7_8_evolution_search(ofa_network, efficiency_predictor, acc_predictor):
    print("\n=== 問題 7 & 8：進化搜尋 ===")
    random.seed(1)
    np.random.seed(1)

    nas_agent = EvolutionSearcher(efficiency_predictor, acc_predictor, **EVO_PARAMS)

    # MACs-constrained search
    subnets_evo_macs = {}
    for millionMACs in [50, 100]:
        search_constraint = dict(millionMACs=millionMACs)
        print(f"Evolutionary search with constraint: MACs <= {millionMACs}M")
        subnets_evo_macs[millionMACs] = search_and_measure_acc(
            nas_agent, search_constraint, ofa_network,
            out_name=f"q8_evo_macs{millionMACs}",
        )

    # memory-constrained search
    subnets_evo_memory = {}
    for KBPeakMemory in [256, 512]:
        search_constraint = dict(KBPeakMemory=KBPeakMemory)
        print(f"Evolutionary search with constraint: Peak memory <= {KBPeakMemory}KB")
        subnets_evo_memory[KBPeakMemory] = search_and_measure_acc(
            nas_agent, search_constraint, ofa_network,
            out_name=f"q8_evo_mem{KBPeakMemory}",
        )

    return subnets_evo_macs, subnets_evo_memory


# %% ---------------------------------------------------------------------
# 問題 9 (15 分 + 10 分加分)：真實世界約束下的進化搜尋
#   - [15 分] 250KB, 60M MACs  (目標準確率 >= 92.5%)
#   - [加分]  200KB, 30M MACs  (目標準確率 >= 90%)
# ------------------------------------------------------------------------
# 兩個任務不必共用同一組 evo_params：第二個約束 (30M MACs / 200KB) 的可行域
# 小很多，隨機取樣命中率低，因此把族群縮小一點、世代數拉長，並提高變異機率
# (在可行域內做細緻的局部搜尋比大範圍交叉更有效)。
EVO_PARAMS_Q9_A = {   # 250KB, 60M MACs
    "arch_mutate_prob": 0.1,
    "resolution_mutate_prob": 0.5,
    "population_size": 100,
    "max_time_budget": 100,
    "parent_ratio": 0.25,
    "mutation_ratio": 0.5,
}

EVO_PARAMS_Q9_B = {   # 200KB, 30M MACs (加分題，可行域更窄)
    "arch_mutate_prob": 0.2,
    "resolution_mutate_prob": 0.3,
    "population_size": 64,
    "max_time_budget": 150,
    "parent_ratio": 0.25,
    "mutation_ratio": 0.5,
}


def question_9_real_world_constraints(ofa_network, efficiency_predictor, acc_predictor):
    print("\n=== 問題 9：真實世界約束 ===")
    results = {}

    for (millionMACs, KBPeakMemory), evo_params, tag, target in [
        ((60, 250), EVO_PARAMS_Q9_A, "q9_60M_250KB", 92.5),
        ((30, 200), EVO_PARAMS_Q9_B, "q9_30M_200KB", 90.0),
    ]:
        random.seed(1)
        np.random.seed(1)
        nas_agent = EvolutionSearcher(efficiency_predictor, acc_predictor, **evo_params)
        print(f"Evolution search with constraint: MACs <= {millionMACs}M, "
              f"peak memory <= {KBPeakMemory}KB")
        acc, subnet = search_and_measure_acc(
            nas_agent,
            dict(millionMACs=millionMACs, KBPeakMemory=KBPeakMemory),
            ofa_network,
            out_name=tag,
        )
        status = "PASS" if acc >= target else "FAIL"
        print(f"[{status}] measured accuracy = {acc:.2f}% (target >= {target}%)")
        results[(millionMACs, KBPeakMemory)] = (acc, subnet)
        print("Evolution search finished!")

    return results


# %% ---------------------------------------------------------------------
# 問題 10 (10 分)：設計空間的可行性分析
#   A: 激活大小 <= 256KB 且 MACs <= 15M
#   B: 激活大小 <= 64KB
# ------------------------------------------------------------------------
# 回答的推導方式：
#   #MACs 與峰值激活大小在設計空間中都是單調的 —— 兩者都隨解析度、通道寬度、
#   深度、擴展比例與卷積核大小遞增。因此整個設計空間的下界，就是
#   「最小解析度 (96) + 最小 width_mult (0.5) + 最小 ks/e/d」這個子網路。
#   只要把這個下界算出來，就能直接判定某個約束是否可行：
#     - 若下界同時滿足約束 -> 可行 (而且可以真的搜出一個子網路來驗證)
#     - 若下界已經違反約束 -> 整個設計空間都不可能滿足
#   下面的 question_10 會把各解析度下的最小子網路效率印出來，並在 A 的約束下
#   實際嘗試取樣，用實測數字給出結論 (峰值記憶體受第一段 stem/早期 block 的
#   輸入輸出張量支配，很難靠縮小架構壓下去，這正是 B 的關鍵)。
# ------------------------------------------------------------------------
def find_subnet_under_constraint(efficiency_predictor, arch_encoder, constraint,
                                 n_trials=2000):
    """有限次數的隨機取樣：找得到就回傳 (sample, efficiency)，否則回傳 (None, None)。"""
    for _ in tqdm(range(n_trials), desc=str(constraint), leave=False):
        sample = arch_encoder.random_sample_arch()
        efficiency = efficiency_predictor.get_efficiency(sample)
        if efficiency_predictor.satisfy_constraint(efficiency, constraint):
            return sample, efficiency
    return None, None


def question_10_feasibility(ofa_network, efficiency_predictor, arch_encoder):
    print("\n=== 問題 10：可行性分析 ===")

    # 1. 掃描各解析度下的「最小子網路」，得到設計空間的效率下界。
    print("--- 各解析度下最小子網路的效率 (即該解析度的下界) ---")
    lower_bound = None
    for res in image_size_list:
        cfg = ofa_network.sample_active_subnet(sample_function=min, image_size=res)
        eff = efficiency_predictor.get_efficiency(cfg)
        print(f"resolution={res:3d}: {eff['millionMACs']:7.2f}M MACs, "
              f"{eff['KBPeakMemory']:8.2f} KB peak activation")
        if lower_bound is None:
            lower_bound = eff  # 解析度最小 (96) 即為全域下界
    print(f"\n設計空間下界: MACs >= {lower_bound['millionMACs']:.2f}M, "
          f"peak activation >= {lower_bound['KBPeakMemory']:.2f}KB")

    # 2. 約束 A：MACs <= 15M 且激活 <= 256KB
    constraint_a = dict(millionMACs=15, KBPeakMemory=256)
    feasible_a = efficiency_predictor.satisfy_constraint(lower_bound, constraint_a)
    print(f"\n[A] MACs <= 15M 且 peak activation <= 256KB -> "
          f"下界檢查: {'可能可行' if feasible_a else '不可行 (下界已違反)'}")
    if feasible_a:
        sample, eff = find_subnet_under_constraint(
            efficiency_predictor, arch_encoder, constraint_a, n_trials=2000
        )
        if sample is not None:
            print(f"    隨機取樣找到滿足 A 的子網路: {eff}")
            print("    架構圖:", visualize_subnet(sample, "q10_constraint_a"))
        else:
            print("    2000 次隨機取樣未命中 (可行域極小，可改用進化搜尋逼近)。")

    # 3. 約束 B：激活 <= 64KB
    constraint_b = dict(KBPeakMemory=64)
    feasible_b = efficiency_predictor.satisfy_constraint(lower_bound, constraint_b)
    print(f"\n[B] peak activation <= 64KB -> "
          f"下界檢查: {'可能可行' if feasible_b else '不可行 (下界已違反)'}")
    if feasible_b:
        sample, eff = find_subnet_under_constraint(
            efficiency_predictor, arch_encoder, constraint_b, n_trials=2000
        )
        if sample is not None:
            print(f"    隨機取樣找到滿足 B 的子網路: {eff}")
            print("    架構圖:", visualize_subnet(sample, "q10_constraint_b"))
        else:
            print("    2000 次隨機取樣未命中。")
    else:
        print("    原因：峰值激活由 stem / 早期 block 的輸入+輸出張量決定，"
              "即使把 ks/e/d/width 全部取最小、解析度取 96，也降不到 64KB。")

    return dict(lower_bound=lower_bound, feasible_a=feasible_a, feasible_b=feasible_b)


# %% ---------------------------------------------------------------------
# main
# ------------------------------------------------------------------------
PARTS = ["dataset", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]


def main():
    parser = argparse.ArgumentParser(description="MIT 6.5940 Lab 3: NAS")
    parser.add_argument(
        "--part", nargs="+", default=PARTS, choices=PARTS,
        help="要執行的段落 (預設全部)",
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="準確率預測器的訓練 epoch 數")
    args = parser.parse_args()
    parts = set(args.part)

    prepare_environment()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # ---- 超網路 ----
    print("Building OFA supernet ...")
    ofa_network = build_ofa_network()

    if "dataset" in parts:
        visualize_dataset()

    if "q1" in parts:
        question_1_design_space_exploration(ofa_network)

    # ---- 問題 2：效率預測器 ----
    efficiency_predictor = AnalyticalEfficiencyPredictor(ofa_network)
    if "q2" in parts:
        question_2_efficiency_predictor(ofa_network, efficiency_predictor)

    # ---- 問題 3/4：準確率預測器 ----
    arch_encoder = build_arch_encoder(ofa_network)
    acc_pred_checkpoint_path = (
        f"pretrained/{ofa_network.__class__.__name__}_acc_predictor.pth"
    )
    acc_predictor = AccuracyPredictor(
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=acc_pred_checkpoint_path,
        device=device,
    )
    if "q3" in parts:
        print("\n=== 問題 3：準確率預測器 ===")
        print(acc_predictor)

    needs_predictor = parts & {"q4", "q5", "q6", "q7", "q8", "q9"}
    if needs_predictor:
        acc_dataset = AccuracyDataset("acc_datasets")
        train_loader, valid_loader, base_acc = acc_dataset.build_acc_data_loader(
            arch_encoder=arch_encoder
        )
        print(f"The basic accuracy (mean accuracy of all subnets within the dataset) "
              f"is: {(base_acc * 100): .1f}%.")
        if "q3" in parts:
            visualize_acc_dataset_sample(arch_encoder, train_loader)

        if "q4" in parts or not os.path.exists(acc_pred_checkpoint_path):
            acc_predictor = question_4_train_acc_predictor(
                acc_predictor, train_loader, valid_loader, base_acc,
                n_epochs=args.epochs, checkpoint_path=acc_pred_checkpoint_path,
            )
            plot_acc_predictor_correlation(acc_predictor, valid_loader)

    # ---- 問題 5/6：隨機搜尋 ----
    if parts & {"q5", "q6"}:
        question_5_6_random_search(ofa_network, efficiency_predictor, acc_predictor)

    # ---- 問題 7/8：進化搜尋 ----
    if parts & {"q7", "q8"}:
        question_7_8_evolution_search(ofa_network, efficiency_predictor, acc_predictor)

    # ---- 問題 9：真實世界約束 ----
    if "q9" in parts:
        question_9_real_world_constraints(
            ofa_network, efficiency_predictor, acc_predictor
        )

    # ---- 問題 10：可行性分析 ----
    if "q10" in parts:
        question_10_feasibility(ofa_network, efficiency_predictor, arch_encoder)


if __name__ == "__main__":
    main()
