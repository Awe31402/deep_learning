"""
MIT 6.5940 EfficientML.ai Lab 4: LLM Quantization with AWQ (大型語言模型量化)

從 Lab4_zh.md 整理出的完整實作程式碼：
  - Setup   : wikitext-2 困惑度評估 / 模型大小 / 偽量化 (baseline)
  - 問題 1  : 保護顯著權重通道 (1.1 topk / 1.2 random / 1.3 問答)
  - 問題 2  : AWQ scaling (2.1 scale up / 2.2 掃 scale factor + 問答 / 2.3 grid search)
  - 加分題  : 在 2.3 之上再做權重截斷搜尋 (auto clip)，全程不使用混合精度

執行方式：
    python3 hw4.py                       # 依序執行全部段落
    python3 hw4.py --part q1.1 q2.3      # 只跑指定段落 (見 PARTS)
    python3 hw4.py --model facebook/opt-125m --nsamples 4 --part q2.3   # 快速煙霧測試

lab 給的目標數值 (opt-1.3b, w_bit=3, group_size=128)：
    問題 1.1 = 17.15 / 問題 1.2 > 100 / 問題 2.1 = 18.93 / 問題 2.3 = 17.92

本機實測 (RTX 4060 8GB，模型以 fp16 載入，nsamples=40)：
    FP16 baseline          14.47      3-bit RTN              123.88
    問題 1.1 (salient 1%)   17.16      問題 1.2 (random 1%)    120.65
    問題 2.1 (s=2)          18.95      問題 2.2 s=1/2/3/4      123.88 / 18.95 / 19.23 / 21.24
    問題 2.3 (auto scale)   17.89      加分題 (scale+clip)      17.46
實測數字會因為載入精度 (fp32/fp16)、套件版本與隨機種子略有差異。
"""
import argparse
import gc
from functools import partial

import torch
import tqdm
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

PARTS = ["fp32", "rtn", "q1.1", "q1.2", "q1.3", "q2.1", "q2.2", "q2.3", "bonus"]


# %% ---------------------------------------------------------------------
# 環境準備：模型載入 / 困惑度評估 / 模型大小
# ------------------------------------------------------------------------
def load_model(model_path, dtype):
    """重新載入一份乾淨的模型 (每個實驗都要從未量化的權重開始)。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="auto"
    )


def resolve_dtype(dtype_arg):
    """auto: 顯存 >= 16GB 用 fp32 (對齊 lab 的參考數值)，否則用 fp16 以免 OOM。"""
    if dtype_arg != "auto":
        return getattr(torch, dtype_arg)
    if not torch.cuda.is_available():
        return torch.float32
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gib >= 16:
        return torch.float32
    print(f"[info] GPU 顯存只有 {total_gib:.1f} GiB，改用 float16 載入模型 "
          f"(困惑度與 fp32 相差通常在 0.05 以內)")
    return torch.float16


@torch.no_grad()
def evaluate(model, tokenizer, nsamples=40):
    """在 wikitext-2 test set 上計算困惑度 (perplexity)。"""
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")

    testenc = testenc.input_ids.to(model.device)
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        nlls.append(loss.float() * 2048)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))


def get_model_size(model: nn.Module, data_width=16, group_size=-1):
    """模型大小 (bit)。group_size != -1 時要額外算上每組的 scale (fp16) 與 zero point (4-bit)。"""
    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


def report(tag, model, tokenizer, args, data_width):
    """統一的評估 + 印出格式，回傳 (perplexity, model_size_MiB)。"""
    ppl = evaluate(model, tokenizer, nsamples=args.nsamples).item()
    size = get_model_size(model, data_width=data_width, group_size=args.group_size)
    print(f"\n[{tag}] model perplexity: {ppl:.2f}")
    print(f"[{tag}] model size: {size / MiB:.2f} MiB")
    return ppl, size / MiB


# %% ---------------------------------------------------------------------
# 偽量化 (pseudo quantization)：量化後立刻反量化回浮點數，用來模擬量化誤差
# ------------------------------------------------------------------------
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # 每一組的最大值 (\alpha) 與最小值 (\beta)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)

    # 公式 (1)(2)：scale factor 與 zero point
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # 公式 (3)：把 [\beta, \alpha] 映射到 [0, 2^b - 1]
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    # 反量化 (偽量化的關鍵：權重仍以浮點數儲存，只是數值被離散化過)
    w = (w - zeros) * scales

    assert torch.isnan(w).sum() == 0
    return w.reshape(org_w_shape)


@torch.no_grad()
def pseudo_quantize_model_weight(model, w_bit, q_group_size):
    """baseline：對所有 nn.Linear 做 round-to-nearest 量化。"""
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size)


# %% ---------------------------------------------------------------------
# 校準數據集：收集每個 Linear 的輸入激活值大小 (用來判斷哪些權重通道顯著)
# ------------------------------------------------------------------------
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"].strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i * block_size:(i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def get_calib_feat(model, tokenizer):
    """對每個 Linear 掛上 forward hook，記錄輸入在各通道上的平均絕對值。"""
    input_dict = dict()

    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach().float()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")
    samples = get_calib_dataset(tokenizer)
    for input_ids in tqdm.tqdm(samples):
        model(input_ids.to(model.device))

    for hook in hooks:
        hook.remove()
    return input_dict


def salient_channels(importance, ratio=0.01):
    """回傳重要性最高的前 ratio 比例通道的索引 (問題 1.1 / 2.1 共用)。"""
    k = max(1, int(importance.numel() * ratio))
    return torch.topk(importance, k=k, largest=True).indices


# %% ---------------------------------------------------------------------
# 問題 1.1 (20 分)：把 1% 最顯著的權重通道保留成 FP16
#   顯著度 = 該通道輸入激活值的平均絕對值總和 (跨所有校準批次)。
#   做法：量化前先備份這些通道，量化後再寫回原始數值。
# ------------------------------------------------------------------------
@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############
            # Step 1: 依 importance 找出 1% 的顯著權重通道
            outlier_indices = salient_channels(importance, ratio=0.01)
            assert outlier_indices.dim() == 1
            ############### YOUR CODE ENDS HERE #################

            # 備份顯著權重通道的原始數值
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############
            # Step 2: 把這 1% 的通道還原成原本的 FP16 數值
            m.weight.data[:, outlier_indices] = outlier
            ############### YOUR CODE ENDS HERE #################


# %% ---------------------------------------------------------------------
# 問題 1.2 (15 分)：消融實驗 —— 改成「隨機」保護 1% 的通道
#   若顯著通道的假設成立，隨機保護應該幾乎沒有幫助 (困惑度仍會 > 100)。
# ------------------------------------------------------------------------
@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(model, w_bit, q_group_size, input_feat,
                                             seed=0):
    generator = torch.Generator().manual_seed(seed)
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############
            # Step 1: 隨機挑選 1% 的權重通道
            k = max(1, int(importance.numel() * 0.01))
            outlier_mask = torch.randperm(importance.numel(), generator=generator)[:k]
            assert outlier_mask.dim() == 1
            ############### YOUR CODE ENDS HERE #################

            outlier = m.weight.data[:, outlier_mask].clone()

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############
            # Step 2: 把隨機挑到的 1% 通道還原成原本的 FP16 數值
            m.weight.data[:, outlier_mask] = outlier
            ############### YOUR CODE ENDS HERE #################


# %% ---------------------------------------------------------------------
# 問題 1.3 (15 分)：為什麼顯著權重通道這麼重要？
# ------------------------------------------------------------------------
ANSWER_1_3 = """\
解答 1.3：為什麼顯著 (salient) 權重通道如此重要？

1) 誤差是被激活值放大的，不是只看權重本身。
   線性層的輸出為 y = sum_j w_j * x_j，量化只動到 w，因此第 j 個通道對輸出誤差的
   貢獻是 |Δw_j| * |x_j|。RTN 量化的 |Δw_j| 對每個通道而言是差不多的 (同一組共用
   同一個 scale)，所以誤差幾乎完全由 |x_j| 決定。輸入激活值大的通道 = 誤差被放大
   最多的通道，把它們保留成 FP16 就等於把總誤差中最大的那幾項直接消掉。

2) LLM 的激活值離群值是「系統性」的，集中在極少數固定通道。
   同一個離群通道在所有 token 上都很大 (跨 token 變異小)，但通道之間差異可以到
   100 倍以上。也就是說重要性的分佈極度長尾：只要 0.1%~1% 的通道就佔掉輸出量值的
   絕大部分。這也是為什麼保護 1% 就足以把困惑度從 22.8 拉回 17.2，而隨機挑 1%
   幾乎沒有效果 (問題 1.2)——隨機挑到的多半是不重要的小激活通道。

3) 這些通道也支配了下游的計算路徑。
   離群通道往往對應到 attention/FFN 中真正攜帶語意訊息的方向，其誤差會沿著殘差
   連結逐層累積放大；而且下一層的 LayerNorm 之後這些偏差不會被抵銷，最終在
   logits 上造成可觀的偏移，直接反映成困惑度上升。

4) 從量化的角度看，離群通道也是最難量化的部分。
   權重分佈裡與離群激活相關的通道通常數值範圍也較大，在 group 內會撐大 (α - β)，
   讓整組的 scale 變粗、所有通道一起變差。把它們挑出來單獨處理 (保留 FP16 或
   AWQ 的 scale up)，等於同時解決「誤差放大」與「動態範圍被撐大」兩個問題。
"""


# %% ---------------------------------------------------------------------
# 問題 2.1 (20 分)：不留 FP16，改成量化前放大顯著通道、量化後再縮放回來
#   等價於 W' = Q(W · s) · s^-1；由於 RoundErr 幾乎固定 (~0.25)，而 Δ 在 group
#   夠大時不太受單一通道影響，放大 s 倍後該通道的相對量化誤差就縮成 1/s。
# ------------------------------------------------------------------------
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(model, w_bit, q_group_size, input_feat,
                                         scale_factor):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############
            # Step 1: 找出 1% 的顯著權重通道
            outlier_mask = salient_channels(importance, ratio=0.01)
            assert outlier_mask.dim() == 1
            ############### YOUR CODE ENDS HERE #################

            # 模擬套用 scale factor：量化前先乘上去，量化後再除回來
            m.weight.data[:, outlier_mask] *= scale_factor

            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############
            # Step 2: 把顯著權重通道縮放回原本的比例
            m.weight.data[:, outlier_mask] /= scale_factor
            ############### YOUR CODE ENDS HERE #################


# %% ---------------------------------------------------------------------
# 問題 2.2 (15 分)：掃不同的 scale factor，觀察困惑度先降後升
# ------------------------------------------------------------------------
ANSWER_2_2 = """\
解答 2.2：為什麼困惑度會先下降、之後又上升？

兩股力量在拉鋸，s 變大時一好一壞：

(A) 好的一面 —— 顯著通道的相對誤差變成 1/s。
    Err(Q(w·s)(x/s)) = Δ · RoundErr(w/Δ) · x · (1/s)。只要 Δ (= max|w| / 2^(N-1))
    沒有跟著變大，把顯著通道放大 s 倍再除回來，就等於把它的量化誤差除以 s。
    這是 s 從 1 增加到 2 時困惑度下降的原因。

(B) 壞的一面 —— s 太大會撐大整個 group 的動態範圍，Δ 跟著變大。
    量化是 group-wise 的 (group_size=128)，同一組共用一個 scale。當 s 大到讓
    被放大的通道成為該組的新最大值時，(α - β) 變大 -> Δ 變大 -> 這一組「其他
    127 個通道」的量化誤差全部按比例惡化。此時顯著通道賺到的 1/s 已經邊際遞減
    (誤差本來就很小了)，但其餘通道的損失卻線性增加，總誤差開始上升。

因此存在一個最佳點。本機實測 (opt-1.3b, 3-bit, group_size=128)：
    s = 1 : 123.88   <- 等於沒有保護的 RTN baseline
    s = 2 :  18.95   <- 最佳值，(A) 的收益遠大於 (B) 的損失
    s = 3 :  19.23   <- (B) 開始反噬
    s = 4 :  21.24   <- 整組動態範圍被撐大，其餘通道全面惡化

另外兩個次要因素也讓大 s 更不利：
  - 放大後的權重更容易被 clamp 到量化範圍的邊界，產生額外的截斷誤差；
  - 這裡是「所有層都用同一個固定 s」，不同層的激活分佈差異很大，單一常數
    無法同時逼近每一層的最佳值 —— 這正是問題 2.3 要用 per-channel 搜尋
    s = s_X^α 來解決的問題。
"""


# %% ---------------------------------------------------------------------
# 問題 2.3 (15 分)：在搜尋空間 s = s_X^α 中找最佳縮放係數 (AWQ auto scale)
#   目標：L(s) = || Q(W·s)(s^-1·X) - WX ||，用 grid search 掃 α ∈ [0, 1)。
#   由於縮放可以被前一個 LayerNorm / Linear 吸收 (scale_ln_fcs / scale_fc_fc)，
#   推論時不需要任何額外運算，也完全不需要混合精度。
# ------------------------------------------------------------------------
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """把 scales 融進前一層 LayerNorm：LN 除以 s，後面的 fc 權重乘上 s。"""
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """把 scales 融進前一個 Linear 的輸出通道。"""
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):

    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        param = next(block.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

        # s_X：輸入激活值在各通道上的平均絕對值 (以 fp32 計算，避免 fp16 取冪失準)
        s_x = x.view(-1, x.shape[-1]).abs().mean(0).float()

        ############### YOUR CODE STARTS HERE ###############
        # Step 1: 初始化 best_error / best_ratio / best_scales
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        ############### YOUR CODE ENDS HERE #################

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio 就是公式中的 \alpha
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############
            # Step 2: 依公式計算 scales = s_x^ratio
            scales = s_x.pow(ratio).clamp(min=1e-4).view(-1)
            assert scales.shape == s_x.shape
            ############### YOUR CODE ENDS HERE #################

            # 正規化：讓 scales 的幾何平均為 1，避免整體數值被放大或縮小
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:
                scales = scales.to(device=fc.weight.device, dtype=fc.weight.dtype)

                # 放大權重通道
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(
                    fc.weight.data, w_bit, q_group_size)

                ############### YOUR CODE STARTS HERE ###############
                # Step 3: 把權重通道縮放回原本的比例
                fc.weight.data = fc.weight.data / scales
                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float 避免溢位
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)
        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input (q/k/v 共用同一個輸入，縮放融進 self_attn_layer_norm)
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # attn out (縮放融進 v_proj)
    inp = input_feat[name + ".self_attn.out_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(
        module.self_attn.out_proj, [module.self_attn.out_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # fc1 (縮放融進 final_layer_norm)
    inp = input_feat[name + ".fc1"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # fc2 (縮放融進 fc1)
    inp = input_feat[name + ".fc2"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)


@torch.no_grad()
def apply_auto_scale(model, w_bit, q_group_size, input_feat):
    """只做 scaling (等價變換)，還不量化 —— 方便加分題在中間插入 clipping。"""
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, OPTDecoderLayer)]
    for name, module in tqdm.tqdm(layers, desc="auto scale"):
        auto_scale_block(module, name, w_bit, q_group_size, input_feat)


@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    apply_auto_scale(model, w_bit, q_group_size, input_feat)
    pseudo_quantize_model_weight(model, w_bit, q_group_size)


# %% ---------------------------------------------------------------------
# 加分題：AWQ auto clip —— 在 auto scale 之後，再搜尋每一組權重的最佳截斷範圍
#   RTN 用 group 內的 min/max 當量化範圍，只要有一個離群權重就會撐大 Δ，讓其餘
#   127 個權重的解析度變差。把範圍縮成 c * max|w| (c < 1)、犧牲少數被 clamp 的
#   權重，通常能讓整組的輸出誤差更小。
#   這個做法完全是整數量化 (沒有任何 FP16 通道)，只是量化前多一次截斷，
#   截斷後的 min/max 直接決定 scale/zero point，推論時零額外成本。
# ------------------------------------------------------------------------
@torch.no_grad()
def auto_clip_layer(w, inp, n_bit, q_group_size, n_grid=20, max_shrink=0.5,
                    n_sample_token=None, oc_batch_size=256):
    """搜尋每個 (output channel, group) 的最佳截斷值，回傳 best_max_val。"""
    assert w.dim() == 2
    group_size = q_group_size if q_group_size > 0 else w.shape[1]
    assert w.shape[1] % group_size == 0

    inp = inp.view(-1, inp.shape[-1])                      # [n_token, ci]
    if n_sample_token is not None and inp.shape[0] > n_sample_token:
        step = max(1, inp.shape[0] // n_sample_token)
        inp = inp[::step]
    # 搜尋過程一律用 fp32，避免 fp16 累加誤差影響到最佳截斷值的選擇
    inp = inp.to(w.device, dtype=torch.float32)
    inp = inp.reshape(1, inp.shape[0], -1, group_size)     # [1, n, n_group, group]

    best = []
    # 依 output channel 分批，避免中間張量 [co, n, n_group] 過大
    for i_b in range(0, w.shape[0], oc_batch_size):
        w_b = w[i_b:i_b + oc_batch_size].float()
        w_b = w_b.reshape(w_b.shape[0], 1, -1, group_size)  # [co_b, 1, n_group, group]

        org_max_val = w_b.abs().amax(dim=-1, keepdim=True)  # [co_b, 1, n_group, 1]
        best_max_val = org_max_val.clone()
        min_errs = torch.full_like(org_max_val, float("inf"))
        org_out = (inp * w_b).sum(dim=-1)                   # [co_b, n, n_group]

        for i in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i / n_grid)
            cur_w = torch.clamp(w_b, -max_val, max_val)
            q_w = pseudo_quantize_tensor(
                cur_w.reshape(-1, group_size), n_bit, group_size).reshape(cur_w.shape)
            cur_out = (inp * q_w).sum(dim=-1)
            err = (cur_out - org_out).float().pow(2).mean(dim=1).view(min_errs.shape)

            best_max_val = torch.where(err < min_errs, max_val, best_max_val)
            min_errs = torch.minimum(err, min_errs)

        best.append(best_max_val)
        del org_out, w_b

    return torch.cat(best, dim=0)


# q/k/v 的量化誤差會經過 softmax 放大，實務上截斷 query/key 反而更糟；
# lm_head 與 embedding 共用權重，也一併跳過。
CLIP_SKIP = ("lm_head", "q_proj", "k_proj")


@torch.no_grad()
def apply_auto_clip(model, w_bit, q_group_size, input_feat, skip_names=CLIP_SKIP):
    """對每個 Linear 搜尋截斷範圍並就地套用 (之後再統一量化)。"""
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n in input_feat
               and not any(s in n for s in skip_names)]

    for n, m in tqdm.tqdm(linears, desc="auto clip"):
        # 這裡沿用 auto scale 的做法：把每個校準批次的輸入統計量當作代理輸入
        inp = torch.stack(input_feat[n], dim=0)             # [n_batch, ci]
        max_val = auto_clip_layer(m.weight.data, inp, w_bit, q_group_size)
        org_shape, org_dtype = m.weight.shape, m.weight.dtype
        w = m.weight.data.float().reshape(org_shape[0], 1, max_val.shape[2], -1)
        m.weight.data = torch.clamp(w, -max_val, max_val).reshape(
            org_shape).to(org_dtype)


@torch.no_grad()
def pseudo_quantize_model_weight_awq(model, w_bit, q_group_size, input_feat,
                                     tokenizer):
    """加分題完整流程：auto scale -> 重新校準 -> auto clip -> 量化。"""
    apply_auto_scale(model, w_bit, q_group_size, input_feat)
    # scaling 改變了每一層實際看到的輸入 (x -> x/s)，截斷搜尋要用新的統計量
    print("Re-collecting activation scales after auto scale...")
    scaled_feat = get_calib_feat(model, tokenizer)
    apply_auto_clip(model, w_bit, q_group_size, scaled_feat)
    pseudo_quantize_model_weight(model, w_bit, q_group_size)


# %% ---------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EfficientML Lab 4: AWQ")
    parser.add_argument("--part", nargs="+", default=PARTS, choices=PARTS,
                        help="要執行的段落 (預設全部)")
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--w-bit", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=40,
                        help="wikitext-2 上用來算困惑度的區塊數")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--scale-factors", nargs="+", type=float,
                        default=[1, 2, 3, 4], help="問題 2.2 要掃的 scale factor")
    args = parser.parse_args()

    parts = set(args.part)
    dtype = resolve_dtype(args.dtype)
    w_bit, gs = args.w_bit, args.group_size
    print(f"model: {args.model} | w_bit: {w_bit} | group_size: {gs} | "
          f"dtype: {dtype} | device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    results = {}   # tag -> (perplexity, size_MiB)
    model = None

    def fresh_model():
        nonlocal model
        del model
        model = load_model(args.model, dtype)
        return model

    # ---- FP32 baseline ----
    if "fp32" in parts:
        print("\n=== Baseline：未量化模型 ===")
        model = load_model(args.model, dtype)
        # 模型大小一律以 FP32 計 (對齊 lab 的算法)，實際載入精度見 dtype
        results[f"未量化 baseline ({str(dtype).split('.')[-1]})"] = report(
            "fp32", model, tokenizer, args, data_width=32)

    # ---- RTN baseline (未保護任何通道) ----
    if "rtn" in parts:
        print(f"\n=== Baseline：{w_bit}-bit RTN 量化 ===")
        fresh_model()
        pseudo_quantize_model_weight(model, w_bit=w_bit, q_group_size=gs)
        results[f"{w_bit}-bit RTN"] = report("rtn", model, tokenizer, args,
                                             data_width=w_bit)

    # ---- 校準：問題 1.1 之後的段落都需要 input_feat ----
    input_feat = None
    if parts & {"q1.1", "q1.2", "q2.1", "q2.2", "q2.3", "bonus"}:
        fresh_model()
        input_feat = get_calib_feat(model, tokenizer)

    # ---- 問題 1.1：保護 1% 顯著通道 ----
    if "q1.1" in parts:
        print("\n=== 問題 1.1：1% 顯著權重通道保留 FP16 ===")
        fresh_model()
        pseudo_quantize_model_salient_weight_fp16(model, w_bit, gs, input_feat)
        results["Q1.1 salient FP16 (1%)"] = report("q1.1", model, tokenizer, args,
                                                   data_width=w_bit)

    # ---- 問題 1.2：隨機保護 1% 通道 ----
    if "q1.2" in parts:
        print("\n=== 問題 1.2：隨機保護 1% 權重通道 (消融) ===")
        fresh_model()
        pseudo_quantize_model_random_weight_fp16(model, w_bit, gs, input_feat)
        results["Q1.2 random FP16 (1%)"] = report("q1.2", model, tokenizer, args,
                                                  data_width=w_bit)

    # ---- 問題 1.3：問答 ----
    if "q1.3" in parts:
        print("\n=== 問題 1.3 ===")
        print(ANSWER_1_3)

    # ---- 問題 2.1：放大顯著通道 (scale factor = 2) ----
    if "q2.1" in parts:
        print("\n=== 問題 2.1：放大顯著通道 (scale factor = 2) ===")
        fresh_model()
        pseudo_quantize_model_weight_scaleup(model, w_bit, gs, input_feat,
                                             scale_factor=2)
        results["Q2.1 scale up (s=2)"] = report("q2.1", model, tokenizer, args,
                                                data_width=w_bit)

    # ---- 問題 2.2：掃不同的 scale factor ----
    if "q2.2" in parts:
        print("\n=== 問題 2.2：不同 scale factor 的困惑度 ===")
        sweep = {}
        for s in args.scale_factors:
            fresh_model()
            pseudo_quantize_model_weight_scaleup(model, w_bit, gs, input_feat,
                                                 scale_factor=s)
            ppl, size = report(f"q2.2 s={s:g}", model, tokenizer, args,
                               data_width=w_bit)
            sweep[s] = ppl
            results[f"Q2.2 scale up (s={s:g})"] = (ppl, size)

        print("\nscale factor -> perplexity")
        for s, ppl in sweep.items():
            print(f"  s = {s:<4g} : {ppl:.2f}")
        best_s = min(sweep, key=sweep.get)
        print(f"最佳 scale factor = {best_s:g} (perplexity {sweep[best_s]:.2f})")
        print(ANSWER_2_2)

    # ---- 問題 2.3：AWQ grid search ----
    if "q2.3" in parts:
        print("\n=== 問題 2.3：AWQ auto scale (grid search s = s_X^α) ===")
        fresh_model()
        pseudo_quantize_model_weight_auto_scale(model, w_bit, gs, input_feat)
        results["Q2.3 AWQ auto scale"] = report("q2.3", model, tokenizer, args,
                                                data_width=w_bit)

    # ---- 加分題：auto scale + auto clip ----
    if "bonus" in parts:
        print("\n=== 加分題：AWQ auto scale + auto clip (無混合精度) ===")
        fresh_model()
        pseudo_quantize_model_weight_awq(model, w_bit, gs, input_feat, tokenizer)
        results["Bonus AWQ scale + clip"] = report("bonus", model, tokenizer, args,
                                                   data_width=w_bit)

    # ---- 總結 ----
    if results:
        print("\n" + "=" * 62)
        print(f"{'setting':<32}{'perplexity':>12}{'size (MiB)':>14}")
        print("-" * 62)
        for tag, (ppl, size) in results.items():
            print(f"{tag:<32}{ppl:>12.2f}{size:>14.2f}")
        print("=" * 62)
        if "Q2.3 AWQ auto scale" in results and "Bonus AWQ scale + clip" in results:
            gain = (results["Q2.3 AWQ auto scale"][0]
                    - results["Bonus AWQ scale + clip"][0])
            print(f"加分題相對問題 2.3 的困惑度改善：{gain:+.2f}")


if __name__ == "__main__":
    main()
