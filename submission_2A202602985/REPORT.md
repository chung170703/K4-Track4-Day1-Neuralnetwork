# Báo cáo Lab Day 1 — Ngô Đức Chung — 2A202602985

Mọi con số trỏ về `exp_id` trong `experiments.xlsx` (hoặc `results/<exp_id>.json`) và ảnh trong `figures/`. Kết luận chỉ dựa trên **val**; eval chỉ dùng cho baseline và cấu hình cuối cùng (mục 4). "Vượt nhiễu" nghĩa là chênh lệch val macro-F1 lớn hơn 2σ của 5 seed baseline.

## 1. Thiết lập

- **Môi trường:** MacBook Air (Apple Silicon), GPU `mps`, PyTorch 2.8.0, Python 3.9 (README yêu cầu 3.10+; code dùng `from __future__ import annotations` nên chạy được trên 3.9). Cả notebook (`code/lab.ipynb`) chạy từ đầu đến cuối trong ~23 phút, không lỗi. Notebook có nhánh `IN_COLAB` nhưng bản cuối cùng chỉ được chạy trên Mac, chưa chạy lại trên Colab.
- **Dữ liệu:** Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv` (không sửa). Val = 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. Chuẩn hoá 10 cột số bằng mean/std của 371 847 mẫu train (val và eval chỉ được áp dụng, không tham gia tính thống kê).
- **Model:** `M-base` 54→256→128→7, ReLU, 47 879 tham số (có `assert`), logits `(B, 7)`, softmax nằm trong loss.
- **Baseline:** CE, SGD+momentum 0.9, **lr 0.3** (chọn bằng val trong lưới 0.01–1.0), He init, batch 512, 20 epoch, không dropout/clip, FP32.
- **Mốc tham chiếu:** đoán luôn lớp đa số: accuracy val = 0.4876.
- **Chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init (đủ 7/7).
- **Quy ước đo:** `train_loss` đo ở `eval()` trên 50 000 mẫu train cố định; `grad_norm` đo trước khi clip; epoch tốt nhất chọn theo val loss; batch cuối nhỏ hơn được giữ lại (mọi mẫu đều được dùng).

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (ln 7 = 1.946) | 2.378 (seed 42, Part 1); 5 seed baseline: 2.12 ± 0.18 (1.90–2.31) |
| Quá khớp 20 mẫu: loss cuối | 6·10⁻⁶ (accuracy 100%, `figures/overfit20.png`) |
| Mọi tham số có gradient khác 0 | có (grad norm 0.38–2.37) |
| Baseline, số seed đã chạy | 5 (`base-s1..5`) |
| Baseline: val acc (TB ± σ) | 0.9126 ± 0.0026 |
| Baseline: val macro-F1 (TB ± σ) | 0.8577 ± 0.0029 |

**Ngưỡng nhiễu dùng trong báo cáo: 2σ = 0.0058** (val macro-F1).

Loss bước 0 cao hơn ln 7 một cách có hệ thống vì He init cho logit đầu ra độ lệch chuẩn ≈ 0.6, nên softmax không đều (ln 7 + σ²/2 ≈ 2.13, khớp mức trung bình đo được). Phép thử quá khớp đạt nên không phải lỗi code.

**Hình dạng đường cong baseline** (`figures/base-s1.png`): train/val loss giảm đều 0.44 → 0.20 / 0.22; khoảng cách val − train chỉ 0.023; best epoch 18–20 ở cả 5 seed, tức mô hình chưa quá khớp và vẫn đang học khi dừng. Val macro-F1 dao động theo epoch (0.830 ở epoch 10, 0.827 ở 15, 0.855 ở 20) vì các lớp hiếm nhạy với từng cập nhật.

## 3. Kết quả theo chủ đề

Chênh lệch (Δ) tính so với trung bình baseline 0.8577. Mỗi thí nghiệm chạy **một seed** (seed 1), nên Δ nhỏ hơn 2σ chỉ là chưa kết luận được.

### 3.1 Hàm mất mát — CE vs MSE
- **Dự đoán:** MSE chậm hơn và kém CE ở cùng lr (gradient nhỏ hơn); tăng lr bù thì tiến gần CE.
- **Kết quả** (`compare_loss.png`): `loss-mse-lr0.3` macro-F1 **0.782** (Δ −0.076, vượt 2σ), accuracy 0.881; `loss-mse-lr1` **0.628** (Δ −0.229), accuracy 0.855.
- **Cơ chế:** `grad_norm` epoch 1 của MSE 0.056 so với 0.385 của CE (nhỏ hơn ~7 lần): CE có gradient `p − y` theo logit, MSE có `2(z − y)/7` nên bước đi nhỏ và hội tụ chậm hơn trong 20 epoch. **Khác dự đoán:** tăng lr cho MSE lại tệ hơn; mình chưa kiểm chứng nguyên nhân. Không so loss CE với MSE trực tiếp (khác thang); chỉ so accuracy/macro-F1.

### 3.2 Bộ tối ưu hoá
- **Dự đoán:** Adam/AdamW hơn SGD+momentum ở lr tốt nhất; SGD thuần cần lr lớn hơn nhiều; AdamW(wd=0) trùng Adam.
- **Mỗi bộ ở lr tốt nhất** (mỗi bộ thử ≥ 3 lr cách nhau ~3 lần; `compare_optimizer.png`, `compare_optimizer_adam_lr.png`):

| Bộ tối ưu | exp_id | lr | val macro-F1 | best epoch |
|---|---|---|---|---|
| SGD thuần | `opt-sgd-lr1` | 1.0 | 0.8267 | 18 |
| SGD+momentum | `base-s1` | 0.3 | 0.8551 | 20 |
| AdamW (wd 0.01) | `opt-adamw-lr0.003` | 3e-3 | 0.8663 | 20 |
| **Adam** | `opt-adam-lr0.003` | 3e-3 | **0.8727** | 20 |

- Adam hơn SGD+momentum **+0.0150** (vượt 2σ = 0.0058). Cơ chế phù hợp: Adam chuẩn hoá bước theo độ lớn gradient từng tham số, hợp với đầu vào trộn cột liên tục và cột nhị phân thưa. SGD+momentum nhạy lr hơn: tăng đều tới 0.3 rồi **sập ở lr 1.0** (`hp-lr-sgdm-1`, về đoán lớp đa số); Adam giảm êm ở hai phía (0.785 / 0.847 / 0.873 / 0.864 với 3e-4 / 1e-3 / 3e-3 / 1e-2).
- `opt-adamw-wd0-lr3e-3` trùng `opt-adam-lr0.003` **tuyệt đối** (cùng val macro-F1, val loss lệch 0), đúng lý thuyết. AdamW wd 0.01 thấp hơn Adam 0.0064 ở lr 3e-3 (≈ 2σ) và cao hơn 0.0021 ở lr 1e-3 (trong nhiễu): chưa có bằng chứng wd 0.01 có ích.
- **Sai dự đoán:** SGD thuần tốt nhất ở lr 1.0, chỉ gấp ~3.3 lần lr của SGD+momentum (không phải ~10 lần). **Hạn chế:** mọi Adam đều có best epoch = 20, xếp hạng có thể đổi nếu huấn luyện lâu hơn; lưới lr thô, một seed mỗi lr.

### 3.3 Hyper-parameter
- **Dự đoán:** lr nhỏ underfit, lr lớn sập; batch nhỏ (nhiều bước hơn) tốt hơn; mạng rộng/sâu và nhiều epoch giúp; wd 5e-4 không giúp.
- **Kết quả** (`compare_hparam.png`, `compare_hparam_batch.png`):

| exp_id | Thay đổi | val macro-F1 | Δ | vượt 2σ? | giây/epoch |
|---|---|---|---|---|---|
| `hp-lr-sgdm-0.01 / 0.03 / 0.1` | lr | 0.766 / 0.818 / 0.851 | −0.092 / −0.040 / −0.007 | có | 0.9 |
| `hp-lr-sgdm-1` | lr 1.0 | 0.094 (sập) | −0.764 | có | 0.9 |
| `hp-batch128` | batch 128, cùng lr | 0.800 | −0.058 | có | 3.2 |
| `hp-batch128-lr0.075` | batch 128, lr÷4 | 0.8576 | −0.000 | không | 3.5 |
| `hp-batch2048` | batch 2048, cùng lr | 0.833 | −0.025 | có | 0.32 |
| `hp-batch2048-lr1.2` | batch 2048, lr×4 | 0.636 (sập) | −0.222 | có | 0.2 |
| `hp-wide` | M-wide 512-256 | 0.8779 | +0.020 | có | 1.0 |
| `hp-deep` | M-deep 256-128-64 | 0.8573 | −0.000 | không | 0.9 |
| `hp-wd5e-4` | weight decay 5e-4 | 0.709 | −0.149 | có | 0.9 |
| `hp-epochs40` | 40 epoch | **0.8809** | +0.023 | có | 0.8 |

- **Khác dự đoán:** batch 128 ở cùng lr 0.3 tệ hơn (gradient nhiễu hơn, lr quá lớn); giảm lr 4 lần chỉ ngang baseline nhưng chậm gấp 3.4 lần mỗi epoch. Quy tắc tăng lr theo lô (batch 2048, lr×4 = 1.2) làm sập vì vượt ngưỡng ổn định, không có khởi động. `M-deep` không giúp. Weight decay 5e-4 làm xấu rất nhiều (nghi do `lr·wd` kéo trọng số về 0 quá mạnh khi mạng chưa quá khớp; chưa kiểm chứng).
- **Cơ chế:** cùng 20 epoch nhưng batch 2048 chỉ có ~1/4 số bước nên chưa hội tụ; mạng rộng và 40 epoch giúp vì baseline đang chưa khớp. `M-wide` có best epoch 16 và khoảng cách train–val tăng (0.030): bắt đầu quá khớp nhẹ.

### 3.4 Dropout
- **Dự đoán:** không giúp, có thể xấu đi, vì mô hình chưa quá khớp.
- **Kết quả** (`compare_dropout.png`): q = 0 / 0.1 / 0.3 / 0.5 → macro-F1 **0.855 / 0.835 / 0.768 / 0.631** (mọi mức vượt 2σ). Khoảng cách val − train thu hẹp (0.023 / 0.014 / 0.0045 / 0.0031) nhưng chỉ vì **train loss** (đo ở eval mode) tăng nhiều hơn val: mô hình chưa khớp hơn chứ không khớp tốt hơn.
- **Trả lời:** dropout không có ích khi chưa quá khớp; chỉ nên dùng khi train thấp mà val tăng dần.

### 3.5 Gradient clipping
- `grad_norm` baseline có trung bình 0.34–0.39 theo epoch nhưng bước lớn nhất 2.95; chọn `c = 0.35` (trung vị) để clipping kích hoạt thật (80% bước ở epoch 1, 46% ở epoch 20).
- **Dự đoán:** không giúp ở lr baseline; cứu được ở lr cao.
- **Kết quả** (`compare_clipping.png`, `compare_clipping_gradnorm.png`):

| exp_id | lr | clip | val macro-F1 | `grad_norm` lớn nhất |
|---|---|---|---|---|
| `clip-c0.35-lr0.3` | 0.3 | 0.35 | **0.8653** (Δ +0.0076, vượt 2σ sát ngưỡng) | 2.95 |
| `hp-lr-sgdm-1` | 1.0 | không | 0.094 (sập) | 12.3 |
| `clip-c0.35-lr1` | 1.0 | 0.35 | **0.8166** (cứu được) | 2.95 |
| `clip-none-lr3` | 3.0 | không | 0.094 (sập) | 6 633 |
| `clip-c0.35-lr3` | 3.0 | 0.35 | 0.132 (**không** cứu được) | 4.30 |

- **Khác dự đoán:** clipping cải thiện nhẹ ở lr baseline (giả thuyết, chưa kiểm chứng: lr 0.3 sát ngưỡng sập và clipping cắt các gai gấp ~8 lần trung vị). Ở lr 1.0 clipping cứu được huấn luyện (bước ≤ lr·c). Ở lr 3.0 không cứu được (bước hiệu dụng ≈ 3·0.35/(1−0.9) ≈ 10 vẫn quá lớn). **Kết luận:** clipping chặn gai gradient bất ngờ nhưng không thay được việc chọn lr hợp lý.

### 3.6 Mixed precision
- **Dự đoán:** không nhanh hơn trên mạng nhỏ này (chi phí gọi kernel và ép kiểu), độ chính xác tương đương.
- **Kết quả** (`compare_amp.png`): FP32 **0.93 ± 0.03** s/epoch (5 seed); BF16 `amp-bf16` 1.08 s (+16%), macro-F1 0.8588; FP16 `amp-fp16` 1.34 s (+44%), macro-F1 0.8576. Chênh macro-F1 so với baseline (0.8551 / 0.8577) nằm trong 2σ.
- **Cơ chế:** FP16 có khoảng giá trị hẹp (số chuẩn nhỏ nhất ≈ 6·10⁻⁵) nên cần `GradScaler` nhân loss với hệ số s để gradient nhỏ không bị flush về 0; BF16 có số mũ 8 bit như FP32 nên không cần.
- **Hạn chế:** chỉ chạy trên MPS với mạng nhỏ; không kết luận về GPU NVIDIA hay mạng rất rộng. Cột `peak_mem_MB` trên MPS là bộ nhớ đang cấp phát cuối epoch (không có API đỉnh), bị chi phối bởi dữ liệu FP32 nên **không dùng để kết luận về bộ nhớ**.

### 3.7 Khởi tạo tham số
- **Dự đoán:** `zeros` không học được; `normal(0.01)` có kích hoạt co rất nhanh nên học chậm; Xavier/default học được; He không hơn nhiều ở mạng 3 lớp.
- **Kết quả** (`compare_init.png`; std kích hoạt sau Linear 1/2/3 ở bước 0):

| exp_id | std kích hoạt | loss bước 0 | val acc | val macro-F1 |
|---|---|---|---|---|
| `base-s1` (He) | 0.671 / 0.649 / 0.588 | 2.269 | 0.9113 | 0.8551 |
| `init-zeros` | 0 / 0 / 0 | 1.946 | **0.4876** | **0.0936** |
| `init-normal` | 0.035 / 0.004 / ≈ 0 | 1.946 | 0.9142 | 0.8649 |
| `init-xavier` | 0.280 / 0.221 / 0.195 | 2.022 | 0.9130 | 0.8610 |
| `init-default` | 0.278 / 0.115 / 0.060 | 1.983 | 0.9143 | 0.8674 |

- **`zeros` hỏng đúng cơ chế:** accuracy và macro-F1 bằng đúng mốc đoán lớp đa số, best val loss **1.2053 ≈ entropy phân bố lớp (1.2052)**. Khi mọi W = 0 thì h₁ = ReLU(0) = 0 nên gradient của W₂, W₃ (∝ hᵀ) bằng 0 và gradient của W₁ đi qua W₂ᵀ = 0; chỉ bias lớp ra học được. Ngay cả khi có gradient, mọi nơ-ron cùng lớp giống hệt nhau (đối xứng).
- He giữ std kích hoạt gần không đổi qua các lớp (`Var = 2/n_in`), Xavier co dần, `normal(0.01)` co ~10 lần mỗi lớp, đúng phân tích phương sai. **Khác dự đoán:** `normal` vẫn học tốt (0.8649, vượt 2σ) và He **không** hơn (default +0.0097 vượt 2σ; Xavier +0.0033 trong nhiễu). Có thể do lr 0.3 được chọn riêng cho He (logit lớn, loss bước 0 > ln 7): một yếu tố gây nhiễu (confound) mình chưa tách ra.

## 4. Đánh giá cuối trên tập eval

Chọn bằng val: gộp các thay đổi vượt nhiễu của Part 3 (Adam lr 3e-3, `M-wide`, 40 epoch). Ba ứng viên (seed 1, nhóm `other`): `cand-adam-wide` 0.8826, `cand-adam-ep40` 0.8950, **`cand-adam-wide-ep40` 0.9092** → chọn **Adam, lr 3e-3, 54-512-256-7 (161 287 tham số), 40 epoch, batch 512, CE, He, không dropout/clip, FP32**, chạy 5 seed (`final-s1..5`). Cấu hình này khác baseline ở ba yếu tố (bộ tối ưu, độ rộng, số epoch), nên "cải thiện" là của cả gói. Hiệu quả cộng dồn dưới mức tuyến tính (+0.0249 và +0.0373 riêng lẻ, +0.0515 khi gộp).

Điểm eval do `scripts/evaluate.py` tính (mỗi file dự đoán chấm đúng một lần, chỉ cho baseline và cấu hình cuối; file nộp `predictions_eval.csv` là của **`final-s1`**, seed chọn trước khi xem điểm eval):

| Cấu hình | Seed | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (5 seed, TB ± σ) | 1–5 | 0.8577 ± 0.0029 | 0.8585 ± 0.0040 | 0.9108 ± 0.0027 |
| Cấu hình cuối cùng (5 seed, TB ± σ) | 1–5 | 0.9086 ± 0.0006 | 0.9075 ± 0.0021 | 0.9394 ± 0.0013 |
| **File nộp (`final-s1`)** | 1 | 0.9092 | **0.9065** | **0.9402** |

- Cải thiện eval macro-F1 so với baseline: **+0.0490**, lớn hơn 2σ_eval của baseline (0.0079) nên có ý nghĩa; cải thiện val (+0.0509) và eval (+0.0490) nhất quán, và val ≈ eval (0.9092 so với 0.9065 ở `final-s1`), tức không có dấu hiệu lệch do chọn cấu hình bằng val.
- Điểm baseline `base-s1` trên eval là 0.8566.

### 4.1 Phân tích lỗi theo lớp (`final-s1`, `eval_result.json`, `figures/confusion_final.png`)

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 Spruce/Fir | 42 368 | 0.9561 | 0.9175 | 0.9364 |
| 1 Lodgepole Pine | 56 661 | 0.9378 | 0.9611 | 0.9493 |
| 2 Ponderosa Pine | 7 151 | 0.9480 | 0.9360 | 0.9419 |
| 3 Cottonwood/Willow | 549 | 0.8619 | 0.8069 | **0.8335** |
| 4 Aspen | 1 899 | **0.7873** | 0.8947 | 0.8376 |
| 5 Douglas-fir | 3 473 | 0.8805 | 0.9148 | 0.8973 |
| 6 Krummholz | 4 102 | 0.9451 | 0.9534 | 0.9493 |

- **Lớp khó nhất: lớp 3 (F1 = 0.834)**, rồi lớp 4 (F1 = 0.838). Lớp 3 chỉ có 549 mẫu eval (0.47%) nên vài chục lỗi đã kéo recall xuống 0.807; nó hay bị nhầm sang **lớp 2** (69 mẫu = 12.6% support) và lớp 5 (37 mẫu). Lớp 4 có precision thấp nhất (0.787): 370 mẫu thật là lớp 1 bị đoán thành lớp 4. Về số tuyệt đối, nhầm nhiều nhất là 0 ↔ 1 (3 230 mẫu 0 → 1 và 1 602 mẫu 1 → 0), nhưng support rất lớn nên F1 hai lớp này vẫn cao (0.936 / 0.949).
- **Lý giải bằng dữ liệu:** số mẫu ít (lớp 3, 4) và chồng lấn đặc trưng giữa các cặp lớp hay nhầm; macro-F1 (0.9065) thấp hơn accuracy (0.9402) vì lớp hiếm kéo xuống. Phần "chồng lấn đặc trưng" là giả thuyết chưa kiểm chứng bằng thí nghiệm.
- **Cải thiện sẽ thử:** CE có trọng số lớp hoặc lấy mẫu lại cho lớp 3, 4; huấn luyện lâu hơn (best epoch là 40 ở 3/5 seed); mạng rộng hơn.

![so sánh baseline và cấu hình cuối](figures/compare_final.png)

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng khi mỗi cái được chỉnh lr?** Adam (lr 3e-3, 0.8727) hơn SGD+momentum (0.8551) +0.015 (vượt 2σ), và hơn SGD thuần (0.8267). Khi lr không được chỉnh kết luận có thể đảo: Adam ở lr 3e-4 chỉ 0.785, còn SGD+momentum ở lr 0.01 chỉ 0.766 (đều thua lr tốt nhất của bộ kia). SGD+momentum ở lr 1.0 sập về 0.094. Vậy "ai thắng" phụ thuộc rất nhiều vào lr được chọn.
2. **Dropout có giúp khi chưa quá khớp?** Không: q = 0.1 / 0.3 / 0.5 giảm macro-F1 xuống 0.835 / 0.768 / 0.631 vì baseline chưa quá khớp (khoảng cách 0.023). Dùng dropout khi train thấp và val tăng dần.
3. **Clipping giải quyết vấn đề gì?** Chặn các gai gradient bất ngờ để lr cao không làm sập huấn luyện: ở lr 1.0, không clip sập (macro-F1 0.094, `grad_norm` lớn nhất 12.3), có clip c = 0.35 đạt 0.817. Không cứu được lr 3.0 (bước hiệu dụng vẫn quá lớn), nên không thay được việc chọn lr.
4. **Mixed precision có nhanh hơn không?** Không trên mạng nhỏ và MPS này: FP32 0.93 s/epoch, BF16 1.08 s, FP16 1.34 s; độ chính xác không đổi trong nhiễu. Chi phí ép kiểu và gọi kernel lớn hơn lợi ích từ phép nhân ma trận nhỏ. FP16 cần GradScaler vì khoảng giá trị hẹp, BF16 thì không.
5. **Vì sao `zeros` hỏng? He khác Xavier thế nào?** `zeros`: gradient của các lớp trước bằng 0 vì h = ReLU(0) = 0, nơ-ron cùng lớp đối xứng; mạng chỉ còn bias lớp ra nên bằng đoán lớp đa số (`init-zeros`: accuracy 0.4876). He dùng `Var = 2/n_in` (bù cho ReLU triệt một nửa) nên giữ std kích hoạt qua các lớp (0.67 → 0.59); Xavier dùng `2/(n_in + n_out)` nên co dần (0.28 → 0.20). Khác biệt này chỉ quan trọng với mạng sâu; ở mạng 3 lớp He không hơn Xavier (0.8551 vs 0.8610, trong nhiễu).
6. **Mạng có loss không giảm sau 2 000 bước: 3 phép kiểm tra đầu tiên.** (a) Loss bước 0 ≈ ln 7 = 1.946 (loss đo được 2.27 với He; lệch lớn nghĩa là khởi tạo/chuẩn hoá sai); (b) quá khớp 20 mẫu với mọi chính quy hoá tắt: loss phải về ≈ 0 (đạt 6·10⁻⁶), nếu không thì gần như chắc chắn lỗi code (nhãn, softmax hai lần, quên `zero_grad`); (c) kiểm tra mọi tham số có gradient khác 0 và ghi `train loss`, `val loss`, `grad_norm`. Thí nghiệm của mình cho thấy các triệu chứng: loss phẳng quanh entropy lớp 1.205 (`init-zeros`, `hp-lr-sgdm-1`, `clip-none-lr3`) là mạng chỉ đoán lớp đa số (gradient bằng 0 hoặc lr quá lớn làm nơ-ron chết), và `grad_norm` lên tới 6 633 cho thấy lr quá cao.

## 6. Hạn chế và điều bất ngờ

- **Một seed cho phần lớn thí nghiệm:** chỉ baseline và cấu hình cuối có 5 seed; các so sánh còn lại là một seed, so với σ của baseline. σ ước lượng từ 5 seed nên còn thô. Δ nhỏ hơn 2σ (ví dụ `hp-deep`, `amp-*`, `hp-batch128-lr0.075`) chỉ là chưa kết luận được.
- **Lưới lr thô** (cách nhau ~3 lần) và **lr baseline chỉ được chọn cho He**, nên so sánh init, loss có yếu tố gây nhiễu (confound). Baseline dùng 20 epoch nhưng best epoch ≈ 20: mô hình chưa hội tụ, xếp hạng có thể đổi nếu huấn luyện lâu hơn.
- **Bất ngờ so với dự đoán:** MSE ở lr cao tệ hơn; batch 128 cùng lr tệ hơn baseline; `M-deep` không giúp; wd 5e-4 làm xấu rất nhiều; clipping cải thiện ở lr baseline; clipping không cứu được lr 3.0; `normal(0.01)` học tốt và He không hơn default. Các giải thích cho những điểm này đều là giả thuyết chưa kiểm chứng (đã ghi rõ ở từng mục).
- **Trung thực về quy trình:** lưới lr và ba lựa chọn Adam lr 3e-3 / `M-wide` / 40 epoch được định hướng sau một lần quét thăm dò trên **val** (không dùng eval); một số dự đoán (optimizer, clipping, amp) viết sau khi đã thấy sơ bộ kết quả quét lr hoặc thử môi trường, và đã ghi chú ngay trong notebook. Cấu hình cuối cùng khác baseline ở ba yếu tố. Chạy `evaluate.py` chỉ cho baseline và cấu hình cuối (10 file dự đoán, 5 mỗi bên); `final-s1` được chọn làm file nộp trước khi xem điểm eval. Hai lần chạy notebook cho các chỉ số tóm tắt (val macro-F1, val acc, best val loss, best epoch, loss bước 0) trùng khớp tuyệt đối ở cả 42 thí nghiệm được so sánh, nhờ cố định seed.
- **Đo lường:** `peak_mem_MB` trên MPS không phải đỉnh thật; không dùng để kết luận. `experiments.xlsx` giữ các công thức của mẫu (cột `beyond_noise`, Seeds, Summary); mở bằng Excel để chúng tự tính lại.
- **Nếu có thêm thời gian:** chọn lr riêng cho từng init; thử CE có trọng số lớp; chạy nhiều seed cho các thí nghiệm có Δ sát ngưỡng (`clip-c0.35-lr0.3`, `init-default`); huấn luyện > 40 epoch; thử warmup lr cho batch lớn.

## 7. Phụ lục

- **File nộp:** `REPORT.md`, `experiments.xlsx` (50 dòng), `predictions_eval.csv`, `eval_result.json`, `figures/` (50 ảnh `<exp_id>.png`, 13 ảnh `compare_*.png`, `confusion_final.png`, `overfit20.png`), `results/` (50 file `<exp_id>.json`), `code/` (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
- **Thời gian chạy:** toàn bộ notebook ≈ 23 phút (GPU MPS của MacBook Air); ≈ 1 s/epoch với baseline.
