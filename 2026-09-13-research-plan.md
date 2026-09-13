---
layout: doc
title: "研究計劃書"
date: "2026-09-13"
order: 1
summary: "向量圖作為 CV 訓練輸入的有效性與標註效率 — 完整計劃、決策與變更歷程"
---

# 研究計劃書：向量圖作為 CV 訓練輸入的有效性與標註效率

- 版本：v0.2
- 日期：2026-09-06
- 硬體基準：NVIDIA RTX 4060 Laptop, 8GB VRAM / Windows 11 / CUDA 12.7 driver
- 文件約定：每完成一個「大目錄」步驟，回寫本文件並更新下方進度看板

---

## 0. 輪次規劃與進度看板

本計劃採**多輪制**。每一輪走完一次完整循環（架構 → 偽代碼 → 實作 → 實驗設計 → 實作實驗），
產出可驗證的結論後，才決定下一輪要做什麼。**後續輪次在前一輪結束前不展開。**

### 輪次總覽

| 輪次 | 目標 | 為什麼是這個順序 | 狀態 |
|---|---|---|---|
| **共通基礎** | 論點釐清、模型挑選 | 決定整個計劃的地基 | ✅ 完成（§1、§2） |
| **R1** | **重現 YOLaT 基線 + 驗證 H2（標註可大幅降低）** | 沒有可信基線就無法歸因；H2 是文獻缺口，優先卡位 | 🔵 進行中 |
| **R2** | 內層改造（移植 SymPoint 的 ACM / CCL） | 必須先有 R1 基線當對照，否則改動效果無法歸因 | 🔒 凍結 |
| **R3** | 擴展（分割任務、實例層級標註、真實資料） | 需 R1/R2 結論支撐 | 🔒 凍結 |

### R1 步驟看板

| # | 步驟 | 狀態 | 產出 |
|---|------|------|------|
| R1-1 | 模型架構（重現規格 + 對照裝置設計） | ✅ 完成 | §3 |
| R1-2 | 偽代碼生成 | ✅ 完成 | §4 |
| R1-3 | 實作模型 | ✅ 完成（基線 mAP 75.7） | §5 |
| R1-4 | 實驗設計 | 🔵 曲線執行中 | §6 |
| R1-5 | 實作實驗 | 🔵 向量組完成，點陣組進行中 | §7 |

**R1 出口條件**：見 §3.6。達成後才開 R2。

---

## 1. 研究論點

### 1.1 原始命題

> 在 CV 領域中，以向量圖作為訓練正負樣本輸入的效果，推測優於以點陣圖輸入 [1]，
> 且可以減少訓練數據所需標註量 [2]。
>
> 假設原因：
> 1. 向量圖的線條、色塊自帶弱邏輯連結 [1]
> 2. 不需要移動窗口、不需要特別計算特徵相對位置 [1]
> 3. 自帶弱邏輯連結所以標註可以大幅降低 [2]
>
> 範圍：先聚焦 CV 的分辨（分類／偵測）、切割（分割）等基礎任務。

### 1.2 文獻驗證結果

把命題拆成兩個可證偽的假設：

- **H1（效果）**：同任務同資料下，向量輸入模型 > 點陣輸入模型。
- **H2（標註效率）**：達到同等效果，向量輸入所需標註量顯著較少。

#### H1 → 文獻支持，但有明確適用邊界

| 證據 | 任務 | 結果 |
|---|---|---|
| YOLaT (NeurIPS'21 / T-PAMI'24) | 物件偵測 SESYD Floorplan | 向量 GNN mAP **90.59** vs Faster-RCNN-R50（ImageNet 預訓練）90.25；**參數 1.6M vs 41.4M、1.5 vs 165.6 GFLOPs** |
| YOLaT 同上 | Diagram | mAP 89.67，勝過所有「未預訓練」的點陣 baseline |
| SymPoint (ICLR'24) | 全景符號分割 FloorPlanCAD | 向量點表示 **PQ 83.3** vs GAT-CADNet 73.7 vs CADTransformer（含點陣 backbone）68.9 |
| SketchGNN (TOG'21) | 手繪筆劃語意分割 | 筆劃層級分割上 GNN 優於點陣 CNN |

**邊界條件（必須寫進論文限制）**：以上全部是「線條稿類」資料——CAD 圖、電路圖、圖表、手繪 sketch。
**目前沒有任何證據支持「把自然影像向量化後再輸入會比較好」**；自然影像的向量化本身有損且不穩定。

→ 本研究的適用域定義為 **native vector data**（原生即為 SVG / DWG / PDF-path 的資料），不含「點陣轉向量」。

#### 假設原因 1（弱邏輯連結）→ 成立，但機制描述需要精確化

真正生效的不是模糊的「弱邏輯連結」，而是三種**顯式結構先驗**，向量格式免費提供、點陣圖必須用 CNN 從像素學出來：

1. **連通性（connectivity）**：Bézier 曲線的端點共享關係。SymPoint 的 ACM 模組直接把「端點最小距離 < 1px 即視為連接」寫進 attention 的鄰接定義。
2. **群組（grouping）**：同一 `<path>` / `<g>` / CAD layer 的圖元天然屬於同一物件。
3. **屬性（attribute）**：stroke width、fill color、line style 是離散乾淨的欄位，不是需要從像素反推的統計量。

→ **修正表述**：「向量格式免費提供圖結構（graph）與群組監督訊號」。

#### 假設原因 2（不需移動窗口）→ 成立

- **不需滑窗**：計算量從 `O(H×W)` 降到 `O(#primitives)`。這正是 YOLaT 100× FLOPs 降幅的來源。YOLaT 也確實**不做 bbox regression、不需 anchor**，改用區域 cluster 內 10×10 grid 的頂點配對直接生成 proposal。

→ **正式表述**：「以圖元數取代像素數作為計算基底，免除滑窗與 anchor 機制。」

> **已刪除的子假設（v0.4，經決策移除）**：原始命題中的「不需要特別計算特徵相對位置」與文獻證據相反，已從論點中拿掉。
> 事實記錄（供 §3 架構設計參考，不再作為論點）：向量圖元是無序集合，缺少像素網格隱含的空間排序，
> **相對位置必須顯式編碼**。YOLaT 消融顯示拿掉邊屬性（控制點座標）掉 4.34 mAP、拿掉鄰居差掉 2.76 mAP；
> GAT（不顯式建模相對幾何）比 EdgeConv 式低 6.67 mAP。→ 顯式相對位置編碼是**架構必要元件**，而非論點主張。

#### H2（標註效率）→ 文獻未證實，這是研究缺口，也是本計劃最有價值的貢獻點

- **間接正面證據**：YOLaT 僅用 500 / 600 張訓練圖、無任何預訓練即達 SOTA（屬 *sample efficiency*，非 *label efficiency*）。
- **反面證據**：ArchCAD-400K (2025) 指出，向量圖元上的 self-supervised 預訓練雖優於不預訓練，但**仍低於監督式預訓練** → 結構先驗尚不足以取代標註。
- **關鍵事實**：目前**沒有任何論文做過「同任務同資料、向量 vs 點陣的標註量–效果曲線（label-efficiency curve）」的對照實驗**。

→ **結論**：H2 不能當已知前提引用，必須當作**本研究要驗證的主假設**。這也直接決定了 §6 實驗設計的核心。

### 1.3 修正後的研究論點（正式版）

> 在**原生向量圖**資料上執行分類／偵測／分割任務時：
>
> **(H1)** 直接以圖元（primitive）為計算單元的模型，在同等或更低的算力預算下，
> 效果不劣於以點陣化影像為輸入的模型；主因是向量格式顯式提供了連通性、群組與屬性先驗。
>
> **(H2，主假設)** 上述結構先驗可作為隱式監督訊號，使模型在**低標註比例**下的效果衰減
> 顯著慢於點陣輸入模型；即達到同等效果所需的標註量更少。

### 1.4 假設 → 實驗對應（供 §6 展開）

| 假設 | 驗證方式 | 證偽條件 |
|---|---|---|
| H1 | 同資料同預算，向量模型 vs 點陣模型的 mAP / PQ | 向量模型在同算力下顯著較差 |
| H2 | 標註比例 {1%, 5%, 10%, 25%, 50%, 100%} 的效果曲線，兩種輸入各跑一條 | 兩條曲線斜率無顯著差異 |
| 原因 1 | 消融：移除連通性邊 / 移除群組資訊 | 移除後效果不掉 |
| 原因 2 | 對比：圖元數 vs 像素數的實測 FLOPs 與推論延遲 | 向量路線算力未顯著較低 |

### 1.5 已決事項與暫緩事項（v0.4）

**已決**

| 項目 | 決定 |
|---|---|
| 「不需計算特徵相對位置」子假設 | **移除**。與 YOLaT 消融證據相反；改列為 §3 的架構必要元件 |
| 研究適用域 | 限定 native vector data（SVG / DWG / PDF-path），排除「點陣轉向量」 |
| **約束 C1（v0.8）** | **受試模型輸入必須為向量**，管線不得含光柵化步驟。不約束點陣對照組。詳見 §2.6 |
| H1 邊界 | 僅主張線條稿類資料，不主張自然影像 |

**暫緩（已知風險，本階段不處理，保留紀錄以便日後回頭）**

| 風險 | 狀態 | 若日後發生的徵兆 |
|---|---|---|
| **天花板效應**：SESYD 類資料 AP50 已 98.83、標準誤 ±0.0003，接近飽和 | 暫不考慮 | H2 曲線在中高標註比例區間兩條線重疊，看不出差異 |
| **光柵化解析度混淆變因**：點陣對照組的解析度會影響細線保存，未受控 | 不處理 | 審稿人質疑 H1 的比較公平性；點陣 baseline 弱勢無法歸因 |

---

## 2. 模型挑選

### 2.1 算力基準（本機）

| 項目 | 數值 | 備註 |
|---|---|---|
| GPU | RTX 4060 Laptop | Ada, 8GB GDDR6 |
| FP32 峰值 | ~11.6 TFLOPS | |
| FP16 / BF16 Tensor 峰值 | ~46 TFLOPS | dense，非稀疏 |
| 可用 VRAM | **~7.0 GB** | Windows WDDM 桌面佔用約 0.7–1.0 GB |
| 有效吞吐（假設值） | 密集 CNN / Attention + AMP：**12–15 TFLOPS**；稀疏 GNN（gather/scatter, FP32）：**0.5–2 TFLOPS** | GNN 受記憶體頻寬與 kernel launch 限制，利用率通常低於 10% |
| 平均功耗 | ~60 W | 用於能耗估算 |

> 估算公式：`訓練 FLOPs ≈ 3 × 前向 FLOPs × 樣本數 × epoch 數`（前向:反向 ≈ 1:2）
> `wall-clock ≈ 訓練 FLOPs / 有效吞吐 × (1 + 資料處理開銷係數)`

### 2.2 候選架構表

| # | 架構 | 輸入表示 | 任務 | 參數量 | 前向 FLOPs/樣本 | 可改的「內層」 | 優點 | 缺點 | 8GB 可行性 |
|---|---|---|---|---|---|---|---|---|---|
| **A** | **YOLaT / YOLaT++**（dual-stream GNN） | Bézier 控制點無向多重圖（stroke-wise + position-wise 雙邊型） | 偵測 | **1.6 M** | **1.5–2.9 GFLOPs** | 訊息傳遞層、邊型定義、proposal 生成器 | 算力極低；官方 PyTorch 程式碼；與論點完全對齊；從零訓練即 SOTA | 僅驗證於小型線稿資料集；GNN 深度受 over-smoothing 限制；proposal 為矩形枚舉 | 極寬裕（<2 GB） |
| **B** | **SymPoint / SymPoint-V2**（Point Transformer + Mask2Former head） | 圖元轉 8 維點集 | 全景分割 | 35 M | 未公布（點數相依） | ACM 連接注意力、CCL 對比損失、KNN 插值 | FloorPlanCAD SOTA (PQ 83.3)；ACM/CCL 兩模組概念可移植 | **官方訓練需 8×A100 × 1000 epochs**，本機重現不可行 | 全尺寸不可行 |
| **C** | **CADTransformer** | HRNet 點陣特徵 + 圖元查詢 | 全景分割 | ~70 M | 高（含 HRNet-W48） | backbone、查詢層 | 有官方程式碼 | **依賴點陣 backbone，與研究論點自相矛盾** | 勉強，且不該用 |
| **D** | **GAT-CADNet** | 圖元圖 + GAT | 全景分割 | 中（未公布） | 中 | 注意力層、相對位置編碼 | 純圖結構、概念乾淨 | 無官方程式碼（僅第三方複現）；已被 B 大幅超越 | 可行 |
| **E** | **SketchGNN** | 筆劃 / 取樣點圖 | 分割 | ~1–2 M | 低 | 圖卷積層、池化 | 極輕量；分割任務直接可用 | 僅適用手繪 sketch 域 | 極寬裕 |
| **F** | **PointTransformer-v3 / DGCNN（自行縮小）** | 圖元點雲 | 分類 / 分割 | 可調 1–10 M | 可調 | 全部（通用骨幹） | 完全自由、規模可任意調整 | 需自行接任務頭；無領域先驗，等同從頭做 | 可行 |
| **G** | **DeepSVG encoder** | SVG 指令序列（階層 Transformer） | 生成為主 | ~5–10 M | 中 | encoder 層、階層聚合 | 能吃完整 SVG 語法 | 為生成設計，判別任務需重接；序列化會丟失 2D 拓撲 | 可行 |
| **H** | **點陣對照組：YOLOv8-n/s** | 640×640 影像 | 偵測 | 3.2 / 11.2 M | 8.7 / 28.6 GFLOPs | 僅作 baseline | 成熟、易跑 | 非研究主體 | 可行 |

### 2.3 算力估算（RTX 4060 8GB，單次訓練 run）

以 SESYD Floorplan 規模（500 train / 450 test）為基準：

| 架構 | 設定 | 訓練 FLOPs | 純計算時間 | **實務 wall-clock 估計** | VRAM | 能耗 |
|---|---|---|---|---|---|---|
| **A. YOLaT** | 500 img × 500 ep | ~1.1 PFLOP | ~18 min | **2–4 hr**（受 CPU 端建圖與 scatter 支配，非計算受限） | ~1.5–2 GB | ~0.2 kWh |
| **E. SketchGNN** | 類似規模 | ~0.5 PFLOP | ~10 min | 1–2 hr | ~1 GB | ~0.1 kWh |
| **F. PT-v3（縮小版 4M）** | 500 × 300 ep | ~5 PFLOP | ~1.5 hr | 4–8 hr | ~4–6 GB | ~0.4 kWh |
| **H. YOLOv8-s（對照）** | 500 img × 300 ep, 640px, AMP | ~13 PFLOP | ~20 min | **1–2 hr** | ~4 GB @ bs16 | ~0.1 kWh |
| **B. SymPoint 全尺寸** | 11.6k img × 1000 ep, 8×A100 | 推估 400–800 A100-hr | — | **推估 2,500–8,000 hr，不可行** | 需 >24 GB | — |
| **B-lite. SymPoint 縮小** | 通道減半 + 300 ep + 20% 子集 | — | — | 60–150 hr（一週以上） | ~6–7 GB（臨界） | — |

**結論：只有 A / E / F / H 落在本機可負擔範圍。B 只能當文獻對照，不自行訓練；C 與研究論點衝突，排除。**

### 2.4 建議（算力優先）

**主線：以 A（YOLaT / YOLaT++ 的 dual-stream GNN）為骨幹，把 B（SymPoint）的 ACM 與 CCL 概念移植進其內層。**

理由：

1. **算力**：1.6 M 參數 / 1.5 GFLOPs，是候選中唯一能讓「30+ 次 run 的標註效率曲線實驗」在單張 4060 上於數天內跑完的方案。這對 H2 是硬需求——H2 需要大量重複 run，模型必須夠便宜。
2. **論點對齊**：YOLaT 本身就是「向量勝點陣」的原始證據；在它上面做改動，實驗結論可直接對話到 [1]。
3. **可改內層明確**：雙流訊息傳遞層是乾淨的介入點（見 §3）。
4. **對照組現成**：YOLOv8-s 作點陣對照，1–2 hr/run，成本對稱。

**總實驗預算估計**

- Phase A 重現基線：3 run × 3 hr ≈ **9 hr**
- Phase B 標註效率曲線：2 模態 × 6 比例 × 3 seed = 36 run ≈ **90 hr**
- Phase C 內層消融：6 變體 × 3 seed × 3 hr ≈ **54 hr**
- **合計 ≈ 155 GPU-hr ≈ 連續 6.5 天（實務排程約 2–3 週）**，電力成本可忽略。

**風險備註**：本機 Python 為 3.13.0b1，PyTorch / PyTorch-Geometric 尚無穩定 wheel。§5 環境建置需降到 **Python 3.11 或 3.12**。

### 2.5 算力限制解除後的重新評估（v0.7）

> 決策變更：不再以本機 RTX 4060 為約束，改為可租用雲端算力。§2.1–§2.4 的結論需重新檢視。

#### 2.5.1 IconShop 與 StarVector 的實際規格

| 項目 | **IconShop** (TOG 2023) | **StarVector-1B** | **StarVector-8B** |
|---|---|---|---|
| 本質 | 文字 → SVG 生成 | **點陣圖 → SVG 程式碼**（向量化） | 同左 |
| 架構 | 12 層 transformer decoder | CLIP ViT-B/32 + StarCoder-1B + FC adapter | SigLIP so400m + StarCoder2-7B |
| 參數量 | 未明述（推估 50–100 M） | ~1.2 B | ~8 B |
| **序列長度上限** | **512 token（icon）** + 50 token（文字） | **8 k token** | **16 k token** |
| SVG 詞彙 | 10,007（3 命令 + 10,000 座標 + 4 特殊） | StarCoder tokenizer | 同左 |
| 訓練資料 | FIGR-8-SVG，取 300 k（270 k 訓練） | SVG-Stack 2.1 M | 同左 |
| 訓練成本 | A100，300 epochs，batch 192，lr 6e-4 | **8×A100 × 7 天 = 1,344 A100-hr** | **64×H100 × 10 天 = 15,360 H100-hr** |
| 支援任務 | 生成、編輯、內插、語意組合 | 影像轉 SVG、文字轉 SVG、圖表生成 | 同左 |
| **判別任務（分辨／切割）** | **無** | **無** | **無** |
| 其他限制 | **僅黑白單色 icon** | 作者自陳 16k context「不足以應付複雜 SVG」 | 同左 |

#### 2.5.2 三個決定性問題

**問題 1：兩者都是純生成模型，沒有任何判別能力。**
IconShop 支援生成／編輯／內插／語意組合，StarVector 支援影像轉 SVG 與文字轉 SVG。
兩篇論文都**未評估任何分類、偵測或分割任務**。要用於本研究，必須自行加判別頭並重新訓練。

**問題 2：StarVector 的輸入是點陣圖，方向與本研究相反。**
StarVector 是**向量化器**（raster → vector），不是「以向量為輸入的辨識器」。
若要用它，只能取其 LLM 半邊（StarCoder / StarCoder2）讀 SVG 文字——但那就落入問題 3。

**問題 3（最致命）：序列長度天花板與目標資料的規模差 1–2 個數量級。**

| 資料 | 規模 | 對應 token 數 |
|---|---|---|
| IconShop 訓練域 | 單色 icon，數十條路徑 | ≤ 512 |
| StarVector 訓練域 | 一般 SVG icon / logo | ≤ 8 k–16 k |
| **SESYD floorplan / FloorPlanCAD** | **數百至數千個圖元** | **推估 10 k–100 k+** |

Transformer 的 attention 是 O(L²)，而 GNN / point-based 方法是 O(N + E) 的稀疏結構——
**這正是 YOLaT 能用 1.5 GFLOPs 處理整張圖的原因**。序列模型在 CAD 尺度上不是「貴」，是**根本放不進去**。
StarVector 作者自陳「16k context 不足以應付複雜 SVG」即為此問題。

#### 2.5.3 警訊：raw SVG 對語言模型是壞表示

VGBench (EMNLP 2024) 的理解任務準確率：

| 模型 | SVG | TikZ | Graphviz |
|---|---|---|---|
| GPT-4 | **54.9%** | 81.0% | 84.5% |
| GPT-3.5 | 43.7% | 62.6% | 69.9% |
| Llama-3-70B | 53.4% | 71.1% | 69.5% |
| Llama-3-8B | 40.0% | 54.5% | 58.8% |

作者結論：「TikZ 與 Graphviz 含有比 SVG 更高階的語意，SVG 由低階幾何圖元構成。」

GPT-4 在**顏色／類別／用途**這種簡單問題上只有 54.9%。
→ **把 SVG 當文字餵給 LLM 是弱路線**，至少在現有 tokenization 下如此。
（註：VGBench **未**比較「LLM 讀向量碼 vs VLM 讀點陣圖」，故不能作為 H1 的證據，正反皆然。）

#### 2.5.4 唯一有價值的用法：預訓練編碼器——而且它直接打中 H2

兩者都不能當骨幹，但 **IconShop 的訓練目標**值得抽出來用。

IconShop 有 `<Mask>` token、以雙向脈絡做填補（用於 icon 編輯）——這本質上就是 **SVG 版的 MLM**。
而 YOLaT 作者在限制中自陳：「需要大型向量圖資料集來支撐 backbone 預訓練」。兩者正好對接。

**為什麼這條路對 H2 特別有利**：SVG 是**離散 token 序列**，可直接套用 MLM，
不需要像影像自監督那樣設計 augmentation 策略。無標註 SVG 資料極多（SVG-Stack 2.1 M、FIGR-8 1.5 M）。

**但有一個必須處理的矛盾**：一旦引入大規模預訓練，H2 的結論會變成
「預訓練有效」而非「向量表示有效」——與 §4.8 要求對照組 `pretrained=False` 是同一個問題。

**解法：改成 2×2 因子設計**，把「表示形式」與「預訓練」解耦：

| | 無預訓練 | 有預訓練 |
|---|---|---|
| **向量輸入** | YOLaT (from scratch) | SVG-MLM 預訓練 → YOLaT 微調 |
| **點陣輸入** | YOLOv8 (from scratch) | YOLOv8 (ImageNet/COCO 預訓練) |

四條標註效率曲線。這比原本的 2 條資訊量高得多，且能回答「向量的優勢是來自表示本身，還是來自更好的預訓練」。

#### 2.5.5 兩條可行路線（域的選擇）

IconShop / StarVector 的序列上限使它們**只適用於 icon 尺度**。因此出現一個域的分岔：

| | **路線 A：CAD／floorplan 域** | **路線 B：icon 域** |
|---|---|---|
| 任務 | 符號偵測、全景分割 | icon 分類（分辨） |
| 資料 | SESYD、FloorPlanCAD (11.6 k) | FIGR-8-SVG (1.5 M，**自帶類別標籤**) |
| 骨幹 | YOLaT（偵測）、SymPoint（分割） | IconShop 式 transformer + 分類頭 |
| IconShop/StarVector | ❌ 序列長度不足 | ✅ 正是其設計域 |
| H2 的機制 | 結構先驗當歸納偏置 | 大規模 SSL 預訓練 + 少量標註 |
| 預訓練語料 | 缺（需自行蒐集 CAD SVG） | **充足**（FIGR-8 1.5 M、SVG-Stack 2.1 M） |
| 切割任務 | ✅ SymPoint | ❌ icon 無分割標註 |
| 統計檢定力 | 弱（SESYD 500 張、已飽和） | **強**（1.5 M 樣本，可壓到 0.01% 標註） |
| 成本 | $1,300–4,800（SymPoint 全尺寸） | $140–480（IconShop 級重訓） |

#### 2.5.6 租用算力成本估算

假設 A100 80GB $1.4–2.5/hr、H100 $2.5–4.0/hr（2026 年 community cloud 行情）：

| 項目 | 算力 | 估計成本 |
|---|---|---|
| YOLaT 重現（單 run） | 2–4 GPU-hr | < $10 |
| **SymPoint 全尺寸重現** | 8×A100 × 估 5–10 天 = 960–1,920 A100-hr ⚠️ 天數為推估 | **$1,300–4,800** |
| IconShop 級模型重訓 | 估 96–192 A100-hr | $140–480 |
| StarVector-1B 從頭預訓練 | 1,344 A100-hr | $1,900–3,400 |
| StarVector-1B LoRA 微調 | 6–24 A100-hr | $10–60 |
| StarVector-8B 從頭預訓練 | 15,360 H100-hr | **$38,000–61,000 → 排除** |
| R1 完整標註效率曲線（2×2 設計） | 約 300–600 GPU-hr | $400–1,500 |

**結論：算力解除後，SymPoint 全尺寸與 IconShop 級預訓練都進入可行範圍；StarVector 8B 仍排除，
1B 僅在 icon 域且以微調方式使用才合理。**

#### 2.5.7 重新納入的候選（先前因算力被濾除）

**(a) 判別骨幹 — CAD／技術圖域**

| # | 架構 | 年份 | 表示 | 規模 | 為何現在可考慮 | 保留疑慮 |
|---|---|---|---|---|---|---|
| **B** | **SymPoint** | ICLR'24 | 圖元→8 維點 | 35 M；8×A100×1000 ep | 成本 $1.3–4.8 k，已可負擔。分割任務的乾淨純向量 SOTA | 訓練仍最貴；1000 epochs 難壓縮 |
| **B2** | **SymPoint-V2**（Layer Feature Enhancement） | 2024 | 同上 + CAD 圖層特徵 | 同級 | 利用 CAD **layer** 屬性——正是 §1.2「群組先驗」的直接實例 | 依賴 layer 標註品質；非所有 SVG 有此欄位 |
| **I** | **DPSS**（ArchCAD-400K baseline） | NeurIPS'25 | **圖元特徵 + 點陣特徵自適應融合** | 未公布 | **目前 SOTA**，較前 SOTA +3 PQ | ⚠️ **雙路徑融合點陣特徵，非純向量**——見 §2.5.8 |
| **J** | **CADSpotting** | 2024 | 每圖元用**密集點**表示 + 3D 點雲模型 | 未公布 | 針對大尺度 CAD 圖設計，解決 SymPoint 的規模問題 | 密集取樣使點數暴增，成本高 |
| **L** | **Point or Line**（線段表示） | 2025 | 以**線段**而非點為單元 | 未公布 | 直接檢驗「表示粒度」這個變因，與本研究論點高度相關 | 較新，複現資料少 |
| **K** | **VectorGraphNET** | 2024 | GAT on 技術圖 | 未公布 | 純圖注意力路線的近期版本 | 已被 B/I 超越 |
| **C** | **CADTransformer** | CVPR'22 | HRNet 點陣 + 圖元查詢 | ~70 M | 算力已非問題 | **仍與純向量論點衝突**，僅能當「混合路線」對照 |

**(b) 表示學習／預訓練編碼器 — 這是 H2 的關鍵軸**

| # | 架構 | 年份 | 為何重要 | 疑慮 |
|---|---|---|---|---|
| **M** | **SVGformer** (CVPR'23, Adobe) | 2023 | **直接處理連續值、不量化 SVG 參數**，明確支援 reconstruction／**classification**／interpolation／retrieval 等下游任務。是目前最貼近「向量表示學習」的現成方案 | 規模與訓練細節需查全文；是否支援 CAD 尺度未知 |
| **G** | **DeepSVG encoder** | NeurIPS'20 | 階層式 encoder（path 層 + 命令層），能吃完整 SVG 語法 | 為生成設計；序列化丟失 2D 拓撲 |
| **IconShop-MLM** | 自建 | — | 借用其 `<Mask>` 雙向填補目標當 SVG 版 MLM 預訓練 | 需自行實作；僅驗證於 512 token 的單色 icon |
| **N** | **PointTransformer-v3 全尺寸** | 2024 | 通用點雲骨幹，可在無標註圖元上做 SSL 後接任務頭 | 無領域先驗 |

**(c) 明確排除**

| 架構 | 理由 |
|---|---|
| StarVector-8B | $38 k–61 k 預訓練成本，且輸入為點陣圖、context 不足 |
| StarVector-1B（CAD 域） | 8 k context 遠不足；僅 icon 域微調可考慮 |
| IconShop（CAD 域） | 512 token、僅單色 icon |

#### 2.5.8 一個必須面對的壞消息：2025 SOTA 不是純向量

**ArchCAD-400K 的 DPSS 是「雙路徑」——圖元特徵 + 點陣影像特徵自適應融合**，
並以此取得 SOTA（較前 SOTA +3 PQ）。CADTransformer（CVPR'22）同樣是混合路線。

這對**嚴格版 H1**（純向量 > 純點陣）是負面訊號：領域最新進展正在**回頭融合點陣特徵**，
暗示兩種表示的資訊**互補**，而非向量單方面勝出。

**對本研究的影響**：H1 的主張必須從「向量勝過點陣」弱化為
**「向量提供點陣所缺的結構資訊，且在低標註條件下這個資訊更關鍵」**——
亦即把重心明確押在 **H2**，而非 H1。這與 §1.2 已認定「H1 已被 YOLaT 證實、H2 才是缺口」的判斷一致，
但現在有了更強的理由：**H1 不只是已被做過，甚至正在被反向修正。**

#### 2.5.9 H2 的競爭解法：自動標註引擎

ArchCAD-400K 的另一項貢獻是**標註引擎**：利用 CAD 檔案的內建屬性（圖層名稱、區塊定義等）
**自動生成高品質標註，大幅減少人工標註**。

這是「減少標註」的**另一條路徑**，且已被實作。本研究若主張 H2，必須在相關工作中明確區隔：

| 路徑 | 機制 | 本研究關係 |
|---|---|---|
| ArchCAD-400K 標註引擎 | 從 CAD **metadata** 自動產生標註 | 屬**資料工程**，不改變模型所需的標註量 |
| **本研究 H2** | 向量表示的**結構先驗**作為歸納偏置／自監督訊號 | 屬**學習效率**，減少的是模型所需標註量 |

兩者不衝突，但論文必須講清楚差異，否則會被質疑「這問題已經解決了」。

---

## 2.6 硬約束：受試模型的輸入必須為向量（v0.8）

> **約束 C1**：本研究的**受試模型**，其輸入必須是向量表示（圖元／路徑／SVG token），
> 管線中不得包含將輸入光柵化的步驟。

**適用範圍界定**：C1 約束的是**受試模型**，不約束**對照組**。
點陣對照組（YOLOv8 等）必須以點陣為輸入——那正是 H1／H2 要比較的對象。
若把 C1 套到對照組，實驗將無從進行。

### 2.6.1 全候選重新篩選

| 判定 | 架構 | 輸入 | 在本研究的角色 |
|---|---|---|---|
| ✅ **純向量** | **YOLaT / YOLaT++** | Bézier 控制點多重圖 | **偵測主線** |
| ✅ | **SymPoint / SymPoint-V2** | 圖元→8 維點集 | **分割主線**（V2 另用 CAD layer 屬性） |
| ✅ | **SVGformer** (CVPR'23) | 連續值 SVG 參數，不量化 | **表示學習／預訓練軸主線** |
| ✅ | **CADSpotting** | 圖元→密集點 | 大尺度 CAD 備案 |
| ✅ | **Point or Line** (2025) | 線段為單元 | 表示粒度消融 |
| ✅ | VectorGraphNET / GAT-CADNet | 圖元圖 + GAT | 對照參考 |
| ✅ | SketchGNN | 筆劃圖 | sketch 域備案 |
| ✅ | DeepSVG encoder | SVG 指令序列 | 預訓練軸備案 |
| ✅ | **IconShop 式 MLM encoder** | SVG token 序列 | icon 域預訓練軸（**注意：作為 encoder 讀 SVG token，符合 C1**） |
| ✅ | PointTransformer-v3（接圖元） | 圖元點雲 | 通用骨幹備案 |
| ⚠️ **混合** | **DPSS**（ArchCAD-400K, NeurIPS'25 SOTA） | 圖元 + **點陣影像特徵融合** | **違反 C1，不可作受試模型**；僅能引用為「混合路線的效果上限」 |
| ⚠️ | CADTransformer | HRNet 點陣 + 圖元查詢 | 同上，僅作文獻對照 |
| ❌ **點陣輸入** | **StarVector-1B / 8B** | **影像 → SVG 程式碼** | **完全排除**。輸入方向與 C1 相反 |
| 🔵 **對照組** | YOLOv8-n/s、Faster R-CNN | 點陣影像 | **必要對照組**，C1 不適用 |

### 2.6.2 C1 的三個連帶影響

1. **StarVector 完全出局**，不再列入任何角色。其 LLM 半邊（StarCoder2 讀 SVG 文字）雖符合 C1，
   但受 §2.5.2 問題 3（context 長度）與 §2.5.3（VGBench SVG 僅 54.9%）雙重否決。
2. **DPSS 從「候選」降為「文獻對照」。** 但它是現行 SOTA，論文中必須正面處理：
   本研究是在 C1 約束下追求最佳效果，而非追求絕對 SOTA。此立場需在論文中明述，否則會被質疑迴避比較。
3. **§2.5.4 的 2×2 因子設計仍然成立**——向量列與點陣列本就分屬受試組與對照組。

### 2.6.3 C1 下的路線比較（更新）

| | **路線 A：CAD／floorplan** | **路線 B：icon** |
|---|---|---|
| 受試骨幹（皆符合 C1） | YOLaT（偵測）、SymPoint（分割） | IconShop-MLM encoder、SVGformer |
| 預訓練語料 | 缺（需自行蒐集 CAD SVG） | 充足（FIGR-8 1.5 M、SVG-Stack 2.1 M） |
| 統計檢定力 | 弱（SESYD 500 張、已飽和） | 強（1.5 M，可壓至 0.01% 標註） |
| 切割任務 | ✅ SymPoint | ❌ |
| 成本 | $1.3–4.8 k | $140–480 |
| C1 相容性 | ✅ 全部主線候選皆純向量 | ✅ 同左 |

> C1 不改變 A/B 的優劣關係——**兩條路線的主線候選都已是純向量**。
> C1 的實際效果是移除 StarVector 與混合路線，使候選集更乾淨。

## 2.7 最終決策：路線 A，YOLaT 骨幹（v0.9）

> **決策**：採路線 A，以 **YOLaT 雙流 GNN** 為 R1 受試骨幹。§3、§4 已依此撰寫，**內容全部維持有效**。

### 2.7.1 探索過程留下的結論（不再變動）

| 議題 | 結論 |
|---|---|
| IconShop / StarVector | 不採用。StarVector 違反 C1；IconShop 受 512 token 限制，不適用 CAD 尺度 |
| 路線 B（icon 域） | 不採用。無法涵蓋原始計劃書要求的「切割」任務 |
| DPSS / CADTransformer | 降為文獻對照，論文須正面表態 C1 立場（§2.6.2） |
| SVGformer / IconShop-MLM | **保留為 R2 預訓練軸候選**，R1 不啟用 |
| 2×2 因子設計（§2.5.4） | **移至 R2**。R1 維持 2 條曲線，避免預訓練污染 H2 的歸因 |

### 2.7.2 算力解除對 R1 的實際影響

YOLaT 本身極廉價（2–4 GPU-hr/run），因此算力解除**不改變模型與架構**（§3、§4 不動），
只放寬實驗規模。以下三項成本增幅極小、對結論品質提升明顯，建議納入 R1：

| 調整 | 原設定 | 新設定 | 增加成本 |
|---|---|---|---|
| seed 數 | 3 | **低比例區間 5，其餘 3** | +$30 |
| 標註比例點 | 6 點（1%–100%） | **8 點**，下探 0.5%、2% | +$40 |
| 平行執行 | 序列，2–3 週 | 多機平行，**2–3 天** | 同總量 |

> **暫緩事項狀態不變**：天花板效應（§1.5）仍不處理。
> 但註記——算力解除後，改用 FloorPlanCAD（11.6 k 圖）已屬低成本選項，
> 若 R1 曲線出現高比例區間重疊，可低代價回頭處理。

### 2.7.3 環境策略變更

| 用途 | 環境 |
|---|---|
| 開發、除錯、小樣本驗證 | 本機 RTX 4060 8GB / Windows。**須另建 Python 3.11 venv**（3.13.0b1 無 PyG wheel） |
| 正式訓練、全部實驗 run | 租用 Linux + A100/L40S。PyG 安裝較單純，且可多機平行 |

→ 程式碼須同時支援兩種環境，不得寫死 CUDA 版本或路徑。

---

## 3. 模型架構（R1-1）

**選定骨幹：A — YOLaT 雙流 GNN**（決策於 v0.5）

### 3.1 R1 的架構原則：不改內層

R1 的目的是建立**可信基線**與**乾淨的對照裝置**。因此：

> **R1 對 YOLaT 模型本體零改動。** 所有內層改造（ACM、CCL、階層化）一律留到 R2。

理由：H2 要量的是「輸入表示（向量 vs 點陣）對標註效率的影響」。若同時改架構，
曲線差異將無法歸因於表示形式，H2 的結論會被污染。R1 唯一的新增物是**資料層的標註子集裝置**（§3.3）。

### 3.2 需重現的完整規格

| 模組 | 規格 |
|---|---|
| 圖元轉換 | 所有 SVG 圖元 → 三次貝茲曲線；端點成節點，離曲線控制點成邊屬性 |
| 節點特徵 | 7 維：`(pˣ, pʸ, R, G, B, w)` |
| stroke-wise 邊 ℰₛ | 端點間存在貝茲曲線；邊屬性 = 兩個離曲線控制點座標（4 維） |
| position-wise 邊 ℰₚ | 空間 cluster 內全連接；無邊屬性 |
| cluster 劃分 | stroke 邊取連通分量 → 各分量最小外接矩形向外擴張（擴張長度為超參）→ 重疊者合併 |
| GNN 深度 | 雙流各 **2 層**（T=2），hidden dim **64** |
| Stroke 流 | `hᵢᵗ⁺¹ = fˡ(hᵢᵗ) + mean_{j∈𝒩ᵢˢ} fˢ(concat(hᵢᵗ, hⱼᵗ−hᵢᵗ, xᵉᵢⱼ))`，`fˢ` = Linear→ReLU→BN |
| Position 流 | `zᵢᵗ⁺¹ = mean_{j∈𝒩ᵢᵖ∪{i}} fᵖ(zⱼᵗ)`；**僅末層聚合**；`fᵖ` 與 `fˢ` 參數不共享 |
| mean-pool 優化 | 先逐節點算 `fᵖ(zᵢ)` → 每 cluster 取一次均值 → 廣播回節點。O(\|C\|²) → O(\|C\|) |
| Proposal | 每 cluster 切 10×10 網格，枚舉頂點對（i 左上、j 右下）成 axis-aligned 矩形，超尺寸閾值者濾除 |
| 特徵融合 | `r = concat(rₛ⁰..rₛᵀ, rₚ⁰..rₚᵀ)`，`rₛᵗ = mean_{i∈𝒱ʳ} hᵢᵗ`。維度 = 3 時間步 × 64 × 2 流 = **384**（待與官方碼核對） |
| 分類頭 | MLP 384 → 512 → 256 → (C+1)，含背景類 |
| 標籤指派 | `max IoU(B̂, B_gt) ≥ α` 取該類別，否則背景類 C |
| 損失 | 僅 cross-entropy，**無回歸損失** |
| 訓練超參 | batch size 4、lr 2.5e-4；Floorplan `bbox_sampling_step=10`、Diagram `=5` |

### 3.3 R1 唯一新增：標註子集裝置（label-subset harness）

這是**資料層**元件，不動模型。但其設計含一個會決定 H2 成敗的技術判斷：

#### 技術問題：偵測任務的「標註比例 p%」定義有歧義

| 抽樣方式 | 定義 | 用於 R1？ |
|---|---|---|
| **圖片層級（image-level）** | 抽 p% 的圖片，這些圖片 **100% 標註**；其餘圖片完全不進訓練集 | ✅ **採用** |
| 實例層級（instance-level） | 所有圖片都用，但每張圖只標註 p% 的物件 | ❌ R1 不用 |

**為什麼必須用圖片層級：**

YOLaT 的訓練是 proposal 分類，**未匹配任何 GT 的 proposal 會被指派為背景類 C**。
若採實例層級抽樣，未被標註的真實物件所在的 proposal 會被錯誤標成背景 → **假負樣本污染**。
此時效果下降不是因為「標註量少」，而是因為「標籤是錯的」——H2 的曲線會失去意義。

→ 實例層級抽樣需要 partial-label loss（把未標註區域從損失中 mask 掉），列為 **R3 的獨立議題**。

#### 兩個必要的對照控制

1. **向量組與點陣組必須共用同一批圖片 ID。** 同 seed、同 split 檔。
   否則曲線差異可能來自「抽到的圖片難度不同」，而非表示形式。
   → 實作上先產生一份 `splits/{ratio}_{seed}.json`，兩組都讀同一份。
2. **低比例區間變異數極大。** 1% × 500 張 = 5 張圖，單次 run 的隨機性會蓋過訊號。
   → 每個資料點至少 3 seeds（低比例區間建議 5），一律報 **mean ± std**，禁止只報單點。

### 3.4 需要用到的技術

| 技術 | 用途 | 風險 |
|---|---|---|
| PyTorch Geometric 2.x `MessagePassing` | 雙流訊息傳遞 | 與原碼的 PyG 1.x API 不相容 |
| `torch_scatter` / PyG 內建 scatter | cluster mean-pool、區域特徵聚合 | PyG 2.4+ 已內建，原碼的獨立 `torch_scatter` 相依需移除 |
| `scipy.sparse.csgraph.connected_components` | stroke 圖取連通分量 | — |
| Union-Find + 矩形重疊判定 | cluster 合併 | 擴張長度超參需調 |
| `svgpathtools` / `svgelements` | SVG 解析與圖元→貝茲轉換 | 不同 SVG 產生器的語法差異 |
| `ultralytics` (YOLOv8) | 點陣對照組 | — |
| `cairosvg` / `resvg` | SVG → PNG 光柵化（供對照組） | 解析度選定（§1.5 已列為暫緩風險，R1 固定單一解析度） |

### 3.5 預期技術問題

| # | 問題 | 對策 |
|---|---|---|
| 1 | 原碼基於 2021 年 PyG 1.x + DeepGCN + Python 3.8，`MessagePassing` signature 與 `torch_scatter` 相依皆已變更 | 不硬改原碼，**依 §3.2 規格重寫**乾淨版本，用原碼當語意參照 |
| 2 | 本機 Python 3.13.0b1 無 PyG wheel | 建 **Python 3.11 venv**（§5 處理） |
| 3 | SESYD Floorplans / Diagrams 下載連結可能失效 | 備案順序：SESYD 原站 → 論文附帶連結 → FloorPlanCAD 改做分割 |
| 4 | proposal 枚舉的記憶體：10×10 網格 → C(100,2)=4950 proposals/cluster，一張圖多 cluster 可達上萬 | batch 4 已是原設定；必要時對背景 proposal 做負採樣 |
| 5 | GPU scatter 操作非確定性，影響 seed 重現性 | 固定 seed + `torch.use_deterministic_algorithms(True)`；若拖慢過多則改為報多 seed 統計 |
| 6 | **建圖快取與標註無關** | 圖結構只依賴 SVG 幾何。**一次建好可跨所有標註比例重複使用**，省下 36 次 run 的重複前處理 |
| 7 | 低標註比例下 batch 4 可能一個 epoch 只有 1–2 個 batch | 改以 **iteration 數**而非 epoch 數對齊訓練預算，否則低比例組實際訓練步數遠少於高比例組，形成混淆 |

> 問題 7 特別重要：若沿用「固定 epoch 數」，1% 組只會訓練 500 × 1% ÷ 4 ≈ 1 步/epoch，
> 與 100% 組的 125 步/epoch 差 125 倍。**必須固定總 iteration 數**，才能把「標註量」與「訓練量」解耦。

### 3.6 R1 出口條件

| # | 條件 |
|---|---|
| 1 | 重現 YOLaT Floorplan mAP 落在 **90.59 ± 1.5** 之內（100% 標註） |
| 2 | 點陣對照組（YOLOv8-s，未預訓練）跑通並取得基線數字 |
| 3 | 標註效率曲線完成：6 比例 × 2 模態 × ≥3 seeds，全部附 mean ± std |
| 4 | 對 H2 給出明確判定：支持 / 不支持 / 無法判定（含理由） |

達成後才開 R2。

## 4. 偽代碼生成（R1-2）

### 4.1 架構再確認：三個必須向官方碼核對的不確定點

論文文字不足以唯一決定實作，以下三點在動手前必須查 `microsoft/YOLaT-VectorGraphicsRecognition` 確認：

| # | 不確定點 | 兩種可能 | 暫定選擇與理由 |
|---|---|---|---|
| **U1** | **proposal 數量**：「10×10 網格」指 10×10 個 *cell*（→ 11×11=121 個角點）還是 10×10 個*角點* | (a) 角點 11×11 → 有效矩形 `C(11,2)² = 55² = ` **3025**／cluster<br>(b) 角點 10×10 → `C(10,2)² = 45² = ` **2025**／cluster | **暫定 (a) 3025**。cell 切割的說法較自然，且需在程式中設為可調常數 `GRID` |
| **U2** | **預測框 B̂ 的定義**：是 proposal 矩形本身，還是框內圖元的緊緻外接矩形 | (a) proposal 矩形（網格量化，較粗）<br>(b) 框內節點的 tight bbox（精確到圖元） | **暫定 (b) tight bbox**。論文稱「框本來就精確、無需回歸」，且 AP75 高達 94.65——網格量化的 (a) 難以達到此精度 |
| **U3** | **是否做 NMS** | 論文未明述，僅提「保留信心分數高於閾值者」 | **暫定做 class-wise NMS**。同一物件必然被多個 proposal 覆蓋，不去重會嚴重灌水誤報 |

> 三點都寫成設定開關，重現失敗時第一輪就切換它們做排查。

### 4.2 關鍵效能問題與解法：區域池化的積分圖優化

**問題**：每個 cluster 有約 3025 個 proposal，每個 proposal 都要對「框內所有節點」做 mean pooling。
樸素作法是 `for each proposal: for each node: if in box: accumulate`，複雜度 `O(P × N)`。
一張圖多個 cluster、batch 4，這會直接變成瓶頸（比 GNN 本身貴幾個數量級）。

**關鍵觀察**：proposal 的邊界**恰好落在網格線上**。
所以「節點是否在矩形內」等價於「節點所屬的 cell 是否落在該 cell 範圍內」——**沒有任何精度損失**。

**解法：2D 前綴和（積分圖）**

```
# 每個 cluster 內，對 10×10 的 cell 網格：
S[gx, gy] = Σ  h_i        # cell 內節點特徵和，shape (10, 10, D)
N[gx, gy] = count         # cell 內節點數,   shape (10, 10)

cumS = cumsum(cumsum(S, axis=0), axis=1)   # 2D 前綴和
cumN = cumsum(cumsum(N, axis=0), axis=1)

# 任意矩形 [x0,x1) × [y0,y1) 的區域和，O(1)：
rect_sum(cum, x0, y0, x1, y1)
    = cum[x1,y1] - cum[x0,y1] - cum[x1,y0] + cum[x0,y0]

region_mean = rect_sum(cumS, ...) / clamp(rect_sum(cumN, ...), min=1)
```

複雜度從 `O(P × N)` 降到 **`O(N + P)`**，且全部可向量化在 GPU 上執行。

**U2 的 tight bbox 無法用前綴和**（min/max 不可逆）。解法：算 per-cell 的 `min_x, min_y, max_x, max_y`
四張表，再對每個 proposal 在其 cell 範圍上做 masked min/max。cell 範圍上限 100，仍可完全向量化。

### 4.3 資料前處理：SVG → 圖 + proposal 快取

> **只跑一次，可跨所有標註比例與 seed 重複使用**（圖結構與標註無關）。

```
FUNCTION build_graph_cache(svg_path, gt_boxes):

    # ---- 步驟 1：圖元統一化 ----
    primitives = parse_svg(svg_path)          # line / circle / rect / polygon / path
    beziers = []
    FOR prim IN primitives:
        # 直線 → 控制點退化在端點連線上；圓/橢圓 → 4 段三次貝茲近似
        beziers += to_cubic_bezier(prim)      # 每段回傳 (p0, p1, p2, p3, rgb, width)

    # ---- 步驟 2：節點去重 ----
    # 同一座標的端點必須合併成同一節點，否則拓撲連通性會斷掉
    node_key(p) = (round(p.x, TOL), round(p.y, TOL))
    nodes = unique_by(node_key, all endpoints p0 and p3)
    node_feat[i] = [x, y, R, G, B, width]                      # 7 維

    # ---- 步驟 3：stroke-wise 邊 ----
    FOR (p0, p1, p2, p3) IN beziers:
        i, j = node_id[p0], node_id[p3]
        stroke_edges.append((i, j))
        stroke_attr.append([p1.x, p1.y, p2.x, p2.y])           # 離曲線控制點，4 維
        stroke_edges.append((j, i))                            # 無向 → 加反向邊
        stroke_attr.append([p2.x, p2.y, p1.x, p1.y])           # 反向邊控制點順序也要反

    # ---- 步驟 4：空間 cluster ----
    comps = connected_components(stroke_edges)                 # scipy.sparse.csgraph
    rects = [expand(bbox(c), EXPAND_LEN) FOR c IN comps]       # EXPAND_LEN 為超參
    # union-find 合併所有重疊的擴張矩形
    uf = UnionFind(len(comps))
    FOR (a, b) IN overlapping_pairs(rects):                    # 用 R-tree 或排序掃描線加速
        uf.union(a, b)
    cluster_id[i] = uf.find(comp_of[i])                        # 每個節點的 cluster

    # ---- 步驟 5：proposal 生成（逐 cluster）----
    FOR k IN clusters:
        cb = bbox(nodes in cluster k)
        xs = linspace(cb.x0, cb.x1, GRID + 1)                  # 11 條網格線（U1）
        ys = linspace(cb.y0, cb.y1, GRID + 1)
        # 節點的 cell 索引，供 §4.2 積分圖使用
        cell[i] = (digitize(x_i, xs), digitize(y_i, ys))
        FOR (a, b) IN combinations(range(GRID+1), 2):          # x 方向兩條線
            FOR (c, d) IN combinations(range(GRID+1), 2):      # y 方向兩條線
                rect = (xs[a], ys[c], xs[b], ys[d])
                IF area(rect) > SIZE_THRESH * area(cb): CONTINUE   # 尺寸過濾
                IF count_nodes_in(rect) == 0:            CONTINUE   # 空框直接丟
                proposals.append((k, a, c, b, d))

    # ---- 步驟 6：標籤指派 ----
    # 注意：此處存的是「該圖完整標註下的標籤」。
    # 標註比例的控制在 §4.4 用「整張圖取或不取」實現，不在這裡篩。
    FOR p IN proposals:
        ious = IoU(pred_box(p), gt_boxes)                      # pred_box 依 U2 決定
        label[p] = argmax(ious) IF max(ious) >= ALPHA ELSE BACKGROUND

    SAVE {node_feat, stroke_edges, stroke_attr, cluster_id,
          cell, proposals, label, gt_boxes} TO cache
```

**步驟 2 的節點去重是最容易出錯的地方**：SVG 中相鄰圖元的端點常有浮點誤差（如 `100.0` vs `99.99998`）。
若不用容差合併，stroke 圖會碎成大量孤立分量，連通性先驗直接失效——**而這正是 H1 主張的來源**。
`TOL` 必須實測調整並記錄。

### 4.4 標註子集裝置

```
FUNCTION make_split(all_image_ids, ratio, seed):
    rng = Random(seed)
    shuffled = rng.shuffle(copy(all_image_ids))
    n = max(1, round(len(all_image_ids) * ratio))
    RETURN {"train": shuffled[:n], "ratio": ratio, "seed": seed}

# 產生一次，向量組與點陣組共讀同一份（§3.3 對照控制 1）
FOR ratio IN [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
    FOR seed IN [0, 1, 2]:
        SAVE make_split(train_ids, ratio, seed) TO f"splits/{ratio}_{seed}.json"
```

**巢狀性質**：因為先 shuffle 再取前 n 個，**同一 seed 下小比例的集合是大比例的子集**。
這讓曲線更平滑、可歸因（不是換了一批完全不同的圖），建議保留此性質並在論文中說明。

### 4.5 模型 forward

```
FUNCTION forward(batch):
    # batch 內多張圖已由 PyG 合併，node 索引與 cluster 索引皆已 offset

    h = Linear(7 -> 64)(batch.node_feat)     # t = 0
    z = h.clone()
    h_steps = [h]
    z_steps = [z]

    # ---- Stroke 流：2 層 EdgeConv 式訊息傳遞 ----
    FOR t IN [0, 1]:
        # msg_ij = f_s( concat(h_i, h_j - h_i, edge_attr_ij) )   輸入 64+64+4 = 132
        src, dst = stroke_edges
        msg = f_s[t](concat(h[dst], h[src] - h[dst], stroke_attr))
        agg = scatter_mean(msg, dst, dim_size=num_nodes)
        h = f_l[t](h) + agg                                     # 殘差式
        h_steps.append(h)

    # ---- Position 流：僅末層聚合，避免 over-smoothing ----
    FOR t IN [0, 1]:
        z = f_p[t](z)                        # 逐節點變換，尚未跨節點聚合
        IF t == LAST:
            # §4.2 的 mean-pool 優化：cluster 內取均值後廣播回節點
            cmean = scatter_mean(z, cluster_id, dim_size=num_clusters)
            z = cmean[cluster_id]
        z_steps.append(z)

    # ---- 區域池化（§4.2 積分圖，O(N + P)）----
    region_feats = []
    FOR feat IN h_steps + z_steps:           # 共 3 + 3 = 6 組
        cumS, cumN = build_integral(feat, cell, cluster_id)
        region_feats.append(rect_mean(cumS, cumN, proposals))
    r = concat(region_feats, dim=-1)         # 6 × 64 = 384 維

    logits = MLP(384 -> 512 -> 256 -> C+1)(r)
    RETURN logits
```

**`h[src] - h[dst]` 的方向必須與 `stroke_attr` 的控制點順序一致**（見 §4.3 步驟 3 的反向邊處理），
否則邊屬性會與幾何方向錯配。這種錯誤不會報錯，只會讓 mAP 悄悄掉幾個點。

### 4.6 訓練迴圈：固定 iteration（§3.5 問題 7）

```
FUNCTION train(split, TOTAL_ITERS = 60000):
    dataset = [cache[i] FOR i IN split["train"]]
    loader  = InfiniteLoader(dataset, batch_size=4, shuffle=True)   # 循環取樣，不用 epoch
    opt = Adam(model.parameters(), lr=2.5e-4)
    sched = CosineAnnealing(opt, T_max=TOTAL_ITERS)

    FOR it IN range(TOTAL_ITERS):            # 所有標註比例共用同一個 TOTAL_ITERS
        batch = next(loader)
        logits = model(batch)
        loss = cross_entropy(logits, batch.label)      # 唯一損失項
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()

        IF it % EVAL_EVERY == 0:
            record(evaluate(model, val_set))
```

**`InfiniteLoader` 是把「標註量」與「訓練量」解耦的關鍵**：1% 組（5 張圖）會把這 5 張圖
重複看 48000 次，100% 組把 500 張各看 480 次，兩者梯度更新次數完全相同。
曲線量到的才是純粹的標註量效應。

> **副作用（須記錄）**：1% 組必然嚴重過擬合。這是**預期且正確**的現象——
> 它正是「標註不足」的表現形式，不是 bug，不要用 early stopping 掩蓋。
> 但驗證集必須固定且全標註，否則低比例組連模型選擇都做不了。

### 4.7 評估

```
FUNCTION evaluate(model, image_ids):
    all_dets = []
    FOR img IN image_ids:
        logits = model(cache[img])
        scores, classes = softmax(logits).max(dim=-1)
        keep = (classes != BACKGROUND) & (scores > SCORE_THRESH)
        boxes = pred_box(proposals[keep])                  # 依 U2
        boxes, scores, classes = class_wise_nms(boxes, scores, classes, IOU_NMS)   # 依 U3
        all_dets.append(...)
    RETURN coco_eval(all_dets, gts)        # AP50, AP75, mAP@[.5:.95]
```

### 4.8 點陣對照組

```
FUNCTION build_raster_dataset(split, RES = 1024):
    FOR img IN split["train"]:
        png = rasterize_svg(img, width=RES, height=RES)     # cairosvg / resvg
        scale = RES / svg_viewbox_width(img)
        boxes = gt_boxes(img) * scale                        # GT 必須同步縮放
        SAVE yolo_format(png, boxes)
    # 用同一份 split 檔、同一組 seed；YOLOv8-s，from scratch（不載預訓練權重）
    # 訓練預算同樣以 iteration 對齊，不用 epoch
```

**必須關閉預訓練權重**。否則對照組帶著 ImageNet 的外部知識，
在低標註區間會有不對等優勢——那量到的是「預訓練 vs 無預訓練」，不是「點陣 vs 向量」。

### 4.9 R1-2 新發現的實作問題（補充 §3.5）

| # | 問題 | 對策 |
|---|---|---|
| 8 | **節點去重容差 `TOL`**：浮點誤差會讓 stroke 圖碎成孤立分量，直接摧毀連通性先驗 | 以容差合併；統計「平均連通分量大小」當健檢指標，異常小即代表 TOL 太嚴 |
| 9 | **反向邊的控制點順序**必須跟著反轉，否則邊屬性與幾何方向錯配 | 見 §4.3 步驟 3；寫單元測試驗證 |
| 10 | **背景類極度不平衡**：3025 proposals/cluster 絕大多數是背景 | 先靠尺寸閾值 + 空框過濾；仍不平衡則加背景負採樣或 focal loss（但**偏離原始設定，需記錄**） |
| 11 | **驗證集必須固定且全標註**，否則低比例組無法做模型選擇 | val / test 一律 100% 標註，只有 train 受 ratio 影響 |
| 12 | **1% 組過擬合是預期現象**，不可用 early stopping 掩蓋 | 固定 iteration 跑完，報最終值與最佳值兩者 |
| 13 | 對照組**不可載入預訓練權重** | YOLOv8 明確設 `pretrained=False` |
| 14 | 巢狀 split 性質需在論文說明 | 小比例是大比例的子集，使曲線可歸因 |

## 5. 實作模型（R1-3）

### 5.1 資料集勘查結果（已完成，2026-09-07）

**取得狀態**：SESYD 原站（`mathieu.delalandre.free.fr`）**存活**，但**僅支援 HTTP，不支援 HTTPS**
（強制 HTTPS 的工具會連線失敗，須用 `curl` 明示 http://）。
Floorplans 10 包 + models 共 42 MB，已下載並解壓至 `data/sesyd/`。授權：YOLaT 程式碼為 MIT。

**實測統計（1000 張 floorplan 全量）**

| 項目 | 數值 |
|---|---|
| SVG / XML 檔 | 1000 / 1004（4 個多餘 XML 待對帳） |
| 物件總數 / 類別數 | **28,065 / 16** ✅ 與論文完全一致 |
| 每檔物件數 | 中位數 26 |
| **圖元總數** | **258,300** |
| 每檔圖元數 | min 158 / **中位數 249** / max 372 |
| 圖元組成 | `<line>` 245,358 (95.0%)、`<path>` 7,252 (2.8%)、`<circle>` 5,690 (2.2%) |
| 畫布尺寸 | 中位數 3047 × 2490；長寬比 0.90–2.37 |
| 類別分布 | table1 5476 … door2 500（**最大/最小 ≈ 11:1**） |

### 5.2 五個影響研究設計的發現

#### 🔴 發現 1（最嚴重）：SVG 只含符號，牆體在點陣背景層

每個 SVG 的結構是：

```xml
<svg width="6775.2" height="2858.4">
  <image x="0" y="0" width="6775.2" height="2858.4" xlink:href="groundfilled-01.png"/>   <!-- 牆體/房間：點陣 -->
  <g style="stroke-linecap:round">
    <line .../>   <!-- 符號筆劃：向量，僅此部分 -->
  </g>
</svg>
```

**1000/1000 個 SVG 皆含 `<image>` 點陣背景層。**

這造成 YOLaT 的 H1 證據存在**結構性偏差**：

| 模型 | 實際看到的內容 |
|---|---|
| YOLaT（向量） | **只有符號筆劃**（約 250 個圖元），牆體完全不可見 |
| 點陣 baseline | 渲染後的 TIFF，**符號 + 牆體雜訊全部都有** |

→ 向量表示在此資料集上不只是「同樣資訊的不同編碼」，而是**已經做過背景減除的乾淨版本**。
「向量 vs 點陣」的比較，有一部分其實是「純符號 vs 符號+雜訊」。

**C1 的交互作用**：C1 本就要求丟棄 `<image>`（否則管線含點陣輸入），
但丟棄後反而給了受試模型一項對照組沒有的資訊優勢。

**已決策（v1.1）：採方案丙** —— 甲、乙兩組都跑，量化此偏差的大小。此偏差本身即為可發表的獨立發現。

**對策方案**：

| 方案 | 作法 | 性質 |
|---|---|---|
| **甲** ✅ | 對照組只渲染**向量層**（不含背景 PNG），兩組資訊量對等 | 公平的 H1 檢定 |
| **乙** ✅ | 維持 YOLaT 原設定（對照組看完整渲染圖） | 與文獻可比，但偏袒向量組 |
| **丙 = 甲+乙（採用）** | 兩組都跑 | 成本 +1 組曲線，可量化偏差大小 |

#### 🔴 發現 2：只有 10 種背景版面

10 個子資料夾各有 **1 張**共用背景 PNG，其中 100 個 SVG 全部引用同一張。
即 **1000 張圖 = 10 種版面 × 100 種符號擺放**。

影響：
- 資料多樣性遠低於「1000 張」的表面數字，是天花板效應（§1.5）的直接成因之一
- **切分洩漏**：隨機 split 會讓同一版面同時出現在 train 與 test。
  對向量組無影響（看不到背景），但**對點陣對照組是實質洩漏**
- 標註效率曲線在 1% ≈ 5 張時，抽到的版面可能高度重複 → 變異數更大

**已決策（v1.1）：採依版面分組切分（GroupKFold by background）**，train / val / test 不共用版面。
10 個版面依 5 : 2 : 3 分配（500 / 200 / 300 張）。注意僅 10 組，分組選擇有限，須固定分組方案並記錄。

#### 🟠 發現 3：光柵化 sub-pixel 問題已是實測事實

線寬與畫布的比值實測：`6.48 / 6775 = 0.00096`、`5.17 / 3047 = 0.0017`。

| 渲染寬度 | 線寬（像素） |
|---|---|
| 640 px | **0.61 – 1.09 px** ← 次像素，嚴重失真 |
| 1024 px | 0.98 – 1.74 px |
| 2048 px | 1.96 – 3.48 px |

→ 在常用的 640 px 下，**符號線條是次像素寬度**，會因反鋸齒而淡化甚至消失。
且畫布長寬比 0.90–2.37，縮放至正方形必然變形。

此項在 §1.5 列為**暫緩**，此處僅補上實測數字備查，不改變暫緩狀態。
若採發現 1 的方案甲／丙，建議至少把渲染寬度定在 **≥ 1024 px** 並保持長寬比。

#### 🟡 發現 4：`--in_channels 5` 與論文的 7 維不符（新增 U4）

官方訓練指令為 `--in_channels 5`，論文則寫 7 維 `(x, y, R, G, B, w)`。

實測佐證：**全資料集 `stroke` 皆為 `black`**（顏色零資訊），
而 `stroke-width` 有多個相異值（5.173549、4.05、4.758、6.48、7.137…），**與符號類別／縮放相關，有資訊量**。

| 假說 | 5 維組成 | 評估 |
|---|---|---|
| H-a | `(x, y, R, G, B)`，**捨棄線寬** | 與論文差異最小，但捨棄了此資料集中唯一有資訊的屬性 |
| H-b | `(x, y, w, ?, ?)` | 需查碼確認 |

→ **U4 必須查官方碼確認。** 若為 H-a，**納入 stroke-width 有機會超越原始結果**——可列為 R2 的低成本改進項。

#### 🟢 發現 5：圖元語法範圍很窄，解析器工作量小

只需支援三種：`<line>`、`<circle>`、`<path>` 且 path 僅含**橢圓弧**（`m` + `a` 命令）。
未出現 `C/Q/S/T` 曲線命令，也沒有 `rect`／`polygon`／`polyline`。

→ §4.3 的「圖元統一化」實作範圍明確：**直線、圓、橢圓弧 → 三次貝茲**，皆有標準轉換公式。
弧轉貝茲需注意大於 90° 的弧要分段（每段 ≤ 90°）以控制誤差。

### 5.3 對既有章節的影響

| 章節 | 影響 |
|---|---|
| §4.1 | 新增 **U4**（in_channels 5 vs 7） |
| §4.3 步驟 1 | 解析器範圍收斂為 line / circle / arc；**須明確丟棄 `<image>` 元素**（C1 要求） |
| §4.8 | 對照組渲染需決策發現 1 的甲／乙／丙；渲染寬度建議 ≥1024 且保持長寬比 |
| §4.4 | split 建議改為依背景版面分組 |
| §1.5 | 天花板效應新增成因：僅 10 種版面 |

### 5.4 環境建置（已完成）

**本機可用 Python：3.13.0b1（numpy 載入失敗，無法使用）、3.10.6、3.9。無 3.11/3.12、無 conda/uv。**
→ 改用 **Python 3.10.6** 建 venv 於 `.venv/`。計劃書先前記載的 3.11 已作廢。

| 套件 | 版本 | 備註 |
|---|---|---|
| numpy | 2.1.3 | 系統 Python 3.13 下的 numpy 2.1.0 為損壞安裝 |
| scipy | 1.15.3 | 連通分量 |
| pillow | 12.3.0 | **自行光柵化用**（見下） |
| torch | cu124 wheel | 驅動為 CUDA 12.7，向下相容 |
| torch-geometric | 2.x | 不需獨立 `torch_scatter` |

**光柵化方案變更**：放棄 `cairosvg`（Windows 需 GTK 系統相依）。
改為**由已解析的貝茲段自行以 PIL 渲染**。優點：
1. 無額外系統相依；
2. 解析度與反鋸齒完全可控；
3. **點陣組與向量組來自同一份解析幾何**，這正是方案甲（§5.2 發現 1）所需的對等性。

### 5.5 實作進度與實測驗證

| 模組 | 檔案 | 狀態 |
|---|---|---|
| SVG 解析 → 三次貝茲 | `src/yolat/svg_parse.py` | ✅ 完成並驗證 |
| 多重圖建構（節點／stroke 邊／cluster） | `src/yolat/graph.py` | ✅ 完成並驗證 |
| Proposal 枚舉 + 積分圖區域池化 | `src/yolat/proposals.py` | ✅ 完成並驗證 |
| 資料快取與 GT 載入 | `src/yolat/dataset.py` | 進行中 |
| 雙流 GNN 模型 | `src/yolat/model.py` | 待 torch 安裝完成 |
| 訓練／評估 | `src/train.py`, `src/eval.py` | 未開始 |

#### 解析器驗證

| 檢查 | 結果 |
|---|---|
| 圓的四段貝茲近似 | 最大徑向誤差 **0.027% of r**（0.01 px，遠小於 6.48 px 線寬） |
| 橢圓弧端點 | 與解析解一致；>90° 自動分段 |
| 全量解析 | **1000/1000 檔成功**，278,674 個貝茲段，16.3 ms/檔 |

#### 建圖超參（實測選定）

| 超參 | 選值 | 依據 |
|---|---|---|
| `dedup_tol` | **0.05 px** | 掃描 0.0001–2.0：0.0001–0.05 結果完全相同（節點 358、分量 114.7、平均分量大小 3.13）→ SESYD 端點座標本就精確吻合，容差不敏感。取 0.05 為安全值 |
| `expand_ratio` | **0.002** | 掃描 0.0–0.05：此值下 cluster 數 27.5 ≈ GT 物件數 28.5（**比值 0.96**），達成「一個 cluster 對一個符號」 |

> 附帶發現：平均連通分量大小僅 **3.13**，每個符號約由 **4 個** 不相連的分量構成
> （例如桌子由數個分離矩形組成）。因此 cluster 合併步驟並非可有可無，而是還原符號完整性的必要步驟。

#### 🟢 U2 已由實測解決：採 tight bbox

在 8 張圖上量測 proposal 的**召回上限**（任一 proposal 與 GT 的最佳 IoU）：

| 框定義 | IoU ≥ 0.50 | IoU ≥ 0.75 |
|---|---|---|
| U2a rect-box（網格矩形） | 99.5% | 97.0% |
| **U2b tight-box（框內節點外接框）** | **99.5%** | **98.8%** |

→ **U2 確定為 tight bbox**。與論文 AP75 = 94.65 相符；rect-box 在高 IoU 區間會成為瓶頸。

**其他實測數字**

| 項目 | 數值 |
|---|---|
| 過濾後 proposal 數 | **19,403 / 圖**（理論 27.8 cluster × 3025 = 84k，經 `min_nodes`＋尺寸過濾降至 23%） |
| 前處理速度 | 0.08 s/圖（解析＋建圖＋proposal＋IoU） |
| **召回上限 IoU≥0.5** | **99.5%** → 足以支撐論文 98.83 AP50，實作路徑已驗證可行 |

### 5.6 完整管線已跑通（2026-09-08）

| 模組 | 檔案 | 狀態 |
|---|---|---|
| SVG 解析 → 三次貝茲 | `src/yolat/svg_parse.py` | ✅ |
| 多重圖建構 | `src/yolat/graph.py` | ✅ |
| Proposal + 積分圖池化 | `src/yolat/proposals.py` | ✅ |
| GT 載入、前處理快取 | `src/yolat/dataset.py` | ✅ |
| 分組切分 + 標註比例子集 | `src/yolat/splits.py` | ✅ |
| 雙流 GNN | `src/yolat/model.py` | ✅ |
| 批次化 + InfiniteLoader | `src/yolat/torch_data.py` | ✅ |
| NMS + COCO-style AP | `src/yolat/evaluate.py` | ✅ 自建，不依賴 pycocotools（Windows 無維護版 wheel） |
| 前處理腳本 | `scripts/prepare_data.py` | ✅ |
| 訓練腳本 | `scripts/train.py` | ✅ |

#### 全量前處理結果

| 項目 | 數值 |
|---|---|
| 檔案 | 1000 |
| 平均節點 / 邊 | 348.5 / 557.3 |
| 平均 cluster | 27.7 |
| 平均 proposal | 18,984 |
| 平均正樣本 | 468.6（**正樣本率 2.47%**，約 1:40） |
| 平均 GT 物件 | **28.065 → 總計 28,065，與論文完全一致** ✅ |
| 耗時 / 快取大小 | 123.7 s / 251 MB |

#### 分組切分（發現 2 的決策落實）

10 個版面 → train 5 / val 2 / test 3 = **500 / 200 / 300 張**，版面不跨切分。
產生 40 個 split 檔（8 比例 × 5 seed），`n_train` = 2 / 5 / 10 / 25 / 50 / 125 / 250 / 500。
向量組與點陣組讀同一批檔案。

#### ⚠️ 新增 U5、U6：參數量與 FLOPs 與論文不符

| 項目 | 本實作 | 論文 |
|---|---|---|
| 參數量 | **367,185** | 1.6 M |
| 推估 FLOPs / 圖 | ~55 GFLOPs | **1.5 GFLOPs** |

依論文 §3.2 規格（hidden 64、2 層、MLP 512→256）推算即得 367 k，缺口在論文側未說明。
FLOPs 的 36 倍落差反推出**論文每張圖僅約 2,200 個 proposal**，而本實作為 18,984。

**已驗證的假說與否證**：若「頂點配對」的頂點指**圖節點**而非網格角點，proposal 數確實落在該量級，
但實測召回不足：

| Proposal 方案 | 數量/圖 | IoU≥.50 | IoU≥.75 |
|---|---|---|---|
| **網格 10×10（採用）** | 18,984 | **99.5%** | **98.8%** |
| 節點配對 step=1 | 3,581 | 89.2% | 73.6% |
| 節點配對 + tighten | 3,581 | 89.2% | 73.6% |
| 節點配對 step=2 | 871 | 70.2% | 49.8% |

→ 節點配對的召回上限 89.2%，**無法支撐論文的 98.83 AP50**，故否證。
**維持網格方案**；U5／U6 列為待查（需官方碼），不阻擋重現，但**論文中不可直接引用 YOLaT 的效率數字作為本實作的效率宣稱**。

#### 訓練管線實測

| 項目 | 數值 |
|---|---|
| 前向+反向（batch 4, `neg_per_pos=4`） | **22 ms**，峰值 VRAM **107 MB** |
| 前向+反向（全部 proposal，不採樣） | 412 ms，峰值 VRAM 639 MB |
| 訓練吞吐 | **約 20 iter/s** |
| 單次 run（20k iters + 8 次評估） | **約 20 分鐘** |

> **算力估算大幅下修**：§2.3 原估 2–4 GPU-hr/run。實測後單 run 約 **0.35 GPU-hr**。
> 40 個 run 的完整向量組曲線約 **13 GPU-hr → 本機 RTX 4060 一天內可跑完，向量組不需租用算力**。
> 租用需求僅剩點陣對照組（YOLOv8）與可能的 R2。

#### 負樣本採樣（偏離原始設定，須記錄）

正樣本率僅 2.47%，且 18,984 個 proposal 會主導激活記憶體。
實作 `neg_per_pos`（預設 4.0）：保留全部正樣本 + 4 倍數量的隨機背景 proposal。
評估時不採樣。**此為對原始設定的偏離，所有 run 均記錄此參數**（§4.9 問題 10）。

### 5.7 🔴 重現失敗：分組切分下泛化崩潰（2026-09-08）

**首次完整重現 run**（100% 標註、20k iters、分組切分、絕對座標）：

| iter | train loss | AP50 | AP75 | mAP |
|---|---|---|---|---|
| 2500 | 0.364 | 12.4 | 11.3 | 9.8 |
| **5000** | 0.121 | **19.7** | 15.4 | **13.9** ← 峰值 |
| 10000 | 0.068 | 14.7 | 13.0 | 11.4 |
| 20000 | 0.045 | 14.7 | 13.1 | 11.7 |

**論文 mAP 90.59，本次 13.9。未達 §3.6 出口條件。**

#### 診斷：不是 bug，是泛化崩潰

以訓練好的模型同時在 train 與 val 上評估（各 60 張）：

| 評估集 | AP50 | AP75 | mAP | proposal 分類正確率 | 精確率 | 召回率 |
|---|---|---|---|---|---|---|
| **train** | **96.53** | 59.49 | 60.31 | 97.87% | 50.3% | 89.5% |
| **val** | 9.29 | 7.78 | 5.84 | 91.40% | 11.5% | 28.4% |

→ 管線在**見過的資料**上可達 96.5 AP50，證明解析、建圖、proposal、池化、損失、NMS、AP 計算全部正確。
**問題在於模型無法泛化到未見過的版面。**

#### 主假說：絕對座標 + 版面分組切分

節點特徵含**絕對座標** `(x/scale, y/scale)`。SESYD 同一版面內，傢俱位置分布高度固定
（房間位置不變），模型可直接記憶「位置 → 類別」。
在**隨機切分**下訓練與測試共用全部 10 個版面，此捷徑仍然有效；
改為**依版面分組切分**後捷徑失效，效果崩潰。

> **若此假說成立，其意涵重大**：YOLaT 論文回報的 90.59 mAP 是在
> **train / test 共用同樣 10 種版面**的隨機切分上量測的，
> 亦即該數字包含**版面洩漏**，並非真正的跨版面泛化能力。
> 這將是本研究的一項獨立發現，且直接影響 §1.2 對 H1 的證據評估。

#### 對照實驗結果（已完成）

| 實驗 | 切分 | 座標 | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| **C（原重現）** | 分組 | 絕對 | 14.71 | 13.06 | 11.67 |
| **A** | **隨機（洩漏）** | 絕對 | **97.27** | 58.24 | 57.29 |
| **B** | 分組 | **cluster 相對** | 47.12 | 27.88 | 27.08 |
| 論文 | 隨機 | — | 98.83 | 94.65 | 90.59 |

**兩個假說皆證實：**

1. **版面洩漏效應已量化。** 僅將分組切分換成隨機切分（其餘完全相同），
   AP50 從 **14.71 → 97.27**，與論文的 98.83 幾乎一致。
   → **論文的 AP50 可在隨機切分下重現，但在無洩漏的分組切分下崩潰至 14.71。**
2. **絕對座標是主要捷徑來源。** 在分組切分下改用 cluster 相對座標，
   AP50 從 **14.71 → 47.12（3.2 倍）**，mAP 11.67 → 27.08。

#### 🟠 尚未解決的第二個落差：AP75

即使在隨機切分下（實驗 A），AP75 僅 **58.24**，論文為 **94.65**；mAP 57.29 vs 90.59。
AP50 已重現而 AP75 未重現 → 問題在**定位精度**，不在分類。

**已排除**：proposal 召回上限在 IoU≥0.75 為 98.8%（§5.5），框本身夠精確。

**主要嫌疑：標籤指派閾值。** 目前所有 IoU ≥ **0.5** 的 proposal 都指派為同一正類別，
模型因此**沒有任何訊號**去偏好 IoU 0.95 的框而非 IoU 0.55 的框。
一個符號平均對應 17 個正 proposal（§5.6），NMS 後留下的未必是最準的那個。

**可驗證的對策**（依成本排序）：

| 對策 | 作法 | 成本 |
|---|---|---|
| 提高標籤指派閾值 α | α 由 0.5 提高至 0.7 / 0.8；快取已存 `prop_box` 與 `gt_boxes`，**可於載入時重算標籤，無須重跑前處理** | 低 |
| IoU 感知評分 | 分類分數乘以預測 IoU，或加一個 IoU 迴歸頭 | 中，且偏離原始設定 |
| 減少 proposal 密度 | 對應 U6 的每圖約 2,200 個，稀疏 proposal 使最佳框更易勝出 | 中 |

#### 對計劃的確定影響

| 章節 | 修訂 |
|---|---|
| §1.2 H1 證據 | **YOLaT 在 SESYD floorplan 的效果優勢有相當部分來自版面洩漏。** H1 的文獻支持強度須下修，且此為本研究可獨立發表的發現 |
| §2.7.1「R1 不改內層」 | **須破例**。絕對座標在分組切分下使模型無法學習，相對座標為必要修正而非改進 |
| §3.6 出口條件 | 「重現 90.59 ± 1.5」僅在隨機切分下才是合理目標。分組切分須另訂基準（目前最佳 47.12 AP50） |
| §6 實驗設計 | 標註效率曲線應在**分組切分**下進行；隨機切分僅作為與文獻對照的附加組 |


### 5.8 決策與 AP75 修正（v1.6）

**已決策**

| 項目 | 決定 |
|---|---|
| 出口條件（§3.6） | **先把 AP75 修到接近論文（隨機切分下追 90.59），再回頭訂分組切分的基準** |
| 洩漏發現的定位 | **先擱置**。不現在決定論文重心，優先把 H2 標註效率曲線跑出來 |

> 註：兩項決策的執行順序為 **修 AP75 → 確認可重現 → 跑 H2 曲線**。
> 洩漏發現的事實與數據已完整記錄於 §5.7，隨時可回頭啟用。

#### AP75 根因與修正

**根因不只是「α 太低」，而是中間地帶的處理方式。**
原設定將 IoU ≥ 0.5 的 proposal 全部指派為同一正類別，
模型因此對 IoU 0.55 與 IoU 0.95 的框輸出相同目標，NMS 等同隨機挑選。

**實測 IoU 分布**（20 張圖，390,946 個 proposal）：

| IoU 門檻 | proposal 數 | 佔比 |
|---|---|---|
| ≥ 0.3 | 27,442 | 7.02% |
| ≥ 0.5 | 9,888 | 2.53% |
| ≥ 0.7 | 3,251 | 0.83% |
| ≥ 0.8 | 1,591 | 0.41% |
| ≥ 0.9 | 743 | 0.19% |

**每個 GT 物件對應的正 proposal 數**：

| pos_iou | neg_iou | 正樣本/圖 | 忽略/圖 | 負樣本/圖 | **正樣本 / GT** |
|---|---|---|---|---|---|
| 0.5（原設定） | 0.5 | 494 | 0 | 19,053 | **17.5** |
| 0.7 | 0.4 | 163 | 704 | 18,681 | 5.7 |
| 0.8 | 0.4 | 80 | 787 | 18,681 | 2.8 |
| 0.9 | 0.4 | 37 | 829 | 18,681 | 1.3 |

**已實作：positive / ignore / negative 三分方案**（`dataset.relabel`）

- IoU ≥ `pos_iou` → 正樣本
- IoU < `neg_iou` → 背景
- 兩者之間 → **`-1`，以 `ignore_index` 排除於損失之外**（RPN 慣例），不再誤標為背景

快取已加存 `prop_iou`、`prop_gt`（float16 / int16），
**閾值可任意掃描而無須重跑前處理**。評估階段不重標、不採樣。

### 5.9 追平論文的除錯歷程（v1.7）

以隨機切分（對齊論文設定）逐步逼近 90.59 mAP。

#### 步驟 1：標籤指派閾值掃描（10k iters）

| pos_iou | AP50 | AP75 | mAP |
|---|---|---|---|
| 0.5（原） | 96.06 | 60.44 | 59.36 |
| 0.7 | 87.27 | 63.62 | 57.66 |
| **0.8** | 84.71 | **68.72** | **62.25** |
| 0.9 | 67.43 | 56.55 | 53.12 |

有效但不足（mAP +2.9）。且 AP50 下降，出現召回與定位的拉鋸 → **瓶頸不在標籤指派**。

#### 步驟 2：🔴 找到真 bug —— 預測框少算線寬

SESYD 的 GT 框涵蓋**畫出來的墨跡**，本實作的 tight box 卻用**幾何中心線**，每邊少了半個線寬。

| 框定義 | 平均最佳 IoU | ≥0.90 | **mAP 天花板** |
|---|---|---|---|
| 端點（原） | 0.8985 | 68.4% | **84.7%** |
| ＋曲線外擴 | 0.9011 | 70.2% | ~85% |
| **＋線寬一半（採用）** | **0.9493** | **92.3%** | **95.2%** |

> **原實作的 mAP 幾何天花板僅 84.7%，低於論文的 90.59。**
> 亦即在修正前，**無論分類器多完美都不可能重現論文數字**。
> 曲線外擴僅貢獻 +0.3%，可忽略；**線寬才是主因**。

修正：`proposals.enumerate_proposals(stroke_pad=True)`，節點座標各向外擴 `width/2`。

#### 步驟 3：修正後結果（20k iters，隨機切分）

| 設定 | AP50 | AP75 | mAP |
|---|---|---|---|
| 修正前 pos_iou 0.5 | 97.27 | 58.24 | 57.29 |
| 修正後 pos_iou 0.5 | 97.62 | 59.57 | 61.18 |
| **修正後 pos_iou 0.8** | 90.56 | **71.52** | **68.39** |
| 論文 | 98.83 | 94.65 | 90.59 |
| 幾何天花板 | — | — | 95.2 |

mAP 57.3 → **68.4**，仍差 22 分。

#### 步驟 4：模型卡在「cluster 框」的天花板

只用「涵蓋整個 cluster」的那一個 proposal 計算上限：

| 指標 | 數值 |
|---|---|
| 每 GT 的 IoU 中位數 | 0.9592 |
| 每 GT 的 IoU 平均 | 0.7810（**雙峰**：約 20% 的 GT 匹配極差） |
| **cluster 框的 mAP 天花板** | **70.7%** |
| 實測模型 mAP | **68.39** |

→ **模型實際上只做到「分類每個 cluster」，完全沒有利用 3025 個矩形去細修框。**

#### 步驟 5：`expand_ratio` 的最佳化目標選錯了

原本以「cluster 數 ≈ GT 數」為目標選了 0.002。應以**框品質**為目標：

| expand | cluster 數 | cl/GT | **cluster 框天花板** | 全 proposal 天花板 |
|---|---|---|---|---|
| **0.0（不合併）** | 41.5 | 1.46 | **86.8%** | 90.9% |
| **0.0005** | 31.3 | 1.10 | 80.6% | **96.6%** |
| 0.001 | 30.2 | 1.06 | 76.6% | 96.0% |
| 0.002（原選值） | 27.5 | 0.96 | 70.7% | 95.2% |
| 0.008 | 16.8 | 0.59 | 38.0% | 89.6% |

> **修正 §5.5 的一項錯誤結論**：先前記載「cluster 合併是還原符號完整性的必要步驟」。
> 實測顯示**合併反而傷害框品質**——擴張會把鄰近符號拉進同一 cluster。
> 兩個目標互相衝突：`0.0` 對 cluster 框最好，`0.0005` 對全 proposal 集最好。
> 由於模型目前運作在 cluster 框層級，兩者需以訓練實測仲裁（進行中）。

#### 步驟 6：`expand_ratio` 訓練實測仲裁

隨機切分、pos_iou 0.8、20k iters：

| expand | AP50 | AP75 | mAP (20k) | mAP (10k 峰值) |
|---|---|---|---|---|
| 0.002（原） | 90.56 | 71.52 | 68.39 | — |
| 0.0（不合併） | 92.08 | 70.75 | 70.67 | 71.33 |
| **0.0005（採用）** | **97.96** | **74.56** | **73.64** | **75.74** |

→ **`expand_ratio` 定為 0.0005**。與 §5.5 的 0.002 相比 mAP +7.4。
（10k 後略有下滑，屬輕微過擬合；正式實驗應以驗證集選點。）

#### 步驟 7：框投票後處理（不改模型，已否決）

以既有 checkpoint 重評，測試分數加權平均的框投票：

| 後處理 | AP50 | AP75 | mAP |
|---|---|---|---|
| **plain NMS（維持）** | 97.96 | 74.56 | **73.64** |
| 框投票 @0.5 | 98.95 | 45.96 | 54.38 |
| 框投票 @0.6 | 98.41 | 67.37 | 64.05 |
| 框投票 @0.7 | 97.96 | 76.05 | 67.10 |
| 框投票 @0.8 | 97.92 | **79.38** | 70.62 |

→ 框投票能提升 AP75（74.56 → 79.38），但**傷害整體 mAP**（73.64 → 70.62）：
平均會把已經精準的框拉鬆，在 IoU ≥ 0.85 的高門檻區間淨損失。**否決，維持 plain NMS。**

#### 重現進度總表

| 階段 | 修正內容 | mAP |
|---|---|---|
| 起點（分組切分） | — | 11.67 |
| 改隨機切分 | 揭露版面洩漏 | 57.29 |
| 修線寬 bug | 框加 `width/2` | 61.18 |
| 標籤三分 | pos_iou 0.8 + ignore band | 68.39 |
| **expand_ratio 0.0005** | cluster 品質最佳化 | **75.74** |
| — | 框投票（否決） | 70.62 |
| **論文** | | **90.59** |
| 幾何天花板 | | 96.6 |

**目前差距 15 分。** 已排除：幾何天花板、標籤指派、cluster 劃分、後處理。

#### 剩餘唯一的主要槓桿：定位品質訊號

所有 IoU ≥ α 的 proposal 共用相同標籤，模型**沒有任何訊號**去偏好最準的框，
分數只反映分類信心。標準解法為 **IoU 預測頭**（分數 = 分類信心 × 預測 IoU，
如 IoU-Net / FCOS centerness）。

> ⚠️ **此為架構層新增，偏離 YOLaT「僅 cross-entropy、無迴歸」的原始設定，需決策。**

> **重要**：H2（標註效率曲線）**不要求與論文完全一致的絕對數值**。
> 只要向量組與點陣組在同一實作下受到對等處理，曲線比較即成立。
> mAP 75.7 已是可用的工作基線。

#### 其他待查項

| 假說 | 說明 | 成本 |
|---|---|---|
| **缺乏定位品質訊號** | 所有 IoU ≥ α 的 proposal 標籤相同，模型無從偏好最準的框。標準解法為 IoU 預測頭（分數 = 分類信心 × 預測 IoU），如 IoU-Net / FCOS centerness | 中，偏離原始設定 |
| U6 的 proposal 密度 | 論文每圖約 2,200 個，稀疏集合使最佳框更易勝出 | 中 |
| 官方碼細節 | U4（in_channels 5）、U5（參數 1.6M）尚未核對 | 需取得官方碼 |

### 5.10 R1-3 收束：接受 75.7 為工作基線（v1.9）

**決策：不再追那 15 分，直接進 H2 曲線。**

理由：H2 比較的是「向量組與點陣組在同一實作下的相對衰減」，
兩組受對等處理即成立，**不需要與論文一致的絕對數值**。
IoU 預測頭列為 R2 的可選改進，不在 R1 啟用。

#### R1 定案設定

| 參數 | 值 | 依據 |
|---|---|---|
| `dedup_tol` | 0.05 | §5.5 掃描，不敏感 |
| `expand_ratio` | **0.0005** | §5.9 步驟 6，訓練實測 |
| `stroke_pad` | **True** | §5.9 步驟 2，天花板 84.7% → 95.2% |
| `grid` | 10 | U1 暫定 |
| 框定義 | tight box | §5.5 U2 |
| `pos_iou` / `neg_iou` | **0.8 / 0.4** | §5.9 步驟 1 |
| `neg_per_pos` | 4.0 | §5.6 |
| 後處理 | plain NMS @0.3 | §5.9 步驟 7，框投票否決 |
| 訓練 | 20k iters、batch 4、Adam 2.5e-4、cosine | 固定 iteration（§4.6） |

## 6. 實驗設計（R1-4）

### 6.1 H2 標註效率曲線：向量組

**已啟動**（`scripts/run_all.sh`）。

| 項目 | 設定 |
|---|---|
| 切分 | **依版面分組**（5/2/3 → 500/200/300），無版面洩漏 |
| 標註比例 | 0.5%, 1%, 2%, 5%, 10%, 25%, 50%, 100%（`n_train` = 2/5/10/25/50/125/250/500） |
| seeds | 5（0–4） |
| 總 run 數 | **40** |
| 訓練預算 | 每個 cell 固定 20k iterations（與標註量解耦） |
| 回報 | mean ± std，同時記錄 final 與 val-selected best |
| 預估時間 | 約 7.5 小時（單張 RTX 4060） |

**前置校準**：分組切分下 `absolute` vs `cluster_relative` 座標各跑 10k iters，
自動以勝出者跑完整曲線。（§5.7 顯示修正前相對座標大幅較優，修正後需重新確認。）

### 6.2 點陣對照組（資料已備妥）

依 §2.6 決策丙，兩種渲染皆已產生（`scripts/prepare_raster.py --long-side 1024`）：

| 變體 | 影像 | 磁碟 | 平均線寬 | **最小線寬** |
|---|---|---|---|---|
| `vector_only`（資訊對等） | 1000 | 541 MB | 1.53 px | **0.89 px** |
| `composited`（YOLaT 原設定） | 1000 | 541 MB | 1.53 px | **0.89 px** |

- 由**已解析的貝茲**自行渲染 → 與向量組幾何同源
- 保持長寬比（不壓成正方形）、2× 超取樣 + LANCZOS 抗鋸齒
- 產生 82 組 split 設定（41 split × 2 變體），**與向量組共用同一批圖片 ID**
- YOLOv8-s 安裝於**獨立 venv（`.venv-yolo`）**，避免更動主 venv 的 torch 而影響進行中的訓練

> ⚠️ **實測記錄（§1.5 暫緩事項的補充）**：1024 長邊下最細筆畫僅 **0.89 px，仍為次像素**。
> 超取樣使其以灰階形式保留而非完全消失，但這是點陣組的先天劣勢。
> **若點陣組表現異常低落，此為第一順位待查變因**；重渲染至 2048 約需 8 分鐘 CPU 與 4.4 GB 磁碟。

### 6.3 coord_mode 校準結果（分組切分、100% 標註、10k iters）

| 座標模式 | AP50 | AP75 | mAP |
|---|---|---|---|
| absolute | 36.10 | 24.48 | 26.85 |
| **cluster_relative（採用）** | **59.38** | **52.88** | **47.52** |

→ 即使線寬 bug 已修、`expand_ratio` 已調，**相對座標在分組切分下仍勝出 1.8 倍**。
確認絕對座標的「位置捷徑」是分組切分下的主要障礙，且與其他修正互相獨立。

**洩漏效應的乾淨量測**（相同設定、相同 10k iters）：

| 切分 | 100% 標註 mAP |
|---|---|
| 隨機（有洩漏） | 75.7 |
| **分組（無洩漏）** | **47.5** |

→ **28 分差距**，在所有實作 bug 修正後依然存在。


### 6.3 待辦

- [ ] 20k iters 完整重現 run，比對 §3.6 出口條件（mAP 90.59 ± 1.5）
- [ ] 點陣對照組：自 PIL 渲染向量層（方案甲）與完整渲染（方案乙）兩組
- [ ] U4／U5／U6 查官方碼
- [ ] 40 個 run 的標註效率曲線



## 7. 實作實驗（R1-5）

### 7.1 ✅ 向量組標註效率曲線（完成 2026-09-10）

40/40 cells，0 失敗，194 分鐘（單張 RTX 4060）。
設定：分組切分、`cluster_relative` 座標、`pos_iou` 0.8、固定 20k iterations、5 seeds。

| 標註比例 | 訓練圖數 | mAP | ± std | min–max | AP50 | 佔滿標註 |
|---|---|---|---|---|---|---|
| 0.5% | 2 | 4.91 | 1.61 | 2.5–7.0 | 6.78 | 9.5% |
| 1% | 5 | 21.13 | 7.78 | 14.4–31.0 | 27.54 | 41.0% |
| 2% | 10 | 40.59 | 6.09 | 35.5–50.7 | 50.04 | **78.7%** |
| 5% | 25 | 46.77 | 2.91 | 42.3–49.6 | 58.11 | **90.7%** |
| 10% | 50 | 51.28 | 3.18 | 46.3–54.6 | 62.93 | 99.4% |
| 25% | 125 | 54.08 | 3.47 | 48.7–57.3 | 66.60 | 104.9% |
| 50% | 250 | **55.00** | 3.16 | 51.5–59.9 | 68.41 | 106.7% |
| 100% | 500 | 51.56 | 2.58 | 48.9–54.7 | 64.33 | 100.0% |

**主要觀察**

1. **飽和極早**：10 張圖達滿標註的 78.7%，25 張達 90.7%，5% 之後曲線基本持平。
   證實 §1.5 的天花板效應，成因為資料多樣性低（10 版面 + 模板化符號）。
2. **H2 的可辨別區間為 0.5%–5%**，右側四點在誤差範圍內重疊，無鑑別力。
3. **變異數隨標註量下降**（1% 時 std 7.78 → 5% 時 2.91），驗證低比例區間需 5 seeds 的設計。
4. ⚠️ **非單調**：100%（51.56）低於 50%（55.00）與 25%（54.08）。
   差距約 1 個標準差，可能為雜訊；但也可能是**固定 iteration 預算的副作用**——
   100% 組每張圖僅被看 160 次，25% 組為 640 次。
   此為 §4.6「解耦標註量與訓練量」設計的另一面，須在論文中揭露。

### 7.2 點陣對照組：吞吐量問題

首次量測暴露 `scripts/train_raster.py` 的 `epochs_for()` 設計缺陷：

> 小 `n_train` 會使 epoch 數暴增（`n_train=25`、batch 16 → 每 epoch 僅 2 步，
> 20k 步需 10,000 個 epoch），而 Ultralytics **每個 epoch 都會執行驗證與存檔**，
> 開銷完全主導訓練時間。

**修正方向**：於 `train.txt` 中**重複圖片清單**，使每個 epoch 約 100+ 步
（等效於向量組的 `InfiniteLoader` 循環），並關閉逐 epoch 驗證。

**預算對齊基準**：向量組共見 20,000 iters × batch 4 = **80,000 個樣本呈現**。
點陣組應以相同的樣本呈現數為預算，而非相同的步數。

### 7.3 🔴 診斷：增強不對等，結論一度完全反轉（2026-09-11）

首個點陣對照 cell 顯示點陣組領先 7 倍（34.00 vs 4.62），觸發五項假設的系統檢驗。

#### 零成本診斷先排除了兩項

以已訓練的 0.5% 向量模型同時在訓練集與驗證集評估：

| 評估集 | AP50 | mAP | proposal 精確率 | 召回率 |
|---|---|---|---|---|
| **訓練集**（2 張） | **100.00** | **82.73** | **100.0%** | **100.0%** |
| 驗證集（200 張） | 3.28 | 2.30 | 7.2% | 12.0% |

訓練 loss 於 10k 步降至 **0.0000**。→ **管線完全正確**（非程式 bug），是純粹的泛化崩潰。
且向量組全程僅佔用 **107 MB / 8 GB** VRAM → **硬體限制亦排除**。

#### 2×2 因子實驗（同一 cell：0.5%、seed 0、相同 2 張圖、相同 80k 樣本預算）

| | **無增強** | **有增強** | 增強增益 |
|---|---|---|---|
| **向量 GNN**（367 k 參數） | **4.62** | 29.45 | 6.4× |
| **點陣 YOLOv8n**（3.2 M 參數） | **0.29** | 34.00 | **119×** |
| **向量 / 點陣** | **16.2×** | 0.87× | |

**結論翻轉**：在**完全對稱的無增強條件**下，向量組勝出 **16.2 倍**
（點陣組 mAP 0.29 等同未學到任何東西——隨機初始化的 CNN 在 2 張圖上無法學習）。
先前「點陣勝 7 倍」的結論**完全是實驗設計缺陷的產物**。

#### 但需誠實記錄另一半

兩邊都用上各自自然的增強後，差距消失（29.45 vs 34.00，點陣略勝 15%）。
而**增強對點陣組的幫助是 119 倍，對向量組僅 6.4 倍**。

> **理論意涵**：向量的結構先驗與資料增強在功能上**部分重疊**——
> 兩者都在告訴模型「哪些變換不改變語意」。向量格式把這件事**內建於表示之中**，
> 點陣模型必須靠增強從外部灌入。一旦強增強到位，先驗的邊際價值大幅下降。

**H2 修正表述**：

> 向量的結構先驗在**缺乏或無法使用強資料增強**的條件下，提供顯著的標註效率優勢；
> 在雙方皆使用其自然增強時，優勢大幅縮小。

此表述較原版更精確且更易辯護。

#### 向量專屬的精確增強（新實作 `src/yolat/augment.py`）

設計依據一項先經驗證的結構事實：**模型完全 cluster-local**——
stroke 邊僅存在於連通分量內、position 流僅在 cluster 內聚合、proposal 僅在 cluster 內枚舉，
**跨 cluster 無任何資訊流**。三項推論：

1. 平移增強對本模型為 **no-op**（座標已是 cluster 相對）
2. **每個 cluster 可獨立變換** → 2 張圖 × 28 cluster 給出 4²⁸ 種組合，而非全域變換的 8 種
3. **無須處理 GT 對應**：損失僅讀 `prop_label`，而 cluster 幾何與其 proposal 框一起變換時，
   proposal 與符號的對應關係不變，標籤自動保持正確

內容：全域轉置 + 每 cluster 獨立水平／垂直翻轉 + 全域相似縮放（座標與線寬同步）。

**正確性驗證**：不變式「`prop_box` = 其 cell 範圍內節點加半線寬的外接框」
在增強前後誤差 **1.6e-7**（float32 極限），標籤零變動，200 次抽樣產生 200 種相異幾何。

> **可精確增強本身是向量表示的優勢**：變換直接作用於控制點，解析且無損；
> 點陣圖做同樣的旋轉翻轉會產生插值模糊，而本資料集最細筆畫僅 0.89 px。
> 已納入研究主張，非僅對照修正。

#### 向量組增強前後對照（分組切分，各 5 seeds）

| 標註比例 | 訓練圖數 | 無增強 | 有增強 | 增益 |
|---|---|---|---|---|
| 0.5% | 2 | 4.91 ± 1.61 | **31.39 ± 6.95** | **6.4×** |

#### 五項假設的最終判定

| 假設 | 判定 |
|---|---|
| **#1 實驗設計有大問題** | ✅ **確認，為主因**。增強不對等單一變因解釋全部落差 |
| #2 YOLaT 設計有問題 | 部分成立（無框迴歸限制 AP75），非低標註落差主因 |
| #3 任務太簡單 | 確認（25 張達滿標註 90.7%），影響曲線右側 |
| #4 硬體不足 | ❌ 排除（僅用 107 MB / 8 GB，loss 收斂至 0） |
| #5 假設相反 | ❌ **證據反向**：對稱條件下向量勝 16.2 倍 |

#### 待補：無增強條件的完整點陣曲線

目前無增強對照僅有 0.5% 單點。**H2 最強的證據形式是「雙方皆無增強」的完整曲線對比**，
需補跑點陣組 `--no-augment` 於 0.5/1/2/5% × 5 seeds。

### 7.4 先驗拆解與三組對照：R1 的決定性實驗（2026-09-12）

#### 觸發：一項無法用既有實驗區分的質疑

> 低資料量下先驗知識本來就非常強效（比照低規模時 CNN > ViT 的原理），
> 所以 16 倍優勢不能完全歸因於「向量表示」，可能只是「強先驗」。

亦即存在兩個競爭解釋：

- **(a) 表示**：向量格式承載了點陣模型必須自行學出的結構
- **(b) 先驗強度**：資料稀少時強歸納偏置本來就佔優，與表示無關

既有的「向量 GNN vs YOLOv8」比較**無法區分**，因為它一次換掉了
表示、backbone、偵測頭、參數量、框迴歸**五樣東西**。

#### 實驗一：向量端逐項拆先驗（0.5%、無增強、seeds 0–2）

| 變體 | mAP | sd | Δ | 保留 |
|---|---|---|---|---|
| 完整模型 | 4.22 | 1.53 | — | 100% |
| **− 連通性**（stroke 邊） | **0.06** | 0.08 | **−4.17** | **1%** |
| − 相對幾何（邊屬性） | 7.77 | 3.14 | +3.54 | 184% |
| − 群組（position 流） | 4.95 | 2.47 | +0.73 | 117% |

**拿掉連通性，向量模型歸零（0.06）——低於點陣 YOLOv8 的 0.29。**

> **與論文消融方向相反的意外結果**：YOLaT 在滿標註下拿掉邊屬性掉 4.34、拿掉 position 邊掉 3.42；
> 我們在 2 張圖的極端低資料下**兩者都是正增益**。合理解釋是這些額外參數在此條件下
> 提供的是**過擬合能力**而非有用偏置。n=3、sd 3.14，屬提示性證據，列為 R2 待驗證項。

#### 實驗二：三組對照 — 只換特徵抽取器（`src/yolat/raster_ctrl.py`）

設計原則：**共用 pipeline，單一變因**。

```
共用：同一批圖 → 同一批 proposal → 同一組標籤 → 同一個分類頭 → 同一套 NMS 與 AP
差異：只有「如何產生區域特徵向量」
```

| 模型 | 表示 | 先驗 | 總參數 | backbone | head |
|---|---|---|---|---|---|
| 向量 GNN | 向量 | 強（連通／群組／幾何） | 367 k | 34,368 | 332,817 |
| **TinyCNN** | 點陣 | 強（局部性＋權重共享） | 527 k | 194,528 | **332,817** |
| **TinyViT** | 點陣 | 弱（全域注意力） | 755 k | 422,304 | **332,817** |

三者 stride 均為 16、特徵圖均為 32×32、分類頭參數完全相同。
區域特徵以 `roi_align` 池化成與 GNN 相同的 384 維向量，餵進同一個 head。

**結果**（無增強、seeds 0–2）：

| 標註 | 圖數 | 向量 GNN | TinyCNN | TinyViT | GNN/CNN | **CNN/ViT** |
|---|---|---|---|---|---|---|
| 0.5% | 2 | 4.22 ± 1.5 | 0.78 ± 0.7 | 0.33 ± 0.5 | 5.4× | **2.3×** |
| 1.0% | 5 | 20.18 ± 9.4 | 1.45 ± 1.5 | 0.87 ± 1.0 | 13.9× | **1.7×** |

#### 判定

| 效應 | 量化 |
|---|---|
| **通用歸納偏置**（CNN vs ViT，同輸入同 head） | **1.7–2.3 倍** |
| **向量 GNN 相對 TinyCNN** | 再多 5.4–13.9 倍 |

配合實驗一：向量優勢幾乎全部來自**連通性先驗**，而連通性**在點陣表示中不存在**——
CNN 無法被賦予它，只能從像素推論。

> **(a) 與 (b) 不是互斥解釋。**
> 向量表示的價值，正是它免費提供了一個點陣表示無法提供的先驗。

**兩項必須記錄的保留**

1. **比值不穩定**：TinyCNN 0.78、TinyViT 0.33 皆接近完全失敗，小數字相除放大雜訊。
   應表述為「點陣兩組都學不起來、向量組能學」，而非精確倍數。
2. **解析度混淆未排除**：控制組渲染在 512 px，最細筆畫 **0.44 px**（次像素）。
   CNN vs ViT 不受影響（同輸入），但 **GNN vs CNN 受影響**。
   排除方式：TinyCNN 拉到 1024 px 重跑 6 run（約 2.5 小時）。**尚未執行。**

---

## 8. R1 結論與理論討論

### 8.1 H2 的判定

> **H2 在 R1 未獲支持。**
>
> 在工程成熟度對等的條件下（雙方皆使用自身自然的增強），向量輸入未展現顯著的標註效率優勢
> （31.39 vs 34.00，0.5% 標註）。
> 在完全對稱的受控條件下（同 pipeline、無增強），向量輸入明顯較佳（4.22 vs 0.78），
> 但雙方絕對效果都極低，且存在未排除的解析度混淆。

**不是「方向相反」，而是「優勢被另一側的工程成熟度抵銷」。**

差距可歸因於三項對照組優勢，且皆已量化或確認：

| 對照組優勢 | 幅度 |
|---|---|
| 參數量 | YOLOv8n 3.2 M vs 向量 367 k（**9 倍**） |
| 成熟度 | 現代偵測頭、框迴歸、經大量調校的訓練配方 |
| 本方實作打折 | 重現僅到 75.7 vs 論文 90.6 |

### 8.2 核心機制：先驗與增強的功能重疊

2×2 因子實驗顯示增強對兩者的增益極不對稱（點陣 119×、向量 6.4×）。解釋：

> 結構先驗與資料增強**在功能上重疊**——兩者都在告訴模型「哪些變換不改變語意」。
> 向量格式把這件事**內建於表示之中**；點陣模型必須靠增強從外部灌入。
> 一旦強增強到位，先驗的邊際價值大幅下降。

這是 R1 最可發表的理論貢獻，且有明確的實證支撐。

### 8.3 對一項替代解釋的評估

曾提出的替代框架：「特徵輸入 ≈ 邏輯型輸入 ≈ 圖神經輸入，所以效果相近」。

**本研究數據不支持此機制**：特徵與拓撲在我們的消融中**作用方向相反**——

| 拿掉 | 結果 |
|---|---|
| 連通性（純拓撲） | 4.22 → **0.06**（歸零） |
| 邊屬性（幾何特徵） | 4.22 → **7.77**（反而變好） |

特徵向量編碼**整體屬性**，圖編碼**元素間的關係**。將兩者歸為同一類會掩蓋此差異。

**更貼切的框架見 §8.4 文獻 ②**：表示的理論優勢會被另一側的工程成熟度、
預訓練資源與解析度優勢抵銷。此說法可驗證、有前例，且與本研究數據一致。

### 8.4 相關文獻（R1 結論的定位）

**① 最直接的鏡像實驗：[Benchmarking Graph Neural Networks for fMRI analysis](https://arxiv.org/abs/2211.08927) (2022)**

**1D CNN 在所有樣本量下都勝過 GNN，小樣本時差距最明顯。** 作者原話：

> 「採用 GNN 的動機是，納入資料結構的歸納偏置應能減少對大樣本的需求——
> 這個優勢我們在實證結果中很遺憾地沒有觀察到。」

歸因：**圖建構本身有噪音**（功能連結圖無客觀 ground truth，靠任意閾值建圖）。

> **對本研究有利的對比**：我們的消融顯示連通性先驗**確實有效**（拿掉即歸零），
> 代表 SESYD 的圖是「建對的」。這是本研究相對該文的結構性優勢，值得在論文中明說。

**② 本研究結論的最佳理論框架：[Volumetric and Multi-View CNNs for Object Classification on 3D Data](https://arxiv.org/abs/1604.03265) 及後續**

3D 領域的直接前例：**渲染成 2D 圖再用成熟 CNN，長期勝過直接吃原生 3D 表示**。
歸納原因為三點：能沿用成熟的 2D 架構、能用大規模 2D 預訓練、記憶體需求低故能用更高解析度。

→ **與本研究「優勢被工程成熟度抵銷」的結論是同一個機制**，且為已被廣泛接受的先例。

**③ 「邏輯型／特徵輸入」的權威分析：[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) (NeurIPS 2022)**

樹模型在表格資料上仍勝出，原因為三項歸納偏置：對無資訊特徵穩健、**保留資料的方向性**、
能學不規則函數。神經網路的**旋轉不變性反而有害**。

**④ 理論基礎：[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) (Battaglia et al., 2018)**

後續研究的關鍵補充：**壞的歸納偏置對 GNN 的傷害大於對 CNN**，
因為圖支援任意成對關係，偏置更靈活也更強勢。

**⑤ 與「語意 token 無助」相反的證據**

文獻傾向顯示語意化 tokenization **有幫助**：superpixel tokenization 提升 DINO-ViT 多數指標；
object-centric token 在僅保留 2.2% token 的壓縮下維持 97.4% 效能。
→ 「特徵輸入沒比 patch 輸入好」此前提本身**不是一致結論**，引用時須謹慎。

---

## 9. 未來展望

### 9.1 收尾 R1 的最小工作（可選）

| 項目 | 目的 | 成本 |
|---|---|---|
| TinyCNN @ 1024 px × 2 比例 × 3 seeds | **排除唯一未解的混淆變因**，確定 GNN vs CNN 的 5.4 倍有多少來自解析度 | 2.5 hr |
| 向量增強曲線補完 1%/2%/5% | 完成增強條件下的完整對照 | 3 hr |
| 點陣無增強完整曲線 | H2 最強的證據形式（雙方皆無增強） | 6 hr |

> **優先序建議**：第一項最重要——它是目前唯一還能實質改變結論的變因。

### 9.2 R2 候選方向（依價值排序）

**① 版面洩漏（最可發表，且已有完整數據）**

隨機切分 AP50 **97.27** vs 分組切分 **14.71**。意涵：YOLaT 回報的數字建立在共用版面的切分上，
可能高估跨版面泛化能力。可獨立成文，且不需新實驗——數據已在手。

**② 更難的資料集：FloorPlanCAD**

SESYD 飽和過早（25 張達 90.7%），右側曲線無鑑別力。
FloorPlanCAD：11,602 張真實 CAD、35 類、多樣性高一個數量級。
需新解析器（其 SVG 格式與 SESYD 不同），約 1–2 天。

**③ 定位品質訊號（IoU 預測頭）**

剩餘重現差距（75.7 vs 90.6）已定位於此。分數 = 分類信心 × 預測 IoU（IoU-Net / FCOS centerness）。
**偏離 YOLaT「僅 cross-entropy、無迴歸」的原始設定，須在論文明載。**

**④ SymPoint 的 ACM / CCL 移植**

原訂 R2 主線。連接注意力（端點距離 < 1px 即視為連接）與對比連接學習。
考量到 §7.4 顯示連通性是唯一有效的先驗，**強化連通性建模**在理論上是最對症的方向。

**⑤ 自監督預訓練軸（SVGformer / IconShop-MLM）**

SVG 是離散 token 序列，可直接套 MLM 而無須設計 augmentation。
無標註語料充足（FIGR-8 1.5 M、SVG-Stack 2.1 M）。
配合 §2.5.4 的 2×2 因子設計（表示 × 預訓練），可將兩因素解耦。

**⑥ 低資料下的先驗有效性（由 §7.4 的意外結果衍生）**

實測顯示相對幾何與群組先驗在 2 張圖時**是負增益**，與論文在滿標註下的消融方向相反。
若在多個標註比例上系統重複此消融，可得出「**先驗的價值隨資料量變化，且可能翻號**」的結論。
這是一個未見於文獻、且本研究已有初步證據的方向。

### 9.3 論文架構建議

依目前手上的證據，最可行的敘事是：

1. **重新檢視向量輸入的優勢**（版面洩漏 + 線寬 bug → 原基準被高估）
2. **建立無洩漏的重新基準**（分組切分下的完整曲線）
3. **釐清優勢來源**（連通性先驗是唯一有效者；與增強功能重疊）
4. **給出有條件的結論**（在缺乏強增強時成立；工程成熟度可抵銷）

此架構的優點：**不依賴 H2 成立**。即使 H2 未獲支持，1–3 仍是紮實的貢獻。

---

## 10. R2 / R3

> 🔒 凍結。R1 已於 2026-09-12 終止，結論見 §8。R2 方向見 §9.2，需明確批准後啟動。

---

## 附錄 A：參考文獻（已驗證）

- [1a] Jiang et al., *Recognizing Vector Graphics without Rasterization*（YOLaT）, NeurIPS 2021. arXiv:2111.03281 — 程式碼：github.com/microsoft/YOLaT-VectorGraphicsRecognition
- [1b] Jiang et al., *Hierarchically Recognizing Vector Graphics and A New Chart-Based Vector Graphics Dataset*（YOLaT++）, T-PAMI 2024.
- [1c] Liu et al., *Symbol as Points: Panoptic Symbol Spotting via Point-based Representation*（SymPoint）, ICLR 2024. arXiv:2401.10556
- [1d] Fan et al., *FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting*, ICCV 2021.
- [1e] Zheng et al., *GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings*, CVPR 2022.
- [1f] Yang et al., *SketchGNN: Semantic Sketch Segmentation with Graph Neural Networks*, ACM TOG 2021.
- [1g] Fan et al., *CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings*, CVPR 2022.
- [2a] *ArchCAD-400K: A Large-Scale CAD Drawings Dataset and New Baseline for Panoptic Symbol Spotting*, 2025. arXiv:2503.22346 — 提供 H2 的反面證據
- [2b] Carlier et al., *DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation*, NeurIPS 2020.
- [3a] *Benchmarking Graph Neural Networks for FMRI analysis*, 2022. arXiv:2211.08927 — **本研究最直接的鏡像實驗**：1D CNN 在所有樣本量下勝過 GNN，作者明言「結構歸納偏置應減少樣本需求」的優勢未被觀察到
- [3b] Qi et al., *Volumetric and Multi-View CNNs for Object Classification on 3D Data*, CVPR 2016. arXiv:1604.03265 — 3D 領域前例：渲染成 2D 再用成熟 CNN 勝過原生 3D 表示
- [3c] Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022. arXiv:2207.08815 — 「邏輯型／特徵輸入」的權威分析
- [3d] Battaglia et al., *Relational inductive biases, deep learning, and graph networks*, 2018. arXiv:1806.01261 — 理論基礎


## 附錄 B：YOLaT 深度解析

### B.1 Pipeline

1. **圖元統一化**：所有 SVG 圖元轉三次貝茲曲線 `B(t) = (1-t)³p₀ + 3(1-t)²t·p₁ + 3(1-t)t²·p₂ + t³p₃`。端點 `p₀,p₃` 成為圖節點，離曲線控制點 `p₁,p₂` 成為邊屬性。
2. **建無向多重圖**：節點特徵 7 維 `x = concat(pˣ, pʸ, RGB, w)`。兩種邊：
   - **stroke-wise ℰₛ**：兩端點間存在貝茲曲線 → 拓撲連通性；邊屬性為控制點座標。
   - **position-wise ℰₚ**：同一空間 cluster 內全連接 → 空間鄰近性；無邊屬性。
   - cluster 劃分：stroke 邊取連通分量 → 各分量最小外接矩形向外擴張 → 重疊者合併。
3. **雙流 GNN**（各 2 層，hidden 64）：
   - Stroke 流：`hᵢᵗ⁺¹ = fˡ(hᵢᵗ) + (1/|𝒩ᵢˢ|) Σⱼ fˢ(concat(hᵢᵗ, hⱼᵗ − hᵢᵗ, xᵉᵢⱼ))`
   - Position 流：`zᵢᵗ⁺¹ = (1/|𝒩ᵢᵖ|) Σⱼ∈𝒩ᵢᵖ∪{i} fᵖ(zⱼᵗ)`，僅在最後一層聚合（早聚合 over-smoothing，−2.7 mAP）
   - **mean-pooling 優化**：cluster 內先各自算 `fᵖ(zᵢ)`，再對 cluster 取一次均值後廣播 → O(|C|²) 降為 O(|C|)
4. **Proposal**：每 cluster 切 10×10 網格，枚舉頂點對成 axis-aligned 矩形，超尺寸者濾除。無 anchor、無 bbox regression。
5. **分類**：`r = concat(rₛ⁰..rₛᵀ, rₚ⁰..rₚᵀ)`，`rₛᵗ = mean_{i∈𝒱ʳ} hᵢᵗ` → 3 層 MLP (512→256)。IoU ≥ α 匹配 GT，否則歸背景類 C。損失僅 cross-entropy。

### B.2 消融結果（Floorplan）— 對「假設原因 1」的量化證據

| 拿掉什麼 | mAP | Δ |
|---|---|---|
| 完整 YOLaT | **90.59** | — |
| stroke-wise 邊（拓撲連通） | 86.00 | **−4.59** |
| 邊屬性（控制點座標） | 86.25 | −4.34 |
| position-wise 邊（空間鄰近） | 87.17 | −3.42 |
| 鄰居差 `hⱼ−hᵢ` | 87.83 | −2.76 |

→ **拓撲連通性比空間鄰近性更重要**，這是向量格式獨有的免費資訊。

| 聚合方式 | mAP |
|---|---|
| YOLaT (EdgeConv 式) | **90.59** |
| GraphSage | 85.26 |
| GAT | 83.92 |
| GCN | 83.32 |

→ 關鍵不是「用 GNN」，而是「鄰居差 + 邊屬性」的顯式相對幾何建模。

### B.3 完整效能表（Floorplan）

| 方法 | 預訓練 | mAP | 推論 ms | Params | GFLOPs |
|---|---|---|---|---|---|
| YOLOv3-tiny | ✗ | 53.24 | 1.2 | 8.7M | 13.0 |
| Faster R-CNN R50-FPN | ✗ | 66.53 | 73.3 | 41.4M | 165.7 |
| RetinaNet-R50-FPN | ✗ | 79.18 | 79.2 | 38.0M | 189.2 |
| YOLOv4 (Scaled) | ✗ | 79.59 | 11.7 | 70.3M | 165.5 |
| **YOLaT** | **✗** | **90.59** | **1.3** | **1.6M** | **1.5** |
| Faster R-CNN R50-FPN | ✓ | 90.25 | 71.2 | 41.4M | 165.6 |

Diagram 資料集：YOLaT 89.67，略輸預訓練 Faster R-CNN (90.76)，勝過所有未預訓練 baseline。
三次執行標準誤：±0.0003 (floorplan)、±0.0008 (diagram)。

### B.4 五個風險點（直接影響 §6 實驗設計）

1. **天花板效應**：AP50 已 98.83、標準誤 ±0.0003，任務接近飽和。在此資料集畫標註效率曲線，高比例區間兩條線會重疊 → **H2 最大威脅**。對策：改用 FloorPlanCAD，或把標註比例區間壓到 1%–10%。
2. **點陣對照組公平性存疑**：論文未充分交代光柵化解析度。CAD 細線在低解析度下會斷裂消失 → raster baseline 的弱勢有多少來自表示、多少來自解析度不明。**對策：把光柵化解析度列為受控變因掃描。**
3. **資料為 SESYD 合成**：符號自模板貼上、幾何無雜訊，向量表示天然無損；真實掃描/手繪 CAD 不具此條件。
4. **只能輸出水平矩形框**：網格頂點枚舉的 axis-aligned 矩形，無法做旋轉框或分割 mask。切割任務需走 SymPoint 路線。
5. **無預訓練可用**：節點僅 7 維純幾何特徵，作者自陳需要大型向量資料集支撐 backbone 預訓練。對 H2 是雙面刃——比較乾淨無污染，但低標註時無外部知識可依靠。

### B.5 工程資訊

- Repo：`microsoft/YOLaT-VectorGraphicsRecognition`（NeurIPS'21 + T-PAMI'24）
- 環境：Python 3.8 + DeepGCN 環境腳本（`deepgcn_env_install.sh`），舊版 PyTorch Geometric API，現代環境需修改
- 訓練超參：batch size 4、lr 2.5e-4、arch flag `centernet3cc_rpn_gp_iter2`
- 資料前處理：Floorplan 跑 `svg_utils/build_graph_bbox.py`；Diagram 跑 `build_graph_bbox_diagram.py`（`bbox_sampling_step` 5 vs 10）

---

## 附錄 C：變更紀錄

- v0.1 建立骨架
- v0.2 完成 §1（論點文獻驗證、H2 判定為研究缺口、原因 2 表述修正）與 §2（候選表、算力估算、建議）
- v0.3 新增附錄 B：YOLaT 深度解析（pipeline 公式、消融、完整效能表、五個風險點）
- v0.4 依決策移除「不需計算特徵相對位置」子假設；新增 §1.5 已決／暫緩事項（天花板效應、光柵化解析度列為暫緩風險）
- v3.0 **R1 終止**（另見 `docs/journal.md` 研究日誌）。新增 §7.4（先驗消融 + 三組對照）、§8（H2 判定、先驗/增強功能重疊機制、替代解釋評估、5 篇定位文獻）、§9（未來展望與論文架構建議）。關鍵結果：拆掉連通性向量模型歸零（4.22→0.06）；CNN vs ViT 的純先驗效應僅 1.7–2.3 倍，向量再多 5.4–13.9 倍。**H2 未獲支持，但非方向相反**
- v2.2 🔴 **診斷增強不對等並修正**：2×2 因子實驗顯示**無增強條件下向量勝 16.2 倍**（4.62 vs 0.29），先前「點陣勝 7 倍」全為設計缺陷產物。實作向量專屬精確增強（`augment.py`，per-cluster 變換，不變式誤差 1.6e-7），向量組 0.5% 由 4.91→**31.39**。H2 表述修正為「在缺乏強增強的條件下成立」。五項假設判定完成
- v2.1 ✅ **向量組標註效率曲線完成**（40/40，194 分鐘）：10 張圖達滿標註 78.7%、25 張達 90.7%，**H2 可辨別區間為 0.5%–5%**；發現 100% 非單調低於 50%（固定 iteration 預算的副作用，已記錄）。點陣組暴露 `epochs_for()` 開銷缺陷，改以「樣本呈現數」對齊預算
- v2.0 點陣對照組資料備妥（兩變體各 1000 張，1024 長邊，最小線寬 0.89px 已記錄為待查變因）；ultralytics 隔離於 `.venv-yolo`。coord_mode 校準：**cluster_relative 勝出 1.8 倍**（47.52 vs 26.85）；**洩漏效應在所有 bug 修正後仍達 28 分**（隨機 75.7 vs 分組 47.5）
- v1.9 **R1-3 收束**：接受 mAP 75.7 為工作基線，不再追論文差距（H2 不需絕對數值一致）。定案設定表見 §5.10。**R1-4 啟動**：H2 標註效率曲線 40 run（8 比例 × 5 seed、分組切分、固定 20k iters），前置校準 coord_mode
- v1.8 `expand_ratio` 定為 **0.0005**（mAP 68.4→**75.7**）；框投票經實測**否決**（提升 AP75 但傷 mAP）。重現進度 11.7→75.7，距論文 90.6 尚差 15 分。已排除幾何天花板、標籤指派、cluster 劃分、後處理四項；剩餘主要槓桿為**定位品質訊號（IoU 預測頭）**，屬架構層新增需決策
- v1.7 追平論文除錯：🔴 **找到真 bug —— 預測框少算半個線寬**，原幾何天花板僅 84.7% < 論文 90.59，**修正前不可能重現**；修正後 mAP 57.3→**68.4**。再定位到模型卡在 **cluster 框天花板 70.7%**，並發現 `expand_ratio` 最佳化目標選錯（合併反而傷害框品質，修正 §5.5 的錯誤結論）
- v1.6 決策：**先修 AP75 追平論文再訂出口條件**；洩漏發現**先擱置**、H2 優先。實作 positive/ignore/negative 三分標籤（`relabel`），快取加存 `prop_iou`/`prop_gt` 以便零成本掃描閾值。實測每 GT 對應 17.5 個正 proposal（pos_iou=0.5），確認為 AP75 落差主因
- v1.5 ✅ **兩個假說皆證實**：隨機切分 AP50 **97.27**（≈論文 98.83），分組切分僅 14.71 → **版面洩漏效應已量化**；cluster 相對座標使分組切分下 AP50 提升至 **47.12（3.2 倍）**。剩餘落差在 AP75（58.24 vs 94.65），已定位為**標籤指派閾值 α=0.5 缺乏定位品質訊號**
- v1.4 🔴 **重現失敗**（§5.7）：分組切分下 mAP 13.9 vs 論文 90.59。診斷確認**非 bug**（train AP50 96.53 vs val 9.29），為泛化崩潰。主假說：節點的**絕對座標**在版面內構成「位置→類別」捷徑，論文的隨機切分共用全部 10 版面故捷徑有效。已實作 `coord_mode=cluster_relative` 並啟動兩個對照實驗驗證
- v1.3 R1-3 管線全線跑通：前處理 1000 檔（28,065 物件與論文一致）、分組切分 500/200/300、40 個 split 檔、雙流 GNN + 積分圖池化 + 自建 COCO AP、訓練 22 ms/iter（107 MB VRAM）。**算力大幅下修：單 run 約 0.35 GPU-hr，全曲線約 13 GPU-hr，向量組本機即可完成**。新增 U5/U6（參數 367k vs 1.6M、FLOPs 55 vs 1.5 GFLOPs），並以召回實測否證「節點配對 proposal」假說
- v1.2 R1-3 實作：Python **3.10** venv 建成（3.11/3.12 本機不存在）；光柵化改為**自行以 PIL 渲染已解析貝茲**（無系統相依且保證兩組幾何同源）。完成 svg_parse / graph / proposals 三模組並實測驗證：解析器 1000/1000、圓近似誤差 0.027%；建圖超參選定 `dedup_tol=0.05`、`expand_ratio=0.002`（cluster:GT = 0.96）；**U2 由實測解決 → tight bbox**（AP75 區間 98.8% vs 97.0%）；proposal 召回上限 **99.5% @IoU0.5**
- v1.1 決策：發現 1 採**方案丙**（向量層渲染 + 完整渲染兩組對照皆跑）；發現 2 採**依版面分組切分**（GroupKFold by background，5:2:3）
- v1.0 R1-3 起步：SESYD 取得成功（HTTP-only）並完成全量勘查（§5）。五項發現：**SVG 僅含符號、牆體為點陣背景層**（H1 結構性偏差）、僅 10 種版面（切分洩漏 + 天花板成因）、光柵化次像素實測、新增 U4（in_channels 5 vs 7）、解析器範圍收斂為 line/circle/arc
- v0.9 **最終決策：路線 A + YOLaT 骨幹**（§2.7）。§3、§4 維持有效。2×2 因子設計與預訓練軸移至 R2；R1 實驗規模放寬（8 個比例點、低比例 5 seeds、多機平行）；環境策略改為本機開發 + 租用訓練
- v0.8 新增**硬約束 C1**（§2.6）：受試模型輸入必須為向量，不含光柵化；對照組不受此限。全候選重新篩選：StarVector 完全排除，DPSS／CADTransformer 由候選降為文獻對照
- v0.7 **算力限制解除**（改為租用雲端）。新增 §2.5：IconShop / StarVector 規格與判定（生成模型、序列長度不足、StarVector 輸入方向相反）、VGBench 警訊、2×2 因子設計、路線 A/B 分岔、租用成本表、重新納入 7 個先前被算力排除的候選、§2.5.8 揭示 2025 SOTA 為向量+點陣融合、§2.5.9 區隔 ArchCAD-400K 自動標註引擎
- v0.6 完成 R1-2 偽代碼（§4）：三個待核對不確定點 U1–U3、積分圖區域池化優化、六段偽代碼與逐段解釋、新增實作問題 8–14
- v0.5 改為**多輪制**（§0）：R1 = 重現 YOLaT + 驗證 H2，R2/R3 凍結。骨幹選定 A（YOLaT 雙流 GNN）。完成 R1-1 模型架構（§3）：重現規格、圖片層級標註子集裝置、7 項技術問題、出口條件
