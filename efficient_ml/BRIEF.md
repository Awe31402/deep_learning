# BRIEF：EfficientML.ai 隨堂筆記（Lec13～Lec24）

通用規則看 `/home/awe/disk/math-note/.claude/skills/lecture-notes/`（SKILL.md、TEMPLATE.md）。這份寫本課專屬的細節，和通用規則衝突時以這份為準。

## 目的

自己讀書用。不用重看 80 分鐘影片就能讀懂一講：補上教授口頭講、投影片沒寫的內容，用 ⭐ 標重點，最後有一頁速查表。

## 課程

- **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing（EfficientML.ai），Fall 2023
- **講者**：Song Han
- **課程網頁**：https://hanlab.mit.edu/courses/2023-fall-65940
- **播放清單**：https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB
- **範圍**：Lec13～Lec24。Lec02～Lec12 已有舊格式筆記，**不改**。不做 Lec01。

## 資料來源

1. **主要**：YouTube 英文自動字幕（`yt-dlp --write-auto-subs --sub-langs "en.*"`）。同一講有一般版和 Zoom 版時用 **Zoom 版**（跟 Lec12 一樣）。
2. **次要**：投影片 PDF，在本資料夾 `Lec<NN>-<主題>.pdf`。數字、圖表、公式以投影片為準。標記名稱用 `【投影片補充】`。
3. 投影片上引用的論文只拿來確認細節，不擴寫。
4. 不用其他網路文章、部落格、別人的筆記。

## 影片對照

| 講次 | 影片 ID（Zoom 版） | 長度 |
|---|---|---|
| 13 | caEDCA3Jo_4 | 1:17:03 |
| 14 | fcmOYHd57Dk | 1:14:19 |
| 15 | WU7tpyesm68 | 1:08:13 |
| 16 | 7s2YAGoLcCA | 1:16:27 |
| 17 | u6NEbhNiyAE | 1:00:45 |
| 18 | ZSZQL0q8eYY | 54:30 |
| 19 | 0YABI9mxG60 | 1:16:48 |
| 20 | vOPwwRCZ8q8（Zoom 版 uI4HWD8U6-4 字幕抓不到，改用一般版） | 1:17:33 |
| 21 | Z8GKkgE2840（Zoom 版 xu_6Tz5DPOw 字幕抓不到，改用一般版） | 1:16:12 |
| 22 | 36S3yCY193M | 1:12:51 |
| 23 | kCTzlodCZII（只有一般版） | 1:14:37 |
| 24 | **沒有影片** | — |

## 檔名與連結

- 檔名：`lecture<NN>_<主題英文小寫底線>_note.md`，例如 `lecture13_transformer_llm_part2_note.md`。
- 一講一個檔，上下集不合併。
- 上一集／下一集連到相鄰講次的筆記；「總目錄」連到 `進度.md`。
- frontmatter 的 `episode` 和 `lecture` 都填講次。`notes` 填投影片檔名和頁數。

## 讀者

學過基本深度學習（CNN、反向傳播），但沒上過這門課。

## 限制

- 不改舊的 Lec02～Lec12 筆記，最多在 Lec12 加一行連到 Lec13。
- 不 commit；PDF 不加進 git（太大）。
- **Lec24（期末報告）**：沒有影片，只用投影片寫一份短摘要。
- **Lec21～23（量子）**：照樣寫完整筆記。
- 不加課程沒教的延伸內容（自己寫的程式、實作）。
- 用自己的話整理，不整段照貼逐字稿或投影片。

## 流程

- Lec13 是樣品，已確認（2026-10-27）。
- Lec14 之後**不用先交大綱**，直接寫全文（使用者決定，蓋過 SKILL.md 流程第 2 步）。每寫完一講回報檔名和行數。
- 寫完更新 `進度.md`。
