# 远程机器跑数据 (4080 32GB)

> 替换之前 Colab 跑不出权重的版本。两个文件二选一，跑出的 `vl_attention_data.json` 拷到 `../src/data/` 即可。

## 一、为什么 Colab 那版没跑出来

| 问题 | 之前 | 现在 |
|---|---|---|
| CLIP 评分泄漏（候选里有 GT） | `[gt, "yes", "no", ...]` 必中 | 固定 `TOP_VQA_ANSWERS`（VQA 高频答案） |
| CLIP 注意力切片错 | `[:49,:49]` 是 patch-自相关，渲染出全平 | 取 CLS-token 行 reshape 7×7→14×14 |
| BLIP2 hook 路径不对 | `transformers` 5.x 没有 `.attention.self` | 兼容写法，读 `(context, attn_probs)` |
| BLIP2 评分被 prompt 污染 | 全文 substring 匹配 | 剥掉 prompt 再判，且匹配 10 个标注答案 |
| 图片下载 6.2GB | `wget val2014.zip` | 仅下 200MB JSON，图片按需下载 |
| HF 直连不通 | 直接报错 | 自动设 `HF_ENDPOINT=https://hf-mirror.com` |

## 二、怎么跑

### 方案 A：脚本（推荐，跑批用）

```bash
cd /root/autodl-tmp/VisProj/train
pip install -q transformers accelerate sentencepiece open_clip_torch

# 默认跑 200 个样本 × CLIP/BLIP2/BLIP2-T5 三个模型
python3 extract_attention.py

# 想先小步快跑（验证一切 OK）
python3 extract_attention.py --num-samples 20

# 只跑某个模型
python3 extract_attention.py --models clip
python3 extract_attention.py --models clip,blip2
```

预计耗时（4080 32GB）：

| 配置 | 时间 |
|---|---|
| 首次：下载 BLIP2-OPT (~15GB) + BLIP2-T5-XL (~15GB) | 15–25 min |
| 之后：200 样本 × 3 模型推理 | 8–12 min |

### 方案 B：notebook（可视化跑，调试方便）

```bash
cd /root/autodl-tmp/VisProj/train
jupyter notebook main.ipynb
```

按顺序 Run All。和脚本逻辑完全一致。

## 三、产物

- `./vl_attention_data.json` — 把它拷到 `../src/data/vl_attention_data.json` 就能直接被前端读

JSON 结构：

```json
{
  "matrix_data":     [{"model","sample_id","correct","confidence"}, ...],
  "attention_data":  {"<model>__<sample_id>": {"image_url","question","ground_truth","prediction","weights":14×14}, ...},
  "text_bias":       [{"sample_id","question","ground_truth","text_only_pred","text_only_correct","with_image_pred","with_image_correct"}, ...]
}
```

`text_bias` 是**新加的**，给视图五（语言先验）用：把 BLIP2 在 with-image 和 text-only 两种 setting 下的同一题预测/正确性都存了。

## 四、常见问题

**Q：HF 模型下不下来 / 卡住？**
A：脚本默认走 `https://hf-mirror.com`。若想换原站：`unset HF_ENDPOINT` 后再跑。

**Q：显存炸了？**
A：4080 的 32GB 跑 BLIP2-OPT-2.7b + BLIP2-T5-XL（fp16）峰值约 12–14GB，理论不会炸。如果你后续想加 LLaVA-1.5-7b 进去，那会到 ~16GB，仍然安全。真炸了把 `--num-samples` 降到 50。

**Q：跑了一半失败想接着跑？**
A：图片缓存在 `./vqa_data/val2014/`，VQA JSON 缓存在 `./vqa_data/`，重跑时加 `--skip-download` 跳过下载即可（只是图片仍按需补齐）。

**Q：CLIP 准确率为什么很低？**
A：CLIP 不是 VQA 模型，只是用 image-text 相似度从固定候选集里选答案。它的"错误模式"本身就是项目要分析的对象（视图二的差异热力图就吃这个），低准确率是预期行为，不是 bug。

## 五、跑完之后

```bash
cp vl_attention_data.json ../src/data/vl_attention_data.json
cd ..
npm run dev   # 前端查看效果
```
