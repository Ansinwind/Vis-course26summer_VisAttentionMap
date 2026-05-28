"""
VL Model Attention Extraction — production version for the 32GB-GPU box.

Outputs:
  ./vl_attention_data.json  — drop-in replacement for src/data/vl_attention_data.json

What this script does (mapping to SketchRoadmap.md):
  view 1 (Error Matrix)         -> matrix_data
  view 2 (Diff Heatmap)         -> attention_data[*].weights  (14x14 visual attention)
  view 3 (Layer Flow stub)      -> attention_data[*].layer_flow (per-layer mean attn on key patches)
  view 5 (Text Bias)            -> text_bias[*]  (text-only vs image+text accuracy)

Usage:
  python3 extract_attention.py --num-samples 200
  python3 extract_attention.py --num-samples 50 --skip-download   # reuse VQA jsons / images
"""
import argparse, os, json, gc, random, functools, sys, urllib.request, zipfile
from io import BytesIO
from pathlib import Path

# autodl / 内地机器默认走 hf-mirror.com（huggingface.co 直连不通）。
# 想走原站可以在 shell 里 `unset HF_ENDPOINT` 后再跑。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=200,
                    help="number of VQA samples (300 ≈ ~10 min on a 4080)")
parser.add_argument("--data-dir",   default="./vqa_data")
parser.add_argument("--out",        default="./vl_attention_data.json")
parser.add_argument("--skip-download", action="store_true",
                    help="don't re-download VQA JSON / images")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--models", default="clip,blip2,blip2_t5",
                    help="comma list of: clip, blip2, blip2_t5")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if device == "cuda" else torch.float32
print(f"[env] device={device} dtype={DTYPE}")
if device == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)} "
          f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

DATA_DIR = Path(args.data_dir); DATA_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR  = DATA_DIR / "val2014"

# ---------------------------------------------------------------------------
# 1) VQA v2 metadata download (questions + annotations only — small files).
#    Images are fetched lazily from the official COCO CDN, so we avoid the
#    6.2 GB val2014.zip in environments with limited disk.
# ---------------------------------------------------------------------------
Q_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
A_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
COCO_IMG_TPL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

def _download_unzip(url, dst_dir):
    z_path = dst_dir / "tmp.zip"
    print(f"[dl] {url}")
    urllib.request.urlretrieve(url, z_path)
    with zipfile.ZipFile(z_path) as z: z.extractall(dst_dir)
    z_path.unlink()

q_json = DATA_DIR / "v2_OpenEnded_mscoco_val2014_questions.json"
a_json = DATA_DIR / "v2_mscoco_val2014_annotations.json"

if not args.skip_download:
    if not q_json.exists(): _download_unzip(Q_URL, DATA_DIR)
    if not a_json.exists(): _download_unzip(A_URL, DATA_DIR)

print("[data] loading VQA v2 questions/annotations...")
questions = json.load(open(q_json))["questions"]
annos     = {a["question_id"]: a for a in json.load(open(a_json))["annotations"]}

# ---------------------------------------------------------------------------
# 2) Sample selection. We bias toward question types that are visually
#    grounded (e.g. "what color", "how many", "where") because purely
#    yes/no questions are dominated by the language prior.
# ---------------------------------------------------------------------------
buckets = {"what":[], "how":[], "where":[], "is_are":[], "other":[]}
for q in questions:
    txt = q["question"].lower()
    if   txt.startswith("what")  : buckets["what"].append(q)
    elif txt.startswith("how")   : buckets["how"].append(q)
    elif txt.startswith("where") : buckets["where"].append(q)
    elif txt.startswith(("is ","are ","does ","do ")): buckets["is_are"].append(q)
    else                         : buckets["other"].append(q)

per_bucket = max(1, args.num_samples // len(buckets))
chosen = []
for k, v in buckets.items():
    random.shuffle(v); chosen.extend(v[:per_bucket])
random.shuffle(chosen)
chosen = chosen[:args.num_samples]
print(f"[data] selected {len(chosen)} samples across {len(buckets)} question types")

IMG_DIR.mkdir(exist_ok=True, parents=True)

def fetch_image(image_id: int) -> Image.Image:
    """Try local val2014 dir first, otherwise download to disk on-demand."""
    fname = f"COCO_val2014_{image_id:012d}.jpg"
    fpath = IMG_DIR / fname
    if not fpath.exists():
        url = COCO_IMG_TPL.format(image_id)
        urllib.request.urlretrieve(url, fpath)
    return Image.open(fpath).convert("RGB")

samples = []
for q in tqdm(chosen, desc="fetch images"):
    try:
        img = fetch_image(q["image_id"])
    except Exception as e:
        print(f"[skip] image {q['image_id']}: {e}"); continue
    samples.append({
        "image": img,
        "question": q["question"],
        "ground_truth": annos[q["question_id"]]["multiple_choice_answer"],
        "sample_id": str(q["question_id"]),
        "image_id":  str(q["image_id"]),
    })
print(f"[data] {len(samples)} samples ready")

# ---------------------------------------------------------------------------
# 3) VQA-style answer scorer (10 annotators, accuracy = min(matches/3, 1)).
#    For our purposes a binary "did the model get it" is enough — we use
#    case-insensitive contained-match against the consensus answer plus
#    the per-annotator answers.
# ---------------------------------------------------------------------------
def is_correct(pred: str, sample: dict) -> bool:
    pred = pred.strip().lower()
    if not pred: return False
    gt   = sample["ground_truth"].lower()
    if pred == gt: return True
    # multi-annotator answers
    anno = annos[int(sample["sample_id"])]
    alts = [a["answer"].lower() for a in anno.get("answers", [])]
    return pred in alts or any(a == pred for a in alts)

# ---------------------------------------------------------------------------
# 4) CLIP — VQA scoring via candidate-set similarity.
#    To avoid the leaky [gt, yes, no, ...] candidate trick, we use the
#    top-K most frequent VQA training answers as a fixed candidate set.
# ---------------------------------------------------------------------------
TOP_VQA_ANSWERS = [
    "yes","no","2","1","white","3","red","blue","4","green","black","yellow",
    "brown","right","left","man","woman","wood","table","blue and white",
    "5","stop","6","gray","tennis","baseball","skateboarding","surfing","kite",
    "snow","grass","water","tree","trees","food","pizza","cat","dog","bird",
    "0","standing","sitting","walking","eating","sleeping","cake","laptop",
    "phone","clock","car","truck","bus","train","bike","horse","sheep",
    "cow","elephant","giraffe","zebra","bear","kitchen","living room",
    "bedroom","bathroom","beach","park","street","ocean","sky","sunny",
    "cloudy","day","night","summer","winter","plastic","metal","glass",
    "paper","ceramic","leather","cotton","old","young","big","small",
    "tall","short","happy","sad","frisbee","ball","racket","bat","glove",
    "book","computer","tv","sandwich","apple","orange","banana","carrot",
    "broccoli","donut","hot dog","wine","coffee","tea","water","milk",
]

def run_clip(samples):
    import open_clip
    print("[clip] loading ViT-B/32 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device, dtype=DTYPE).eval()

    # Patch last visual block so MultiheadAttention exposes per-head weights.
    cap = {}
    last_block = model.visual.transformer.resblocks[-1]
    orig_attn  = last_block.attn.forward
    @functools.wraps(orig_attn)
    def patched(q, k, v, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        out, w = orig_attn(q, k, v, **kw)
        cap["weights"] = w.detach()  # (B, heads, seq, seq)
        return out, w
    last_block.attn.forward = patched

    # Pre-encode candidate set once
    with torch.no_grad():
        cand_tok  = tok(TOP_VQA_ANSWERS).to(device)
        cand_feat = model.encode_text(cand_tok)
        cand_feat = cand_feat / cand_feat.norm(dim=-1, keepdim=True)
        cand_feat = cand_feat.to(DTYPE)

    results = []
    for s in tqdm(samples, desc="CLIP"):
        cap.clear()
        img_t = preprocess(s["image"]).unsqueeze(0).to(device, dtype=DTYPE)
        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat.float() @ cand_feat.float().T).squeeze(0)
            idx = sims.argmax().item()
            pred = TOP_VQA_ANSWERS[idx]

        # CLIP ViT-B/32 has 7×7=49 patches + 1 CLS at index 0.
        # We use the CLS-token row (index 0, attending to patch tokens 1..49)
        # averaged over heads, reshaped to 7×7, then upsampled to 14×14
        # for a denser heatmap in the front-end.
        attn = cap.get("weights")
        if attn is not None:
            a = attn[0].float().mean(0)            # (50, 50)
            cls = a[0, 1:50].cpu().numpy()         # (49,)
            grid = cls.reshape(7, 7)
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
            # nearest-neighbor 2× upsample
            grid14 = np.kron(grid, np.ones((2, 2), dtype=np.float32))
            weights = grid14.tolist()
        else:
            weights = [[0.0] * 14 for _ in range(14)]

        results.append({
            "sample_id":    s["sample_id"],
            "image_id":     s["image_id"],
            "model":        "CLIP",
            "question":     s["question"],
            "ground_truth": s["ground_truth"],
            "prediction":   pred,
            "correct":      is_correct(pred, s),
            "confidence":   float(sims.softmax(-1).max().item()),
            "weights":      weights,
        })

    del model, preprocess, tok, cand_feat
    gc.collect(); torch.cuda.empty_cache()
    print(f"[clip] acc={sum(r['correct'] for r in results)}/{len(results)}")
    return results


# ---------------------------------------------------------------------------
# 5) BLIP2 (OPT-2.7b) — VQA via free-form generation; pull cross-attention
#    from the Q-Former. We set output_attentions on the qformer config and
#    install a hook on the last cross-attention's `attention.self`, which
#    captures the (1, heads, 32, 257) tensor (32 query tokens × 257 ViT
#    image patches incl. CLS).
# ---------------------------------------------------------------------------
def run_blip2(samples, model_id="Salesforce/blip2-opt-2.7b", tag="BLIP2"):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"[{tag.lower()}] loading {model_id} ...")
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map="auto").eval()
    model.qformer.config.output_attentions = True

    cap = {}
    def hook(_m, _i, output):
        # Blip2QFormerMultiHeadAttention.forward returns (context, attn_probs)
        if isinstance(output, tuple):
            for x in output:
                if torch.is_tensor(x) and x.dim() == 4:
                    cap["weights"] = x.detach()
                    return

    # transformers 5.x layout: layer.crossattention.attention is the MHA module
    # (older versions had layer.crossattention.attention.self -- we walk down
    # whichever path actually exposes attention probs).
    # cross_attention_frequency=2, so only even layers have crossattention
    last_xattn_layer = next(l for l in reversed(model.qformer.encoder.layer) if l.has_cross_attention)
    xattn = last_xattn_layer.crossattention.attention
    target = getattr(xattn, "self", xattn)
    target.register_forward_hook(hook)

    results = []
    for s in tqdm(samples, desc=tag):
        cap.clear()
        prompt = f"Question: {s['question']} Answer:"
        inputs = processor(images=s["image"], text=prompt,
                           return_tensors="pt").to(device, DTYPE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        # strip prompt echo if present (OPT)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        pred = text.replace(prompt, "").strip()
        if "\n" in pred: pred = pred.split("\n")[0].strip()

        attn = cap.get("weights")
        if attn is not None:
            # (1, heads, 32 queries, 257 keys) — drop CLS key (index 0)
            a = attn[0].float().mean(0).cpu().numpy()           # (32, 257)
            patch = a[:, 1:257]                                 # (32, 256)
            # Aggregate the 32 query tokens to one heatmap by mean,
            # then reshape from 256=16×16 patches to a 14×14 view (drop edges).
            heat = patch.mean(0).reshape(16, 16)                # (16,16)
            heat = heat[1:15, 1:15]                             # crop -> 14x14
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            weights = heat.tolist()
        else:
            weights = [[0.0]*14 for _ in range(14)]

        results.append({
            "sample_id":    s["sample_id"],
            "image_id":     s["image_id"],
            "model":        tag,
            "question":     s["question"],
            "ground_truth": s["ground_truth"],
            "prediction":   pred,
            "correct":      is_correct(pred, s),
            "confidence":   0.5,  # generative — no calibrated score
            "weights":      weights,
        })

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    print(f"[{tag.lower()}] acc={sum(r['correct'] for r in results)}/{len(results)}")
    return results


# ---------------------------------------------------------------------------
# 6) Text-bias measurement (view 5).
#    Run a text-only baseline: same question fed to a tiny LM (the OPT-2.7b
#    inside BLIP2 is reused) without any image, and compare answers.
# ---------------------------------------------------------------------------
def run_text_only(samples, model_id="Salesforce/blip2-opt-2.7b"):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print("[textonly] reusing BLIP2 OPT decoder without image ...")
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map="auto").eval()

    # zero image: feeds an all-zero pixel input so q-former's contribution is
    # minimal — gives the language prior a fair chance.
    blank = Image.new("RGB", (224, 224), color=(127, 127, 127))
    results = []
    for s in tqdm(samples, desc="text-only"):
        prompt = f"Question: {s['question']} Answer:"
        inputs = processor(images=blank, text=prompt,
                           return_tensors="pt").to(device, DTYPE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        pred = text.replace(prompt, "").strip()
        if "\n" in pred: pred = pred.split("\n")[0].strip()
        results.append({
            "sample_id": s["sample_id"],
            "prediction": pred,
            "correct": is_correct(pred, s),
        })
    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# 7) Glue: run, assemble, dump JSON.
# ---------------------------------------------------------------------------
to_run = set(args.models.split(","))
all_results = []
if "clip"     in to_run: all_results += run_clip(samples)
if "blip2"    in to_run: all_results += run_blip2(samples,
                                "Salesforce/blip2-opt-2.7b", "BLIP2")
if "blip2_t5" in to_run: all_results += run_blip2(samples,
                                "Salesforce/blip2-flan-t5-xl", "BLIP2-T5")

text_bias_results = run_text_only(samples) if "blip2" in to_run else []
text_bias_index = {r["sample_id"]: r for r in text_bias_results}

# ---- load existing JSON to merge into (so re-runs append, not overwrite) ----
if os.path.exists(args.out):
    with open(args.out) as f:
        output = json.load(f)
    # index existing matrix rows by (model, sample_id) to avoid duplicates
    existing_keys = {(r["model"], r["sample_id"]) for r in output["matrix_data"]}
else:
    output = {"matrix_data": [], "attention_data": {}, "text_bias": []}
    existing_keys = set()

for r in all_results:
    if (r["model"], r["sample_id"]) in existing_keys:
        continue
    output["matrix_data"].append({
        "model":      r["model"],
        "sample_id":  r["sample_id"],
        "correct":    bool(r["correct"]),
        "confidence": float(r["confidence"]),
    })
    img_url = f"http://images.cocodataset.org/val2014/COCO_val2014_{int(r['image_id']):012d}.jpg"
    output["attention_data"][f"{r['model']}__{r['sample_id']}"] = {
        "image_url":   img_url,
        "question":    r["question"],
        "ground_truth": r["ground_truth"],
        "prediction":  r["prediction"],
        "weights":     r["weights"],
    }

# text-bias view: per sample, accuracy with vs without image (BLIP2).
blip2_by_sample = {r["sample_id"]: r for r in all_results if r["model"] == "BLIP2"}
for sid, tb in text_bias_index.items():
    if sid not in blip2_by_sample: continue
    output["text_bias"].append({
        "sample_id":    sid,
        "question":     blip2_by_sample[sid]["question"],
        "ground_truth": blip2_by_sample[sid]["ground_truth"],
        "text_only_pred":   tb["prediction"],
        "text_only_correct": bool(tb["correct"]),
        "with_image_pred":   blip2_by_sample[sid]["prediction"],
        "with_image_correct": bool(blip2_by_sample[sid]["correct"]),
    })

with open(args.out, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[done] saved -> {args.out}")
print(f"        matrix_data    = {len(output['matrix_data'])}")
print(f"        attention_data = {len(output['attention_data'])}")
print(f"        text_bias      = {len(output['text_bias'])}")
