"""CPU-only: verify the 'glm' chat template builds a correct loss-mask on a real regen sample."""
import json
from transformers import AutoTokenizer
from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
REGEN = "/mnt/18tb_r1/dspark-data/regen/perfectblend_regen.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
parser = GeneralParser(tokenizer=tok, chat_template=TEMPLATE_REGISTRY.get("glm"))

n_ok = 0
with open(REGEN) as f:
    for li, line in enumerate(f):
        if li >= 3:
            break
        d = json.loads(line)
        if d.get("status") != "success":
            continue
        conv = d["conversations"]
        out = parser.parse(conv, max_length=8192)
        ids, lm = out["input_ids"], out["loss_mask"]
        masked = [int(ids[i]) for i in range(len(ids)) if lm[i] == 1]
        unmasked = [int(ids[i]) for i in range(len(ids)) if lm[i] == 0]
        print(f"\n--- sample {li}: {len(ids)} tok, {int(lm.sum())} loss tok ({100*int(lm.sum())/len(ids):.0f}%) ---")
        print("  MASKED (assistant, should be the answer) head:", repr(tok.decode(masked)[:140]))
        print("  MASKED tail:", repr(tok.decode(masked)[-80:]))
        print("  UNMASKED (prompt) tail:", repr(tok.decode(unmasked)[-120:]))
        # sanity: loss should be a contiguous-ish block near the end, not 0 and not ~all
        frac = int(lm.sum()) / len(ids)
        ok = 0.05 < frac < 0.97 and "<|assistant|>" not in tok.decode(masked)
        print("  CHECK:", "✅ looks right" if ok else "⚠️ inspect")
        n_ok += ok
print(f"\n{n_ok}/3 samples passed the loss-mask sanity check")
