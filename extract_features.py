"""Extract frozen CLIP image embeddings for every harvested image (CPU, ONNX Runtime)."""
import io, sys, time, numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
from load_harvest import SRC, load_index
MODEL = sys.argv[1] if len(sys.argv) > 1 else __import__('os').environ.get('CLIP_ONNX', 'clip_vit_b32_vision.onnx')
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
def prep(b, size=224):
    im = Image.open(io.BytesIO(b)).convert('RGB'); w, h = im.size; s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - size) // 2, (h - size) // 2
    a = (np.asarray(im.crop((l, t, l + size, t + size)), np.float32) / 255. - MEAN) / STD
    return a.transpose(2, 0, 1)
if __name__ == '__main__':
    df = pd.read_csv('index.csv'); idx, base = load_index()
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    n = len(df) if LIMIT is None else LIMIT; out = np.zeros((n, 512), np.float32); t0 = time.time()
    with open(SRC, 'rb') as f:
        for s in range(0, n, 32):
            batch = []
            for i in range(s, min(n, s + 32)):
                f.seek(base + int(df.off[i])); batch.append(prep(f.read(int(df.len[i]))))
            out[s:s + len(batch)] = sess.run(None, {'pixel_values': np.stack(batch)})[0]
            if s % 1600 == 0: print(s, round(time.time() - t0, 1), flush=True)
    np.save('emb.npy' if LIMIT is None else 'emb_test.npy', out); print('done', time.time() - t0)
