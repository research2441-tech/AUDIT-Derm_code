"""Decode the ISIC thumbnail harvest: 8-byte LE header length, JSON index, then JPEG blob."""
import json, struct, numpy as np, pandas as pd
import os
SRC = os.environ.get('ISIC_BUNDLE', 'shiftderm_isic_thumbs.bin')
def load_index(path=SRC):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        idx = json.loads(f.read(n).decode('utf-8'))
    return idx, 8 + n
def read_image_bytes(path, base, rec):
    with open(path, 'rb') as f:
        f.seek(base + rec['off']); return f.read(rec['len'])
if __name__ == '__main__':
    idx, base = load_index()
    rows = []
    for r in idx:
        md = r.get('md', {}); cl = md.get('clinical', {}); ac = md.get('acquisition', {})
        rows.append(dict(id=r['id'], coll=r['coll'], attr=r.get('attr'), image_type=ac.get('image_type'),
                         dx1=cl.get('diagnosis_1'), dx2=cl.get('diagnosis_2'), dx3=cl.get('diagnosis_3'),
                         confirm=cl.get('diagnosis_confirm_type'), lesion_id=cl.get('lesion_id'),
                         patient_id=cl.get('patient_id'), age=cl.get('age_approx'), sex=cl.get('sex'),
                         site=cl.get('anatom_site_general', cl.get('anatom_site_1')),
                         fitz=cl.get('fitzpatrick_skin_type'), off=r['off'], len=r['len']))
    df = pd.DataFrame(rows); df.to_csv('index.csv', index=False)
    print(len(df), 'base', base)
    print(df.groupby(['coll', 'attr', 'image_type']).size().to_string())
    print(df.dx1.value_counts().to_string())
