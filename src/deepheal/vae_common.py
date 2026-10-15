"""Frozen-model construction and projection helpers for disease and HepG2 profiles."""

from deepheal.paths import ARTICLE_ROOT

from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch

ROOT = ARTICLE_ROOT
PKG = ROOT / 'results/drug_repurposing'
OLD = ROOT / 'data/repurposing'
FIG8 = ROOT / 'data/cptac'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


projection = load_module(ARTICLE_ROOT / "src/deepheal/projection.py", "vae_frozen_projection")
average = load_module(ROOT / 'src/deepheal/models/average_ensemble.py', 'vae_average_original')
supcon = load_module(ROOT / 'src/deepheal/models/supcon_vae.py', 'vae_supcon_original')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def build_model(branch, payload):
    s = payload['source_summary']
    args = dict(input_dim=int(payload['input_dim']), latent_dim=int(s['source_latent_dim']),
                hidden_dims=(int(s['hidden_1']), int(s['hidden_2'])), dropout_rate=float(s['dropout_rate']))
    model = average.AverageVAE(**args) if branch == 'average' else supcon.Model(**args, proj_dim=32)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.to(DEVICE).eval()

def encode(model, values, batch_size=1024):
    model.eval()
    with torch.inference_mode():
        return np.vstack([model.encode(torch.as_tensor(values[i:i+batch_size], device=DEVICE, dtype=torch.float32))[0].cpu().numpy()
                          for i in range(0, len(values), batch_size)])

def rowcos(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a*b, axis=1), denom, out=np.full(len(a), np.nan), where=denom > 1e-12)

def bank_stem(s):
    return average.cache_stem(int(s['source_latent_dim']), int(s['source_run']),
                              (int(s['hidden_1']), int(s['hidden_2'])), float(s['dropout_rate']))
