from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

URL = 'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def download(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'manifest.json').exists():
        raise ValueError('Use a new data directory; the existing snapshot is preserved')
    archive = folder/'optdigits.zip'
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for name in ['optdigits.tra', 'optdigits.tes', 'optdigits.names']:
            (folder/name).write_bytes(z.read(name))
    manifest = {'source': URL, 'archive_sha256': sha(archive),
                'files': {n: sha(folder/n) for n in ['optdigits.tra','optdigits.tes','optdigits.names']},
                'license': 'CC BY 4.0', 'doi': '10.24432/C50P49'}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

def parse(path):
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    if a.shape[1] != 65 or not len(a) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('Expected rows of 64 integer pixel counts and a digit label')
    if a[:,:64].min() < 0 or a[:,:64].max() > 16 or a[:,64].min() < 0 or a[:,64].max() > 9:
        raise ValueError('Pixel or label outside allowed range')
    return a[:,:64].astype(np.uint8), a[:,64].astype(int)

def partitions(y, seed=42):
    ids = np.arange(len(y))
    train, remaining = train_test_split(ids, test_size=.4, stratify=y, random_state=seed)
    validation, calibration = train_test_split(remaining, test_size=.5, stratify=y[remaining], random_state=seed)
    return {'train': train, 'validation': validation, 'calibration': calibration}

def load(folder):
    folder = Path(folder)
    source = json.loads((folder/'manifest.json').read_text())
    for name, expected in source['files'].items():
        if sha(folder/name) != expected: raise ValueError('Data fingerprint mismatch')
    X, y = parse(folder/'optdigits.tra'); Xt, yt = parse(folder/'optdigits.tes')
    if len(X) != 3823 or len(Xt) != 1797: raise ValueError('Unexpected official split sizes')
    hashes = [{hashlib.sha256(row.tobytes()).hexdigest() for row in a} for a in [X, Xt]]
    audit = {'train_exact_duplicates':len(X)-len(hashes[0]),'test_exact_duplicates':len(Xt)-len(hashes[1]),
             'cross_file_duplicate_images':len(hashes[0] & hashes[1])}
    if any(audit.values()): raise ValueError('Duplicate images require an explicit split review')
    return X, y, Xt, yt, source, audit
