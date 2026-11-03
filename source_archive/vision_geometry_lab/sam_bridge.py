"""Run installed SAM2 lazily, without changing either existing environment."""
from pathlib import Path
import json
import subprocess
import tempfile

from .geometry import write_image, read_image


def sam_rope_proposal(image, points, labels=None, roi=None):
    root = Path(__file__).resolve().parents[1]
    python = root/'vision_training/.venv/Scripts/python.exe'
    if not python.is_file():
        raise RuntimeError('SAM2 Python environment is missing.')
    with tempfile.TemporaryDirectory(prefix='rope_sam2_') as folder:
        folder = Path(folder)
        write_image(folder/'frame.png', image)
        request = dict(image=str(folder/'frame.png'), points=points, labels=labels,
                       roi=roi, mask_output=str(folder/'mask.png'),
                       metadata_output=str(folder/'metadata.json'))
        if labels is None:
            request.pop('labels')
        (folder/'request.json').write_text(json.dumps(request), encoding='utf-8')
        run = subprocess.run([str(python), str(Path(__file__).with_name('sam_worker.py')),
                              str(folder/'request.json')], capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=180,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if run.returncode:
            raise RuntimeError('SAM2辅助失败：\n'+run.stderr[-1800:])
        mask = read_image(folder/'mask.png')[:, :, 0] > 0
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        return mask, metadata
