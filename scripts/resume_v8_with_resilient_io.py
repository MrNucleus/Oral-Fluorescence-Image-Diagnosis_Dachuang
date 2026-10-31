"""Resume the unchanged numerical trainer with an explicitly audited I/O adapter."""
import hashlib
import json
import sys
from pathlib import Path
import torch
import train_v8_architecture as trainer
import resilient_training_io as io


def main():
    if '--resume' not in sys.argv:
        raise ValueError('Recovery adapter requires an existing same-v8 last.pt')
    checkpoint=Path(sys.argv[sys.argv.index('--resume')+1]).resolve()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['source_hashes']!=trainer.fingerprints():
        raise ValueError('Numerical trainer source changed; do not bypass source identity')
    patch=dict(scope='file I/O only; training code/protocol/data identities unchanged',
        adapter_sha256=hashlib.sha256(Path(io.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    io.write_json(checkpoint.parent/'io_adapter.json',patch)
    trainer._write_json=io.write_json
    def audited_save(value,path):
        io.save_checkpoint(dict(value,io_adapter=patch),path)
    trainer._atomic_torch_save=audited_save
    print(json.dumps(dict(recovery='I/O adapter applied',completed_epoch=state['epoch'],patch=patch)),flush=True)
    del state
    trainer.main()


if __name__=='__main__':
    main()
