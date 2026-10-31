import copy
import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

spec = importlib.util.spec_from_file_location('staged', Path(__file__).resolve().parents[1]/'scripts/train_v8_staged_baseline.py')
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


class StagedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_schedule_boundaries(self):
        self.assertEqual([staged.stage_for_epoch(e) for e in (1,20,21,60,61,200)], [1,1,2,2,3,3])

    def test_freeze_bn_updates_and_optimizer_resume(self):
        model = staged.make_model(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x, y = torch.randn(2,3,64,64), torch.tensor([0,1])
        for stage, lr in ((1,1e-3),(2,1e-4),(3,1e-5)):
            staged.configure_stage(model,optimizer,stage)
            self.assertEqual(optimizer.param_groups[0]['lr'],lr)
            before = {n:v.clone() for n,v in model.state_dict().items()}
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x),y).backward()
            optimizer.step()
            for name, parameter in model.named_parameters():
                active = stage==3 or name.startswith('fc.') or (stage==2 and name.startswith(('layer3.','layer4.')))
                self.assertEqual(parameter.requires_grad,active)
                if not active:
                    self.assertIsNone(parameter.grad)
            if stage < 3:
                for name, value in model.state_dict().items():
                    if name.startswith(('conv1.','bn1.','layer1.','layer2.')) or (stage==1 and name.startswith(('layer3.','layer4.'))):
                        self.assertTrue(torch.equal(value,before[name]), name)
            self.assertFalse(torch.equal(model.fc[1].weight,before['fc.1.weight']))
            # Reloading preserves optimizer state, including parameters newly unfrozen.
            clone = staged.make_model(False)
            clone.load_state_dict(model.state_dict())
            opt = torch.optim.Adam(clone.parameters())
            opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            self.assertEqual(len(opt.state),len(optimizer.state))
            staged.configure_stage(clone,opt,stage)
            model.eval(); clone.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x),clone(x)))

    def test_training_only_weights_with_smoothing(self):
        loss, meta = staged.make_loss([0]*292+[1]*490,'cpu')
        self.assertAlmostEqual(meta['weights'][0],782/(2*292),places=6)
        self.assertAlmostEqual(meta['weights'][1],782/(2*490),places=6)
        x,y=torch.randn(4,2),torch.tensor([0,1,1,0])
        expected=F.cross_entropy(x,y,weight=torch.tensor(meta['weights']),label_smoothing=.1)
        self.assertTrue(torch.equal(loss(x,y),expected))

    def test_sidecars_ignored_but_real_extra_image_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[]
            for split in ('train','val','test'):
                p=root/split/'standard/images/0_normal'/f'{split}.jpg'
                p.parent.mkdir(parents=True)
                Image.new('RGB',(16,16)).save(p)
                p.with_name('._'+p.name).write_bytes(b'AppleDouble metadata')
                rows.append(dict(split=split,modality='standard',label=0,image_relpath=p.relative_to(root).as_posix(),
                    image_id=split,group_id=split,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            manifest=root/'dataset.json'
            manifest.write_text(json.dumps(dict(schema_version=1,task='image_classification',class_names=staged.CLASS_NAMES,records=rows)))
            self.assertEqual(staged.validate_handoff(manifest)['images'],3)
            Image.new('RGB',(16,16)).save(p.with_name('unexpected.jpg'))
            with self.assertRaisesRegex(ValueError,'inventory differs'):
                staged.validate_handoff(manifest)


if __name__ == '__main__':
    unittest.main()
