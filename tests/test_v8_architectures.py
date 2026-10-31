import sys
import unittest
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v8_architecture_support import build_classifier, configure_stage, HEADS, LAST_STAGES


class ArchitectureTests(unittest.TestCase):
    def test_freeze_forward_backward_and_bn_per_stage(self):
        torch.set_num_threads(2)
        for name in ('resnet50','efficientnet_b3','densenet121'):
            with self.subTest(model=name):
                model=build_classifier(name,False)
                optimizer=torch.optim.Adam(model.parameters(),weight_decay=1e-4)
                self.assertEqual(model.get_submodule(HEADS[name])[0].p,.5)
                x,y=torch.randn(2,3,64,64),torch.tensor([0,1])
                for stage in (1,2,3):
                    count=configure_stage(model,optimizer,name,stage)
                    self.assertGreater(count,0)
                    frozen={k:v.clone() for k,v in model.state_dict().items()
                            if k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked')}
                    logits=model(x)
                    self.assertEqual(tuple(logits.shape),(2,2))
                    nn.functional.cross_entropy(logits,y).backward()
                    for key,p in model.named_parameters():
                        prefixes=(HEADS[name],)+(LAST_STAGES[name] if stage==2 else ())
                        expected=stage==3 or any(key.startswith(prefix+'.') for prefix in prefixes)
                        self.assertEqual(p.requires_grad,expected,key)
                        self.assertEqual(p.grad is not None,expected,key)
                    for key,value in frozen.items():
                        module=model.get_submodule(key.rsplit('.',1)[0])
                        if not module.training:
                            self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                clone=build_classifier(name,False)
                clone.load_state_dict(model.state_dict())
                model.eval();clone.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(x),clone(x)))


if __name__=='__main__':
    unittest.main()
