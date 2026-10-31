import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import resilient_training_io as io


class FileIOTests(unittest.TestCase):
    def test_transient_lock_retried_and_no_shared_temporary(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'history.json'
            original=io.os.replace
            attempted=[]
            def replace(source,target):
                attempted.append(source)
                if len(attempted)<3:
                    raise PermissionError('simulated Windows share violation')
                return original(source,target)
            with patch.object(io.os,'replace',replace),patch.object(io.time,'sleep'):
                self.assertTrue(io.write_json(p,{'epoch':14}))
            self.assertEqual(len(attempted),3)
            self.assertEqual(json.loads(p.read_text()),{'epoch':14})
            self.assertNotEqual(attempted[0].name,'history.json.tmp')
            self.assertEqual(list(Path(temp).glob('*.tmp')),[])

    def test_persistent_telemetry_failure_does_not_mask_critical_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            p=root/'live_progress.json';p.write_text('{"epoch":14}')
            with patch.object(io.os,'replace',side_effect=PermissionError('locked')),patch.object(io.time,'sleep'):
                with self.assertWarns(RuntimeWarning):
                    self.assertFalse(io.write_json(p,{'epoch':15}))
                self.assertEqual(json.loads(p.read_text()),{'epoch':14})
                with self.assertRaises(PermissionError):
                    io.write_json(root/'history.json',{'epoch':15})
                with self.assertRaises(PermissionError):
                    io.save_checkpoint({'epoch':15},root/'last.pt')

    def test_checkpoint_roundtrip_and_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            state=torch.get_rng_state().clone()
            p=Path(temp)/'last.pt'
            io.save_checkpoint({'epoch':14,'tensor':torch.tensor([1.,2.])},p)
            loaded=torch.load(p,weights_only=True)
            self.assertEqual(loaded['epoch'],14)
            self.assertTrue(torch.equal(loaded['tensor'],torch.tensor([1.,2.])))
            self.assertTrue(torch.equal(state,torch.get_rng_state()))

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_real_windows_reader_without_delete_sharing(self):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        create.restype=wintypes.HANDLE
        close=kernel.CloseHandle
        close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'live_progress.json';p.write_text('{"epoch":14}')
            handle=create(str(p),0x80000000,0x1,None,3,0x80,None)
            self.assertNotEqual(handle,wintypes.HANDLE(-1).value)
            release=threading.Thread(target=lambda:(time.sleep(.08),close(handle)))
            release.start()
            try:
                self.assertTrue(io.write_json(p,{'epoch':15}))
                self.assertEqual(json.loads(p.read_text()),{'epoch':15})
            finally:
                release.join()


if __name__=='__main__':
    unittest.main()
