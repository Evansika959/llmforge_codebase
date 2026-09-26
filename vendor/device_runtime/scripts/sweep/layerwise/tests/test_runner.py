import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.sweep.layerwise.database import ROOT, prepare
from scripts.sweep.layerwise.tests import test_candidates as fixtures


class RunnerTests(unittest.TestCase):
    def test_live_lock_does_not_clobber_state(self):
        from scripts.sweep.layerwise.runner import run
        from scripts.prediction.active_learning.storage import locked
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'pool'; prepare(fixtures.SPEC,folder)
            saved='{"status":"measuring","pid":123}\n'
            (folder/'run_state.json').write_text(saved)
            with locked(folder/'RUNNING.lock'):
                self.assertEqual(run(folder,'not-a-real-device',1,True),2)
            self.assertEqual((folder/'run_state.json').read_text(),saved)

    @unittest.skipUnless(shutil.which('cc'),'Host compiler unavailable')
    def test_wrapper_workload_and_battery_guard(self):
        import torch
        from scripts.sweep.layerwise.export import export_mock
        from scripts.sweep.layerwise.measurement import parse
        from scripts.sweep.run_sweep_configs import build_prompt_for_tokens,find_tokenizer_gpt2
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); executable=folder/'lw_measure'
            subprocess.run(['cc','-O2','-pthread','-I'+str(ROOT/'src'),str(ROOT/'scripts/sweep/layerwise/measure.c'),
                            '-lm','-o',str(executable)],capture_output=True,check=True,timeout=60)
            for name,value in [('capacity','100'),('status','Discharging'),('current_now','-100000'),('voltage_now','4000000'),('temp','28000')]:
                (folder/name).write_text(value+'\n')
            arch=fixtures.ExportTests.architecture(64); arch['operator_profile'].update(vocab_size=50257,seq_len=256)
            model=folder/'model.rlm'; export_mock(arch,model)
            command=[str(executable),str(model),str(ROOT/find_tokenizer_gpt2()),build_prompt_for_tokens(48),str(folder),str(folder/'temp'),str(folder/'trace.csv')]
            result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30)
            (folder/'timing.json').write_text(result.stdout)
            measured=parse(folder)
            self.assertGreater(measured['decode_tok_s'],0)
            self.assertEqual(measured['timing']['decode_tokens'],31)
            (folder/'capacity').write_text('30\n')
            stopped=subprocess.run(command,capture_output=True,text=True,timeout=5)
            self.assertEqual(stopped.returncode,75)
            self.assertIn('BATTERY_STOP',stopped.stderr)


if __name__=='__main__': unittest.main()
