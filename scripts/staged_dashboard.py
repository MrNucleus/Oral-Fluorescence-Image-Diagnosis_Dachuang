"""Read-only local dashboard for the explicitly selected v8 staged training run."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT/'experiments/oral_v7_resnet18_baseline_20261026/code/outputs/image_classifier_runs'


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def status():
    pointer = read_json(ROOT/'reports/_qa/v8_staged_active.json', {})
    if not pointer:
        return dict(status='waiting', history=[], live={}, message='等待v8三阶段实验启动')
    run = Path(pointer['run']).resolve()
    if not run.is_relative_to(OUTPUTS.resolve()):
        raise ValueError('Run outside training outputs')
    config = read_json(run/'run_config.json', {})
    if config.get('identity',{}).get('protocol',{}).get('version') not in ('v8-section4.3-v1','v8-section4.3-architectures-v1'):
        raise ValueError('Not a v8 staged experiment')
    history = read_json(run/'history.json', [])
    live = read_json(run/'live_progress.json', {})
    state = read_json(run/'continue_state.json', {})
    alive = False
    try:
        proc = psutil.Process(pointer['pid'])
        alive = abs(proc.create_time()-pointer['process_created']) < .1 and proc.is_running()
    except (psutil.Error, KeyError):
        pass
    phase = 'complete' if state.get('status')=='complete' else 'running' if alive else 'stopped'
    queue_pointer=read_json(ROOT/'reports/_qa/v8_architecture_queue.json',{})
    queue={}
    if queue_pointer:
        report=Path(queue_pointer['report']).resolve()
        if report.is_relative_to((ROOT/'reports/v8_architectures').resolve()):
            queue=read_json(report/'queue.json',{})
    return dict(status=phase, process_alive=alive, run=run.name, history=history, queue=queue,
        live=live, state=state, config=config,
        best=min(history,key=lambda r:r['val_metrics']['loss']) if history else None)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/api/status':
                payload=json.dumps(status(),ensure_ascii=False,allow_nan=False).encode('utf-8')
                mime='application/json; charset=utf-8'
            elif self.path in ('/','/index.html'):
                payload=(ROOT/'scripts/staged_dashboard.html').read_bytes()
                mime='text/html; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(payload)
        except (ValueError, KeyError) as error:
            self.send_error(500,str(error))

    def log_message(self,*args):
        pass


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=13949)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'V8 staged dashboard: http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()
