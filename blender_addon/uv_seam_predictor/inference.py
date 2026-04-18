import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass


@dataclass
class InferenceJob:
    temp_dir: str
    obj_path: str
    json_path: str
    stdout_path: str
    stderr_path: str
    process: subprocess.Popen
    start_time: float
    timeout_sec: int
    _stdout_handle: object = None
    _stderr_handle: object = None


def create_temp_work_files():
    temp_dir = tempfile.mkdtemp(prefix='uv_seam_predictor_')
    return {
        'temp_dir': temp_dir,
        'obj_path': os.path.join(temp_dir, 'mesh.obj'),
        'json_path': os.path.join(temp_dir, 'prediction.json'),
        'stdout_path': os.path.join(temp_dir, 'stdout.log'),
        'stderr_path': os.path.join(temp_dir, 'stderr.log'),
    }


def build_cli_args(prefs, settings, obj_path, json_path):
    return [
        os.path.abspath(prefs.python_executable),
        os.path.abspath(prefs.predict_script_path),
        '--mesh-path',
        os.path.abspath(obj_path),
        '--model-weights',
        os.path.abspath(settings.model_weights_path),
        '--threshold',
        str(settings.threshold),
        '--output-json',
        os.path.abspath(json_path),
    ]


def resolve_process_cwd(script_path):
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if os.path.basename(script_dir) == 'tools':
        return os.path.dirname(script_dir)
    return script_dir


def launch_inference(prefs, settings, paths):
    stdout_handle = open(paths['stdout_path'], 'w', encoding='utf-8')
    stderr_handle = open(paths['stderr_path'], 'w', encoding='utf-8')

    try:
        process = subprocess.Popen(
            build_cli_args(prefs, settings, paths['obj_path'], paths['json_path']),
            cwd=resolve_process_cwd(prefs.predict_script_path),
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise

    return InferenceJob(
        temp_dir=paths['temp_dir'],
        obj_path=paths['obj_path'],
        json_path=paths['json_path'],
        stdout_path=paths['stdout_path'],
        stderr_path=paths['stderr_path'],
        process=process,
        start_time=time.monotonic(),
        timeout_sec=prefs.default_timeout_sec,
        _stdout_handle=stdout_handle,
        _stderr_handle=stderr_handle,
    )


def poll_job(job):
    return job.process.poll()


def has_timed_out(job):
    return time.monotonic() - job.start_time > job.timeout_sec


def terminate_job(job, kill=False):
    if job.process.poll() is None:
        if kill:
            job.process.kill()
        else:
            job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=5)
    close_log_handles(job)


def close_log_handles(job):
    for handle in (job._stdout_handle, job._stderr_handle):
        if handle and not handle.closed:
            handle.close()


def cleanup_job(job, keep_temp_files=False):
    close_log_handles(job)
    if not keep_temp_files and os.path.isdir(job.temp_dir):
        shutil.rmtree(job.temp_dir)


def read_text_tail(path, max_chars=4000):
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        data = file.read()
    return data[-max_chars:].strip()
