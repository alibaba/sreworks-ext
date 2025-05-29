import subprocess

def run_command(cmd, cwd, envs={}):
    if type(cmd) is list:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                                   env=envs)
    elif type(cmd) is str:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=cwd,
                                   env=envs)
    else:
        raise ValueError("Invalid command type")
    stdout, stderr = process.communicate()
    return [process.returncode, stdout, stderr]