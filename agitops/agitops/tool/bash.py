#!/usr/bin/env python3

import os
import sys
import json

from agitops.tool import argparse
import subprocess


def exec_shell(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    stdout, stderr = process.communicate()
    return_code = process.returncode

    if stdout:
        sys.stdout.write(stdout)

    if stderr:
        sys.stderr.write(stderr)

    sys.exit(return_code)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="命令行工具")
    subparsers = parser.add_subparsers(dest="command")

    sh_parser = subparsers.add_parser("exec", help="执行shell命令")
    sh_parser.add_argument("-s", "--shell", help="shell命令传入", required=True)

    args = parser.parse_args()

    if args.command == "exec":
        exec_shell(args.shell)
    else:
        subparsers.dump_schema()