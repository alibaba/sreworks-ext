#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import paramiko

class BaseFault:
    def __init__(self, ip, username, password):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(ip, username=username, password=password)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--action", required=True, choices=["inject", "recover"], help="执行故障注入或故障恢复")
        parser.add_argument("--ip", "--host",required=True, help="目标地址 可以为主机名或IP")
        parser.add_argument("--username", "-l", required=True, help="SSH 用户名")
        parser.add_argument("--password", required=False, help="SSH 密码")

    def execute_command(self, command):
        stdin, stdout, stderr = self.ssh_client.exec_command(command)

        ret_code  = stdout.channel.recv_exit_status()
        output_stdout = stdout.read().decode('utf-8')
        output_stderr = stderr.read().decode('utf-8')

        stdin = None
        stdout = None
        stderr = None

        return (ret_code, output_stdout, output_stderr)

    def close_connection(self):
        self.ssh_client.close()

class DiskFault(BaseFault):
    description = "填满指定主机上的所有硬盘"

    @staticmethod
    def add_args(parser):
        BaseFault.add_args(parser)
        parser.add_argument("--target", required=False, default="notsystem", choices=["system", "notsystem", "all"], help="指定填满的目标 system | notsystem | all")

    def pasrse_df(self, text):
        lines = text.split("\n")
        keys = lines[0].split()
        return [dict(zip(keys, line.split())) for line in lines[1:]]

    def get_df(self, ip):
        print(f"Get the current disk capacity and status. host {ip}")
        retCode, stdout, stderr = self.execute_command("df")
        print(stdout)
        return stdout
    
    def get_disks(self, df_data, target):
        mounts = []
        for disk in df_data:
            if disk.get('Mounted') == "/" and target in ["system", "all"]:
                mounts.append(disk)
            elif disk.get('Filesystem','').startswith("/dev") and disk.get('Mounted') != "/" and target in ["notsystem", "all"]:
                mounts.append(disk)
        return mounts

    def inject(self, args):
        for disk in self.get_disks(self.pasrse_df(self.get_df(args.ip)), args.target):
            print(f"目录 {disk['Mounted']} 开始填充")
            command = f"fallocate -l {disk['Available']}k {disk['Mounted']}/fault_file"
            print(">>> " + command)
            (ret, stdout, stderr) = self.execute_command(command)
            print(f"ret code: {ret}\nstdout: {stdout}\nstderr: {stderr}\n")

        self.get_df(args.ip)
        
    def recover(self, args):
        for disk in self.get_disks(self.pasrse_df(self.get_df(args.ip)), args.target):
            print(f"目录 {disk['Mounted']} 开始释放")
            command = f"rm -f {disk['Mounted']}/fault_file"
            print(">>> " + command)
            (ret, stdout, stderr) = self.execute_command(command)
            print(f"ret code: {ret}\nstdout: {stdout}\nstderr: {stderr}\n")

        self.get_df(args.ip)


# 可以按照 DiskFault 类的模式定义更多故障类型
FAULT_TYPES = {
    "disk_full": DiskFault
}

def main():
    parser = argparse.ArgumentParser(description="故障注入工具")
    
    subparsers = parser.add_subparsers(title="故障类型", dest="type")

    # 对每个故障类型创建子解析器，并调用 add_args 方法
    for fault_name, fault_class in FAULT_TYPES.items():
        subparser = subparsers.add_parser(fault_name, help=fault_class.description)
        fault_class.add_args(subparser)
 
    args = parser.parse_args()
    fault_type = FAULT_TYPES[args.type](args.ip, args.username, args.password)

    if args.action == "inject":
        fault_type.inject(args)
    elif args.action == "recover":
        fault_type.recover(args)

    fault_type.close_connection()


if __name__ == "__main__":
    main()

