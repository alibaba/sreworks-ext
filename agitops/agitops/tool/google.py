#!/usr/bin/env python3

import os
import sys
import json

from agitops.tool import argparse
import urllib.request

google_api_key = os.getenv("google_api_cx")
google_api_cx = os.getenv("google_api_cx")

def googleapi(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_api_cx}&q={query}"

    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(json.dumps(data, indent=4, ensure_ascii=False))
            else:
                print(f"请求失败，状态码: {response.status}")
    except urllib.error.URLError as e:
        print(f"请求出错: {e.reason}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="google搜索工具")
    subparsers = parser.add_subparsers(dest="command")

    go_parser = subparsers.add_parser("search", help="进行google搜索")
    go_parser.add_argument("-q", "--query", help="搜索关键词传入", required=True)

    args = parser.parse_args()

    if args.command == "search":
        googleapi(args.query)
    else:
        subparsers.dump_schema()