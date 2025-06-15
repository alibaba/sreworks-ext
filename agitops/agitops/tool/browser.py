#!/usr/bin/env python3

import os
import sys
import json
import logging
import asyncio
from contextlib import redirect_stdout
import io

os.environ["ANONYMIZED_TELEMETRY"] = "false"

from agitops.tool import argparse
from browser_use import Agent, BrowserSession
from langchain_openai import ChatOpenAI

def set_log(log_file):

    class FileLoggingHandler(logging.Handler):
        """A logging handler that writes logs to a file."""

        def __init__(self, file_path) -> None:
            """Initialize the logging handler."""
            super().__init__()
            self.log_file_path = file_path
            self.log_file = open(self.log_file_path, mode="a", encoding="utf-8")

        def emit(self, record: logging.LogRecord) -> None:
            """Write the log record to the log file."""
            log_entry = self.format(record)
            self.log_file.write(log_entry + "\n")
            self.log_file.flush()

        def close(self) -> None:
            """Close the log file."""
            if self.log_file:
                self.log_file.close()
            super().close()

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    handler = FileLoggingHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger('browser_use')
    logger.handlers = []
    logger.addHandler(handler)


async def explore(query, log_file, llm_api_key, llm_api_url, llm_model):
    
    set_log(log_file)
    captured_output = io.StringIO()

    with redirect_stdout(captured_output):
        agent = Agent(
            task=query,
            enable_memory=False,
            llm=ChatOpenAI(
              model=llm_model,
              api_key=llm_api_key,
              base_url=llm_api_url,
            ),
            browser_session=BrowserSession(
                headless=True,
                viewport={'width': 1440, 'height': 900},
                args=['--no-sandbox'],
            ),
        )
        result = await agent.run(max_steps=10)
    
    output = {}
    output["final_result"] = result.final_result()
    output["history"] = []

    for h in result.history:
        result = []
        for r in h.result:
            if r.extracted_content:
                result.append({
                    "content": r.extracted_content,
                    "error": r.error,
                })
        output["history"].append({
            "url": h.state.url,
            "result": result,
        })

    print(json.dumps(output, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="浏览器工具")
    subparsers = parser.add_subparsers(dest="command")

    bs_parser = subparsers.add_parser("explore", help="在互联网中探索")
    bs_parser.add_argument("-q", "--query", help="需要探索的问题", required=True)
    bs_parser.add_argument("--log-file", required=True, default="{{call.log_file}}", help="输出日志文件")
    bs_parser.add_argument("--llm-api-key", required=True, default="{{config.llm_api_key}}")
    bs_parser.add_argument("--llm-api-url", required=True, default="{{config.llm_api_url}}")
    bs_parser.add_argument("--llm-model", required=True, default="{{config.llm_model}}")

    args = parser.parse_args()

    if args.command == "explore":
        asyncio.run(explore(args.query, 
                   log_file=args.log_file, 
                   llm_api_key=args.llm_api_key, 
                   llm_api_url=args.llm_api_url, 
                   llm_model=args.llm_model))
    else:
        subparsers.dump_schema()