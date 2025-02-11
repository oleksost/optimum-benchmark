import os
import traceback
from contextlib import ExitStack
from logging import Logger
from multiprocessing import Pipe, Process, get_start_method, set_start_method
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import psutil
import json

from ...benchmark.report import BenchmarkReport
from ...logging_utils import setup_logging
from ...process_utils import sync_with_child, sync_with_parent
from ..base import Launcher
from .config import ProcessConfig


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
            set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        child_connection, parent_connection = Pipe()
        main_process_pid = os.getpid()
        isolated_process = Process(
            target=target, args=(worker, worker_args, child_connection, main_process_pid, self.logger), daemon=False
        )

        with ExitStack() as stack:
            if self.config.numactl:
                stack.enter_context(self.numactl_executable())

            isolated_process.start()

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            if self.config.device_isolation:
                stack.enter_context(self.device_isolation(isolated_process.pid))

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response = parent_connection.recv()
        else:
            raise RuntimeError("Received no response from isolated process")

        if "traceback" in response:
            self.logger.error("\t+ Received traceback from isolated process")
            raise ChildProcessError(response["traceback"])
        elif "exception" in response:
            self.logger.error("\t+ Received exception from isolated process")
            raise ChildProcessError(response["exception"])
        elif "chunks" in response:
            self.logger.info(f"\t+ Receiving chunked report from isolated process ({response['chunks']} chunks)")
            chunks = []
            for _ in range(response["chunks"]):
                if parent_connection.poll(timeout=60):  # 60 second timeout
                    chunk_response = parent_connection.recv()
                    if "chunk" in chunk_response:
                        chunks.append(chunk_response["chunk"])
                    else:
                        raise RuntimeError(f"Received unexpected chunk response: {chunk_response}")
                else:
                    raise RuntimeError("Timeout waiting for report chunk")
            
            try:
                report_dict = json.loads("".join(chunks))
                report = BenchmarkReport.from_dict(report_dict)
                self.logger.info("\t+ Successfully reconstructed report from chunks")
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct report from chunks: {str(e)}")
        elif "report" in response:
            self.logger.info("\t+ Received report from isolated process")
            report = BenchmarkReport.from_dict(response["report"])
        else:
            raise RuntimeError(f"Received an unexpected response from isolated process: {response}")

        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    child_connection: Connection,
    main_process_pid: int,
    logger: Logger,
) -> None:
    main_process = psutil.Process(main_process_pid)

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Sending traceback to main process")
        child_connection.send({"traceback": traceback.format_exc()})
    else:
        report_size = len(str(report.to_dict()))
        logger.info(f"\t+ Report size: {report_size} bytes")
        logger.info("\t+ Sending report to main process")
        if report_size > 3e5: # 300kb
            send_chunked_report(child_connection, report.to_dict(), logger, chunk_size=100000)
        else:
            child_connection.send({"report": report.to_dict()})
    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
        exit(0)


def send_chunked_report(connection: Connection, report_dict: dict, logger: Logger, chunk_size: int = 1_000_000) -> None:
    """Send large report in chunks to avoid pipe buffer issues.
    
    Args:
        connection: The connection to send chunks through
        report_dict: The report dictionary to send
        logger: Logger for debugging
        chunk_size: Maximum size of each chunk in bytes
    """
    try:
        report_str = json.dumps(report_dict)
        chunks = [report_str[i:i + chunk_size] for i in range(0, len(report_str), chunk_size)]
        
        logger.info(f"\t+ Sending report in {len(chunks)} chunks")
        connection.send({"chunks": len(chunks)})
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"\t+ Sending chunk {i}/{len(chunks)}")
            connection.send({"chunk": chunk})
            
        logger.info("\t+ Finished sending all chunks")
    except Exception as e:
        logger.error(f"\t+ Error while sending chunked report: {str(e)}")
        connection.send({"exception": str(e)})