#! /usr/bin/env python3
import argparse
import contextlib
import getopt
import http.server
import os
import runpy
import sys
import tempfile

from megengine.logger import get_logger


def main():
    parser = argparse.ArgumentParser(
        prog="megengine.tools.svg_viewer",
        description="View SVG Graph produced bt megengine profiler",
    )
    parser.add_argument("-p", "--port", type=int, default=8000, help="server port")
    parser.add_argument(
        "-a", "--address", type=str, default="localhost", help="server address"
    )
    args = parser.parse_args()
    address = args.address
    port = args.port
    src_filename = "svg_viewer.html"
    dst_filename = "index.html"
    src_path = os.path.join(os.path.dirname(__file__), src_filename)
    url = "http://{}:{}/{}".format("localhost", port, dst_filename)
    ssh_fwd_cmd = "ssh -L {}:localhost:{} <remote ip>".format(port, port)
    with tempfile.TemporaryDirectory() as serve_dir:
        dst_path = os.path.join(serve_dir, dst_filename)
        os.symlink(src_path, dst_path)
        os.chdir(serve_dir)
        get_logger().info("cd to serve directory: {}, starting".format(serve_dir))
        server = http.server.HTTPServer(
            (address, port), http.server.SimpleHTTPRequestHandler
        )
        get_logger().info(
            "server started, please visit '{}' to watch profiling result".format(url)
        )
        get_logger().info(
            "if you are in remote environment, use '{}' to forward port to local".format(
                ssh_fwd_cmd
            )
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            get_logger().info("server exiting")


if __name__ == "__main__":
    main()
