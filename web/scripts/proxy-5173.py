#!/usr/bin/env python3
"""IPv4 proxy so 127.0.0.1:5173 still reaches the app on 8501."""

from __future__ import annotations

import select
import socket
import threading

LISTEN = ("0.0.0.0", 5173)
DEST = ("127.0.0.1", 8501)


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            ready, _, _ = select.select([a, b], [], [], 60)
            if not ready:
                continue
            for src in ready:
                data = src.recv(65536)
                if not data:
                    return
                (b if src is a else a).sendall(data)
    except OSError:
        return
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(LISTEN)
    server.listen(64)
    print(f"proxy {LISTEN[0]}:{LISTEN[1]} -> {DEST[0]}:{DEST[1]}", flush=True)
    while True:
        client, _ = server.accept()
        try:
            dest = socket.create_connection(DEST, timeout=5)
        except OSError:
            client.close()
            continue
        threading.Thread(target=pipe, args=(client, dest), daemon=True).start()


if __name__ == "__main__":
    main()
