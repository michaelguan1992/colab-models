"""python -m controller --config models.yaml [--port 9000]"""

import argparse

import uvicorn

from .app import create_app
from .manager import Manager
from .registry import Registry, RegistryError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="models.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--load", default=None, help="model id to load at startup")
    args = p.parse_args()

    try:
        registry = Registry.load(args.config)
    except RegistryError as e:
        raise SystemExit(f"{args.config} 有误: {e}")
    manager = Manager(registry)
    app = create_app(manager, autoload=args.load)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
