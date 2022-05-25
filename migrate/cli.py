# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import argparse
import pathlib

from . import log, migrate


def _load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", help="Files to migrate.")
    parser.add_argument(
        "--use-namespace-alias", action="store_true", help="Introduce ct namespace alias for clean_test."
    )
    parser.add_argument(
        "--use-literals", action="store_true", help="Prefer namespace literals over lifting expressions."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increse log output verbosity. Can be specified multiple times.",
    )
    return parser


def _parse_commandline():
    result = vars(_load_parser().parse_args())
    result["files"] = [pathlib.Path(f) for f in result["files"]]
    result["namespace"] = "ct" if result["use_namespace_alias"] else "clean_test"
    return result


def main():
    args = _parse_commandline()
    log.setup(verbosity=args["verbose"])
    handlers = migrate.load_handlers(**args)
    for f in args.pop("files"):
        log.notice(f"Migrating {f} ...")
        migrate.convert(path=f, handlers=handlers, **args)
