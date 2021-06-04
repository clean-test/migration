# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re

from . import base

# TODO: tests
# TODO: lifiting of literls


def load_handlers():
    return [
        base.ReFilterHandler({re.compile('^#include\s+[<"]boost/test')}),
        base.FilterHandler(forbidden={"#define BOOST_TEST_MAIN"}),
    ]
