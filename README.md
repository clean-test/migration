[//]: # (Copyright m8mble 2022)
[//]: # (SPDX-License-Identifier: BSL-1.0)

# Clean Test Migration

[![Tests][badge-tests]][link-tests]
![python][badge-python]
![Codestyle: Black][badge-black]
[![License: BSL-1.0][badge-BSL]](LICENSE)

Clean Test is a modern test framework for C++.
This accompanying project provides utilities for migrating existing tests to Clean Test.

We enable conversion of standard test registration and assertion macros.
Most importantly this includes a lifting utility that prepares any expression for `ct::expect` with proper output.

### Demo

```cpp
BOOST_TEST(f() == 0 and 1 == g());
```
will be converted to
```cpp
ct::expect(f() == 0_i and 1_i == g());
```
Use of user-defined literals as well as the `clean_test`-namespace alias are configurable.

## Usage

You can install from source via
```py
poetry install
```

This enables
```py
poetry run migrate -h
```
which explains how to use this tool and elaborates all available options.


## Disclaimer

Existing unit-test frameworks differ significantly from Clean Test.
This is why certain constructs only work with or without Clean Test (e.g. fixtures).
We don't aim for migrating everything perfectly, but rather for automating the tedious bits of refactoring.
Consequently the migrated tests might need small manual adaptations in order to compile.

We currently only support migration from boost-test.
Other unit-test frameworks will be added in the future based on the lifting utilities.


[badge-black]: https://img.shields.io/badge/code%20style-black-000000.svg
[badge-BSL]: https://img.shields.io/badge/license-BSL--1.0-informational
[badge-tests]: https://github.com/clean-test/migration/actions/workflows/test.yml/badge.svg
[badge-python]: https://img.shields.io/badge/python-3.9%20%7C%203.10-informational
[link-tests]: https://github.com/clean-test/clean-test/actions/workflows/test.yml
