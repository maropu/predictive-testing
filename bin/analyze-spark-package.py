#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Analyze Spark repository to extract build dependency and test list
"""

import json
import glob
import os
import re
from typing import Any
from typing import Any, Dict, List, Optional

from ptesting import depgraph
from ptesting import javaclass


# Compiled regex patterns for Spark repository
# TODO: Generalize this
RE_IS_JAVA_CLASS = re.compile('\/[a-zA-Z0-9/\-_]+[\.class|\$]')
RE_IS_JAVA_TEST = re.compile('\/[a-zA-Z0-9/\-_]+Suite\.class$')
RE_IS_PYTHON_TEST = re.compile('\/test_[a-zA-Z0-9/\-_]+\.py$')
RE_JAVA_CLASS_PATH = re.compile("[a-zA-Z0-9/\-_]+\/classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+)[\.class|\$]")
RE_JAVA_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/test-classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+Suite)\.class$")
RE_PYTHON_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/python\/(pyspark\/[a-zA-Z0-9/\-_]+)\.py$")


def _is_java_class(path: str) -> bool:
    return RE_IS_JAVA_CLASS.search(path) is not None and \
        RE_IS_JAVA_TEST.search(path) is None


def _is_java_test(path: str) -> bool:
    return RE_IS_JAVA_TEST.search(path) is not None


def _is_python_test(path: str) -> bool:
    return RE_IS_PYTHON_TEST.search(path) is not None


def _extract_package(path: str, regex: Any) -> Optional[str]:
    result = regex.search(path)
    if result:
        return result.group(1).replace('/', '.')
    else:
        return None


def _enumerate_java_classes(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_java_class(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_JAVA_CLASS_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_java_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_java_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_JAVA_TEST_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_python_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_python_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_PYTHON_TEST_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_spark_files(root_path: str) -> Any:
    files = [p for p in glob.glob(f'{root_path}/**', recursive=True)]
    java_classes = _enumerate_java_classes(files, root_path)
    java_tests = _enumerate_java_tests(files, root_path)
    python_tests = _enumerate_python_tests(files, root_path)
    return java_classes, java_tests, python_tests


def _format_path(path: str, root_path: str) -> str:
    prefix_path = os.path.abspath(root_path)
    return path[len(prefix_path) + 1:]


def _write_data_as(prefix: str, path: str, data: Any) -> None:
    with open(f"{path}/{prefix}.json", mode='w') as f:
        f.write(json.dumps(data, indent=2))


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--root-path', dest='root_path', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()

    if args.overwrite:
        import shutil
        shutil.rmtree(args.output, ignore_errors=True)

    # Make an output dir in advance
    os.mkdir(args.output)

    java_classes, java_tests, python_tests = _enumerate_spark_files(args.root_path)
    tests = dict(map(lambda t: (t[0], _format_path(t[1], args.root_path)), ({**java_tests, **python_tests}).items()))
    _write_data_as('test-files', args.output, tests)

    java_files = list(({**java_classes, **java_tests}).items())
    extract_edges_from_path = javaclass.create_func_to_extract_refs_from_class_file('org.apache.spark')
    # dep_graph = depgraph.build_dependency_graphs(java_files[0:100], extract_edges_from_path)
    dep_graph = depgraph.build_dependency_graphs(java_files, extract_edges_from_path)
    _write_data_as('dep-graph', args.output, dep_graph)


if __name__ == "__main__":
    main()
