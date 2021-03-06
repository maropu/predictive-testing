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

import re
from typing import Any, List, Optional, Tuple

# Regex pattern for parsing source paths
RE_PARSE_PATH_PATTERN = "[a-zA-Z0-9/\-]+/(org\/apache\/spark\/.+\/)([a-zA-Z0-9\-]+)\.scala"

# Compiled regex patterns to extract information from a Spark repository
RE_IS_JAVA_CLASS = re.compile('\/[a-zA-Z0-9/\-_]+[\.class|\$]')
RE_IS_JAVA_TEST = re.compile('\/[a-zA-Z0-9/\-_]+Suite\.class$')
RE_IS_PYTHON_TEST = re.compile('\/test_[a-zA-Z0-9/\-_]+\.py$')
RE_JAVA_CLASS_PATH = re.compile("[a-zA-Z0-9/\-_]+\/classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+)[\.class|\$]")
RE_JAVA_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/test-classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+Suite)\.class$")
RE_PYTHON_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/python\/(pyspark\/[a-zA-Z0-9/\-_]+)\.py$")
RE_PARSE_PATH = re.compile(RE_PARSE_PATH_PATTERN)
RE_PARSE_SCALA_FILE = re.compile("class\s+([a-zA-Z0-9]+Suite)\s+extends\s+")


_target_workflow_runs = [
    'Build and test'
]


_target_workflow_jobs = [
    'pyspark-sql, pyspark-mllib, pyspark-resource',
    'pyspark-core, pyspark-streaming, pyspark-ml',
    'pyspark-pandas',
    'pyspark-pandas-slow',
    'core, unsafe, kvstore, avro, network-common, network-shuffle, repl, launcher, examples, sketch, graphx',
    'catalyst, hive-thriftserver',
    'streaming, sql-kafka-0-10, streaming-kafka-0-10, mllib-local, mllib, yarn, mesos, kubernetes, '
    'hadoop-cloud, spark-ganglia-lgpl',
    'hive - slow tests',
    'hive - other tests',
    'sql - slow tests',
    'sql - other tests',
    'Run docker integration tests',
    'Run TPC-DS queries with SF=1'
]


_test_failure_patterns = [
    "error.+?(org\.apache\.spark\.[a-zA-Z0-9\.]+Suite)",
    "Had test failures in (pyspark\.[a-zA-Z0-9\._]+) with python"
]


_compilation_failure_patterns = [
    "error.+? Compilation failed",
    "Failing because of negative scalastyle result"
]


def create_workflow_handlers() -> Tuple[List[str], List[str], List[str], List[str]]:
    return _target_workflow_runs, _target_workflow_jobs, \
        _test_failure_patterns, _compilation_failure_patterns


def create_func_to_computer_path_difference() -> Any:
    excluded_paths = set(['src', 'main', 'scala', 'target', 'scala-2.12', 'test-classes', 'test'])
    path_excluded = lambda p: p.difference(excluded_paths)

    def _func(x: str, y: str) -> int:
        return len(path_excluded(set(x.split('/')[:-1])) ^ path_excluded(set(y.split('/')[:-1])))

    return _func


def create_func_to_transform_path_to_qualified_name() -> Any:
    parse_path = re.compile(f"[a-zA-Z0-9/\-]+/(org\/apache\/spark\/[a-zA-Z0-9/\-]+)\.scala")

    def _func(path: str) -> Optional[str]:
        result = parse_path.search(path)
        return result.group(1).replace('/', '.') if result else None

    return _func
