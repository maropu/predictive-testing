#!/usr/bin/env bash

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

#
# Predicts a preferred test set with a given predictive model

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ ! -z "$CONDA_ENABLED" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}"
fi

print_err_msg_and_exit() {
  echo "Required arguments not specified and usage: ${0} <root_paths> <#commits> <#selected_tests>" 1>&2
  exit 1
}

TARGET_PATH="$1"
if [ -z "${TARGET_PATH}" ]; then
  print_err_msg_and_exit
fi

NUM_COMMITS="$2"
if [ -z "${NUM_COMMITS}" ]; then
  print_err_msg_and_exit
fi

NUM_SELECTED_TESTS="$3"
if [ -z "${NUM_SELECTED_TESTS}" ]; then
  print_err_msg_and_exit
fi

PYTHONPATH="${FWDIR}/python" \
exec python3 -u ${FWDIR}/bin/ptesting-model.py \
  --target ${TARGET_PATH} \
  --num-commits ${NUM_COMMITS} \
  --num-selected-tests ${NUM_SELECTED_TESTS} \
  --model ${FWDIR}/models/spark/model.pkl \
  --test-files ${FWDIR}/models/spark/indexes/latest/test-files.json \
  --failed-tests ${FWDIR}/models/spark/failed-tests.json \
  --build-dep ${FWDIR}/models/spark/indexes/latest/dep-graph.json \
  --contributor-stats ${FWDIR}/models/spark/logs/contributor-stats.json
