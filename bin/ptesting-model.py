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

import json
import os
import pandas as pd  # type: ignore[import]
import pickle
from pyspark.sql import Column, DataFrame, SparkSession, functions as funcs
from typing import Any, Dict, List, Optional, Tuple

from ptesting import train


def _setup_logger() -> Any:
    import logging
    logfmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)
    return logging


_logger = _setup_logger()


def _create_func_for_path_diff() -> Any:
    # TODO: Removes package-depenent stuffs
    excluded_paths = set(['src', 'main', 'scala', 'target', 'scala-2.12', 'test-classes', 'test'])
    path_excluded = lambda p: p.difference(excluded_paths)

    def _func(x: str, y: str) -> int:
        return len(path_excluded(set(x.split('/')[:-1])) ^ path_excluded(set(y.split('/')[:-1])))

    return _func


def _create_func_to_transform_path_to_qualified_name() -> Any:
    import re
    # TODO: Removes package-depenent stuffs
    parse_path = re.compile(f"[a-zA-Z0-9/\-]+/(org\/apache\/spark\/[a-zA-Z0-9/\-]+)\.scala")

    def _func(path: str) -> Optional[Any]:
        return parse_path.search(path)

    return _func


def _create_func_to_enrich_authors(spark: SparkSession, contributor_stats: Optional[List[Tuple[str, str]]]) -> Any:
    if not contributor_stats:
        return lambda df: df.withColumn('num_commits', funcs.expr('0'))

    contributor_stat_df = spark.createDataFrame(contributor_stats, schema='author: string, num_commits: int')
    return lambda df: df.join(contributor_stat_df, 'author', 'LEFT_OUTER').na.fill({'num_commits': 0})


def _create_func_to_enumerate_related_tests(spark: SparkSession,
                                            dep_graph: Dict[str, List[str]],
                                            test_files: Dict[str, str]) -> Any:
    broadcasted_dep_graph = spark.sparkContext.broadcast(dep_graph)
    broadcasted_test_files = spark.sparkContext.broadcast(list(test_files.items()))

    def _func(df: DataFrame, output_col: str, depth: int, max_num_tests: Optional[int] = None) -> DataFrame:
        @funcs.pandas_udf("string")  # type: ignore
        def _enumerate_tests(file_paths: pd.Series) -> pd.Series:
            parse_path = _create_func_to_transform_path_to_qualified_name()
            path_diff = _create_func_for_path_diff()

            dep_graph = broadcasted_dep_graph.value
            test_files = broadcasted_test_files.value

            def _enumerate_tests_from_dep_graph(target):  # type: ignore
                if not target:
                    return []

                result = parse_path(target)
                if not result:
                    return []

                subgraph = {}
                visited_nodes = set()
                keys = list([result.group(1).replace('/', '.')])
                for i in range(0, depth):
                    if len(keys) == 0:
                        break

                    next_keys = set()
                    for key in keys:
                        if key in dep_graph and key not in visited_nodes:
                            nodes = dep_graph[key]
                            next_keys.update(nodes)
                    visited_nodes.update(keys)
                    keys = list(next_keys)
                if keys is not None:
                    visited_nodes.update(keys)
                tests = list(filter(lambda n: n.endswith('Suite'), visited_nodes))
                return tests

            ret = []
            for file_path in file_paths:
                related_tests = _enumerate_tests_from_dep_graph(file_path)
                if not related_tests and file_path:
                    # If no related test found, adds the tests whose paths are close to `file_path`
                    related_tests = [t for t, p in test_files if path_diff(file_path, p) <= 1]

                ret.append(json.dumps({'tests': related_tests}))

            return pd.Series(ret)

        related_test_df = df.selectExpr('sha', 'explode_outer(files.file.name) filename') \
            .withColumn('tests', _enumerate_tests(funcs.expr('filename'))) \
            .selectExpr('sha', 'from_json(tests, "tests ARRAY<STRING>").tests tests') \
            .selectExpr('sha', 'explode_outer(tests) test') \
            .groupBy('sha') \
            .agg(funcs.expr(f'collect_set(test) tests')) \
            .selectExpr('sha', 'size(tests) target_card', f'tests {output_col}') \
            .where(f'{output_col} IS NOT NULL')

        return df.join(related_test_df, 'sha', 'LEFT_OUTER')

    return _func


def _create_func_to_enumerate_all_tests(spark: SparkSession, test_files: Dict[str, str]) -> Any:
    all_test_df = spark.createDataFrame(list(test_files.items()), ['test', 'path'])

    def _func(df: DataFrame, output_col: str) -> DataFrame:
        return df.join(all_test_df.selectExpr(f'collect_set(test) {output_col}')) \
            .withColumn('target_card', funcs.expr(f'size({output_col})'))

    return _func


def _create_func_to_enrich_tests(failed_test_df: DataFrame) -> Tuple[Any, List[Dict[str, Any]]]:
    def _intvl(d: int) -> str:
        return f"current_timestamp() - interval {d} days"

    def _failed(d: int) -> str:
        return f'case when commit_date > intvl_{d}d then 1 else 0 end'

    failed_test_df = failed_test_df \
        .where('size(failed_tests) > 0') \
        .selectExpr(
            'to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss") commit_date',
            'explode_outer(failed_tests) failed_test',
            f'{_intvl(7)} intvl_7d',
            f'{_intvl(14)} intvl_14d',
            f'{_intvl(28)} intvl_28d') \
        .where('failed_test IS NOT NULL') \
        .selectExpr(
            'failed_test',
            f'{_failed(7)} failed_7d',
            f'{_failed(14)} failed_14d',
            f'{_failed(28)} failed_28d',
            '1 failed') \
        .groupBy('failed_test').agg(
            funcs.expr('sum(failed_7d) failed_num_7d'),
            funcs.expr('sum(failed_14d) failed_num_14d'),
            funcs.expr('sum(failed_28d) failed_num_28d'),
            funcs.expr('sum(failed) total_failed_num')) \
        .cache()

    def _func(df: DataFrame) -> DataFrame:
        return df.join(failed_test_df, df.test == failed_test_df.failed_test, 'LEFT_OUTER') \
            .na.fill({'failed_num_7d': 0, 'failed_num_14d': 0, 'failed_num_28d': 0, 'total_failed_num': 0}) \
            .drop('failed_test')

    failed_tests = list(map(lambda r: r.asDict(), failed_test_df.collect()))
    return _func, failed_tests


def _create_func_to_compute_distances(spark: SparkSession,
                                      dep_graph: Dict[str, List[str]],
                                      test_files: Dict[str, str]) -> Any:
    broadcasted_dep_graph = spark.sparkContext.broadcast(dep_graph)
    broadcasted_test_files = spark.sparkContext.broadcast(test_files)

    @funcs.pandas_udf("int")  # type: ignore
    def _path_diff(filenames: pd.Series, test: pd.Series) -> pd.Series:
        path_diff = _create_func_for_path_diff()

        test_files = broadcasted_test_files.value

        ret = []
        for names, t in zip(filenames, test):
            if t in test_files:
                distances = []
                for n in json.loads(names):
                    distances.append(path_diff(n, test_files[t]))
                ret.append(min(distances))
            else:
                ret.append(128)

        return pd.Series(ret)

    @funcs.pandas_udf("int")  # type: ignore
    def _distance(filenames: pd.Series, test: pd.Series) -> pd.Series:
        parse_path = _create_func_to_transform_path_to_qualified_name()

        dep_graph = broadcasted_dep_graph.value

        ret = []
        for names, t in zip(filenames, test):
            distances = [128]
            for n in json.loads(names):
                result = parse_path(n)
                if result:
                    ident = result.group(1).replace('/', '.')
                    if ident == t:
                        distances.append(0)
                        break

                    visited_nodes = set()
                    keys = list([ident])
                    for i in range(0, 16):
                        if len(keys) == 0:
                            break

                        next_keys = set()
                        for key in keys:
                            if key in dep_graph and key not in visited_nodes:
                                nodes = dep_graph[key]
                                next_keys.update(nodes)
                        if t in next_keys:
                            distances.append(i + 1)
                            break

                        visited_nodes.update(keys)
                        keys = list(next_keys)

            ret.append(min(distances))

        return pd.Series(ret)

    def _func(df: DataFrame, files: str, test: str) -> DataFrame:
        return df.withColumn('path_difference', _path_diff(funcs.expr(f'to_json({files})'), funcs.expr(test))) \
            .withColumn('distance', _distance(funcs.expr(f'to_json({files})'), funcs.expr(test)))

    return _func


def _expand_updated_files(df: DataFrame) -> DataFrame:
    def _sum(c: str) -> Column:
        return funcs.expr(f'aggregate({c}, 0, (x, y) -> int(x) + int(y))')

    def _expand_by_update_rate(df: DataFrame) -> DataFrame:
        return df \
            .withColumn('udpated_num_3d', _sum('files.updated[0]')) \
            .withColumn('udpated_num_14d', _sum('files.updated[1]')) \
            .withColumn('udpated_num_56d', _sum('files.updated[2]')) \
            .na.fill({'udpated_num_3d': 0, 'udpated_num_14d': 0, 'udpated_num_56d': 0})

    df = _expand_by_update_rate(df) \
        .withColumn('num_adds', _sum('files.file.additions')) \
        .withColumn('num_dels', _sum("files.file.deletions")) \
        .withColumn('num_chgs', _sum("files.file.changes"))

    return df


# This method extracts features from a dataset of historical test outcomes.
# The current features used in our model are as follows:
#  - Change history for files: the count of commits made to modified files in the last 3, 14, and 56 days
#    (`updated_num_3d`, `updated_num_14d`, and `updated_num_15d`, respectively).
#  - File update statistics: the total number of additions, deletions, and changes made to modified files
#    (`num_adds`, `num_dels`, and `num_chgs`, respectively).
#  - File cardinality: the number of files touched in a test run (`file_card`).
#  - Target cardinality: the number of tests invoked in a test run (`target_card`).
#  - Historical failure rates: the count of target test failures occurred in the last 7, 14, and 28 days
#    (`failed_num_7d`, `failed_num_14d`, and `failed_num_28d`, respectively) and the total count
#    of target test failures in historical test outcomes (`total_failed_num`).
#  - Minimal distance between one of modified files and a prediction target: the number of different directories
#    between file paths (`path_difference`). Let's say that we have two files: their file paths are
#    'xxx/yyy/zzz/file' and 'xxx/aaa/zzz/test_file'. In the example, a minimal distance is 1.
#
# NOTE: The Facebook paper [1] reports that the best performance predictive model uses a change history,
# failure rates, target cardinality, and minimal distances (For more details, see
# "Section 6.B. Feature Selection" in the paper [1]). Even in our model, change history for files
# and historical failure rates tend to be more important than the other features.
#
# TODO: Needs to improve predictive model performance by checking the other feature candidates
# that can be found in the Facebook paper (See "Section 4.A. Feature Engineering") [1]
# and the Google paper (See "Section 4. Hypotheses, Models and Results") [2].
def _create_train_feature_from(df: DataFrame,
                               enrich_authors: Any,
                               enumerate_tests: Any,
                               enrich_tests: Any,
                               compute_minimal_distances: Any) -> DataFrame:
    # TODO: Revisit this feature engineering
    df = df.selectExpr('sha', 'author', 'commit_date', 'files', 'failed_tests')
    df = enrich_authors(df).drop('author')
    df = _expand_updated_files(df)

    related_test_df = enumerate_tests(df, output_col='related_tests', depth=2, max_num_tests=16)

    passed_test_df = related_test_df \
        .selectExpr('*', 'array_except(related_tests, failed_tests) tests', '0 failed') \
        .where('size(tests) > 0')

    failed_test_df = related_test_df \
        .where('size(failed_tests) > 0') \
        .selectExpr('*', 'failed_tests tests', '1 failed')

    df = passed_test_df.union(failed_test_df) \
        .selectExpr('*', 'explode(tests) test') \
        .drop('sha', 'related_tests', 'tests')

    df = enrich_tests(df)
    df = compute_minimal_distances(df, 'files.file.name', 'test')
    df = df.withColumn('file_card', funcs.expr('size(files)'))
    df = df.drop('commit_date', 'files', 'failed_tests', 'test')
    return df


def _create_test_feature_from(df: DataFrame,
                              enrich_authors: Any,
                              enumerate_tests: Any,
                              enrich_tests: Any,
                              compute_minimal_distances: Any) -> DataFrame:
    df = df.selectExpr('sha', 'author', 'commit_date', 'files')
    df = enrich_authors(df).drop('author')
    df = _expand_updated_files(df)
    df = enumerate_tests(df, output_col='related_tests') \
        .selectExpr('*', 'explode_outer(related_tests) test') \
        .drop('related_tests')
    df = enrich_tests(df)
    df = compute_minimal_distances(df, 'files.file.name', 'test')
    df = df.withColumn('file_card', funcs.expr('size(files)'))
    df = df.drop('commit_date', 'files')
    return df


# Our predictive model uses LightGBM, an implementation of gradient-boosted decision trees.
# This is because the algorithm has desirable properties for this use-case
# (the reason is the same with the Facebook one):
#  - normalizing feature values are less required
#  - fast model training on commodity hardware
#  - robustness in imbalanced datasets
def _build_predictive_model(df: DataFrame, to_features: Any) -> Any:
    pdf = to_features(df).drop('target_card').toPandas()
    X = pdf[pdf.columns[pdf.columns != 'failed']]  # type: ignore
    y = pdf['failed']
    X, y = train.rebalance_training_data(X, y)
    # clf, score = train.build_model(X, y, opts={'hp.timeout': '3600', 'hp.no_progress_loss': '1000'})
    clf, score = train.build_model(X, y, opts={'hp.timeout': '3600', 'hp.no_progress_loss': '1'})
    _logger.info(f"model score: {score}")
    return clf


def _train_test_split(df: DataFrame, test_ratio: float) -> Tuple[DataFrame, DataFrame]:
    test_nrows = int(df.count() * test_ratio)
    test_df = df.orderBy(funcs.expr('to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss")').desc()).limit(test_nrows)
    train_df = df.subtract(test_df)
    return train_df, test_df


def _predict_failed_probs_for_tests(test_df: DataFrame, clf: Any, to_features: Any) -> DataFrame:
    test_feature_df = to_features(test_df)
    pdf = test_feature_df.selectExpr('sha', 'target_card', 'test').toPandas()
    X = test_feature_df.drop('sha', 'target_card', 'failed', 'test').toPandas()
    predicted = clf.predict_proba(X)
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = test_df.sql_ctx.sparkSession.createDataFrame(pdf) \
        .withColumn('pmf', to_map_expr) \
        .selectExpr('sha', 'target_card', 'test', to_failed_prob)

    compare = lambda x, y: \
        f"case when {x}.failed_prob < {y}.failed_prob then 1 " \
        f"when {x}.failed_prob > {y}.failed_prob then -1 " \
        "else 0 end"
    df_with_failed_probs = df_with_failed_probs.groupBy('sha') \
        .agg(funcs.expr('first(target_card)').alias('target_card'),
             funcs.expr('collect_set(named_struct("test", test, "failed_prob", failed_prob))').alias('tests')) \
        .selectExpr('sha', 'target_card', f'array_sort(tests, (l, r) -> {compare("l", "r")}) tests')

    return df_with_failed_probs


def _predict_failed_probs(df: DataFrame, clf: Any) -> DataFrame:
    pdf = df.toPandas()
    predicted = clf.predict_proba(pdf.drop('test', axis=1))
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = df.sql_ctx.sparkSession.createDataFrame(pdf[['test', 'predicted']]) \
        .withColumn('pmf', to_map_expr) \
        .selectExpr('test', to_failed_prob)

    compare = lambda x, y: \
        f"case when {x}.failed_prob < {y}.failed_prob then 1 " \
        f"when {x}.failed_prob > {y}.failed_prob then -1 " \
        "else 0 end"
    df_with_failed_probs = df_with_failed_probs \
        .selectExpr('collect_set(named_struct("test", test, "failed_prob", failed_prob)) tests') \
        .selectExpr(f'array_sort(tests, (l, r) -> {compare("l", "r")}) tests')

    return df_with_failed_probs


def _compute_eval_stats(df: DataFrame) -> DataFrame:
    rank = 'array_position(transform(tests, x -> x.test), failed_test) rank'
    to_test_map = 'map_from_arrays(transform(tests, x -> x.test), transform(tests, x -> x.failed_prob)) tests'
    return df.selectExpr('sha', 'explode(failed_tests) failed_test', 'tests') \
        .selectExpr('sha', 'failed_test', rank, 'tests[0].failed_prob max_score', to_test_map) \
        .selectExpr('sha', 'failed_test', 'rank', 'tests[failed_test] score', 'max_score')


def _compute_eval_metrics(predicted: DataFrame, eval_num_tests: List[int]) -> Any:
    # This method computes metrics to measure the quality of a selected test set; "test recall", which is computed
    # in the method, represents the emprical probability of a particular test selection strategy catching
    # an individual failure (See "Section 3.B. Measuring Quality of Test Selection" in the Facebook paper [1]).
    # The equation to compute the metric is defined as follows:
    #  - TestRecall(D) = \frac{\sum_{d \in D} |SelectedTests(d) \bigcamp F_{d}|}{\sum_{d \in D} |F_{d}|}
    # , where D is a set of code changes and F_{d} is a set of failed tests.
    #
    # TODO: Computes a "chnage recall" metric.
    def _metric(num_tests: int, score_thres: float = 0.0) -> Tuple[float, float]:
        # TODO: Needs to make 'num_dependent_tests' more precise
        filtered_test_expr = funcs.expr(f'filter(tests, x -> x.failed_prob >= {score_thres})')
        eval_df = predicted.withColumn('filtered_tests', filtered_test_expr) \
            .withColumn('tests_', funcs.expr(f'case when size(filtered_tests) <= {num_tests} then filtered_tests '
                                             f'else slice(tests, 1, {num_tests}) end')) \
            .selectExpr('sha', 'target_card', 'failed_tests', 'transform(tests_, x -> x.test) tests') \
            .selectExpr(
                'sha',
                'target_card num_dependent_tests',
                'size(failed_tests) num_failed_tests',
                'size(tests) num_tests',
                'size(array_intersect(failed_tests, tests)) covered') \
            .selectExpr(
                'SUM(covered) / SUM(num_failed_tests) recall',
                'SUM(num_tests) / SUM(num_dependent_tests) ratio')
        row = eval_df.collect()[0]
        return row.recall, row.ratio

    to_metric_values = lambda m: {'num_tests': m[0], 'test_recall': m[1], 'selection_ratio': m[2]}
    metrics = [(num_tests, *_metric(num_tests)) for num_tests in eval_num_tests]
    metrics = list(map(to_metric_values, metrics))  # type: ignore
    return metrics


def _format_eval_metrics(metrics: List[Dict[str, Any]]) -> str:
    strbuf: List[str] = []
    strbuf.append('|  #tests  |  test recall  |  selection ratio  |')
    strbuf.append('| ---- | ---- | ---- |')
    for m in metrics:  # type: ignore
        strbuf.append(f'|  {m["num_tests"]}  |  {m["test_recall"]}  |  {m["selection_ratio"]}  |')  # type: ignore

    return '\n'.join(strbuf)


def _save_metrics_as_chart(output_path: str, metrics: List[Dict[str, Any]], max_test_num: int) -> None:
    import altair as alt
    x_opts = {
        'scale': alt.Scale(domain=[0, max_test_num]),
        'axis': alt.Axis(title='#tests', titleFontSize=16, labelFontSize=14)
    }
    y_opts = {
        'scale': alt.Scale(domain=[0.0, 1.0]),
        'axis': alt.Axis(title='test recall', titleFontSize=16, labelFontSize=14)
    }
    df = pd.DataFrame(metrics)
    plot = alt.Chart(df).mark_point().encode(x=alt.X("num_tests", **x_opts), y=alt.Y("test_recall", **y_opts)) \
        .properties(width=600, height=400) \
        .interactive()

    plot.save(output_path)


def _train_and_eval_ptest_model(output_path: str, spark: SparkSession, df: DataFrame,
                                test_files: Dict[str, str],
                                dep_graph: Dict[str, List[str]],
                                contributor_stats: Optional[List[Tuple[str, str]]],
                                test_ratio: float = 0.20) -> None:
    train_df, test_df = _train_test_split(df, test_ratio=test_ratio)
    _logger.info(f"Split data: #total={df.count()}, #train={train_df.count()}, #test={test_df.count()}")

    enrich_authors = _create_func_to_enrich_authors(spark, contributor_stats)
    enumerate_related_tests = _create_func_to_enumerate_related_tests(spark, dep_graph, test_files)
    enumerate_all_tests = _create_func_to_enumerate_all_tests(spark, test_files)
    enrich_tests, failed_tests = _create_func_to_enrich_tests(train_df)
    compute_distances = _create_func_to_compute_distances(spark, dep_graph, test_files)

    def _to_features(df: DataFrame, f: Any, enumerate_tests: Any) -> Any:
        return f(df, enrich_authors, enumerate_tests, enrich_tests, compute_distances)

    to_features = lambda df: _to_features(df, _create_train_feature_from, enumerate_related_tests)
    clf = _build_predictive_model(train_df, to_features)

    with open(f"{output_path}/failed-tests.json", 'w') as f:
        f.write(json.dumps(failed_tests, indent=2))
    with open(f"{output_path}/model.pkl", 'wb') as f:  # type: ignore
        pickle.dump(clf, f)  # type: ignore

    to_features = lambda df: _to_features(df, _create_test_feature_from, enumerate_all_tests)
    predicted = _predict_failed_probs_for_tests(test_df, clf, to_features)
    predicted = test_df.selectExpr('sha', 'failed_tests').join(predicted, 'sha', 'LEFT_OUTER') \
        .selectExpr('sha', 'target_card', 'failed_tests', 'coalesce(tests, array()) tests')

    metrics = _compute_eval_metrics(predicted, eval_num_tests=list(range(4, len(test_files) + 1, 4)))
    stats = _compute_eval_stats(predicted)

    with open(f"{output_path}/model-eval-stats.json", 'w') as f:  # type: ignore
        f.write(json.dumps(stats.toPandas().to_dict(orient='records'), indent=2))  # type: ignore
    with open(f"{output_path}/model-eval-metric-summary.md", 'w') as f:  # type: ignore
        f.write(_format_eval_metrics([metrics[i] for i in [0, 14, 29, 44, 59]]))  # type: ignore
    with open(f"{output_path}/model-eval-metrics.json", 'w') as f:  # type: ignore
        f.write(json.dumps(metrics, indent=2))  # type: ignore

    _save_metrics_as_chart(f"{output_path}/model-eval-metrics.svg", metrics, len(test_files))


def train_main(argv: Any) -> None:
    # Parses command-line arguments for a training mode
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train-log-data', type=str, required=True)
    parser.add_argument('--test-files', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    # TODO: Makes `--build-dep` optional
    parser.add_argument('--build-dep', type=str, required=True)
    parser.add_argument('--excluded-tests', type=str, required=False)
    args = parser.parse_args(argv)

    if not os.path.exists(args.output) or not os.path.isdir(args.output):
        raise ValueError(f"Output directory not found in {os.path.abspath(args.output)}")
    if not os.path.exists(args.train_log_data):
        raise ValueError(f"Training data not found in {os.path.abspath(args.train_log_data)}")
    if not os.path.exists(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if args.contributor_stats and not os.path.exists(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.exists(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")
    if args.excluded_tests and not os.path.exists(args.excluded_tests):
        raise ValueError(f"Excluded test list file not found in {os.path.abspath(args.excluded_tests)}")

    from pathlib import Path
    test_files = json.loads(Path(args.test_files).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None
    excluded_tests = json.loads(Path(args.excluded_tests).read_text()) \
        if args.excluded_tests else None

    # Initializes a Spark session
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppresses user warinig messages in Python
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Suppresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Assigns a random string if 'sha' is an empty string
        # TODO: Needs to validate input log data
        log_data_df = spark.read.format('json').load(args.train_log_data) \
            .withColumn('sha_', funcs.expr('case when length(sha) > 0 then sha else sha(string(random())) end')) \
            .drop('sha') \
            .withColumnRenamed('sha_', 'sha')

        # Removes reduandant entries in `failed_tests`
        log_data_df = log_data_df.withColumn('failed_tests_', funcs.expr('array_distinct(failed_tests)')) \
            .drop('failed_tests') \
            .withColumnRenamed('failed_tests_', 'failed_tests') \
            .distinct()

        # Excludes failed tests (e.g., flaky ones) if necessary
        if excluded_tests:
            excluded_test_df = spark.createDataFrame(pd.DataFrame(excluded_tests, columns=['excluded_test'])) \
                .selectExpr('collect_set(excluded_test) excluded_tests')
            log_data_df = log_data_df.join(excluded_test_df) \
                .withColumn('failed_tests_', funcs.expr('array_except(failed_tests, excluded_tests)')) \
                .drop('failed_tests', 'excluded_tests') \
                .withColumnRenamed('failed_tests_', 'failed_tests')

        # Checks if all the failed tests exist in `test_files`
        failed_tests = log_data_df \
            .selectExpr('explode(failed_tests) failed_test') \
            .selectExpr('collect_set(failed_test) failed_tests') \
            .collect()[0].failed_tests

        unknown_failed_tests = set(failed_tests).difference(set(test_files.keys()))
        if len(unknown_failed_tests) > 0:
            _logger.warning(f'Unknown failed tests found: {",".join(unknown_failed_tests)}')

        _train_and_eval_ptest_model(args.output, spark, log_data_df, test_files, dep_graph,
                                    contributor_stats, test_ratio=0.10)
    finally:
        spark.stop()


def predict_main(argv: Any) -> None:
    # Parses command-line arguments for a prediction mode
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--num-commits', type=int, required=True)
    parser.add_argument('--num-selected-tests', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-files', type=str, required=True)
    parser.add_argument('--failed-tests', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    # TODO: Makes `--build-dep` optional
    parser.add_argument('--build-dep', type=str, required=True)
    args = parser.parse_args(argv)

    if not os.path.isdir(f'{args.target}/.git'):
        raise ValueError(f"Git-managed directory not found in {os.path.abspath(args.target)}")
    if args.num_commits <= 0:
        raise ValueError(f"Target #commits must be positive, but {args.num_commits}")
    if args.num_selected_tests <= 0:
        raise ValueError(f"Predicted #tests must be positive, but {args.num_selected_tests}")
    if not os.path.exists(args.model):
        raise ValueError(f"Predictive model not found in {os.path.abspath(args.model)}")
    if not os.path.exists(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if not os.path.exists(args.failed_tests):
        raise ValueError(f"Failed test list file not found in {os.path.abspath(args.failed_tests)}")
    if args.contributor_stats and not os.path.exists(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.exists(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")

    from pathlib import Path
    clf = pickle.loads(Path(args.model).read_bytes())
    test_files = json.loads(Path(args.test_files).read_text())
    failed_tests = json.loads(Path(args.failed_tests).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None

    # Initializes a Spark session
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppresses user warinig messages in Python
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Suppresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    try:
        commit_info = [{
            'num_commits': 0,
            'udpated_num_3d': 0,
            'udpated_num_14d': 0,
            'udpated_num_56d': 0,
            'num_adds': 0,
            'num_dels': 0,
            'num_chgs': 0
        }]

        compute_distances = _create_func_to_compute_distances(spark, dep_graph, test_files)
        commit_df = spark.createDataFrame(pd.DataFrame(commit_info)) \
            .withColumn('filenames', funcs.expr('array("README.md")')) \
            .withColumn('file_card', funcs.expr('size(filenames)'))

        all_test_df = spark.createDataFrame(list(test_files.items()), ['test', 'path']) \
            .selectExpr('test', '0 failed_num_7d', '0 failed_num_14d', '0 failed_num_28d', '0 total_failed_num')
        all_test_df = compute_distances(all_test_df.join(commit_df), 'filenames', 'test') \
            .drop('filenames')

        predicted = _predict_failed_probs(all_test_df, clf)
        selected_test_df = predicted \
            .selectExpr(f'slice(tests, 1, {args.num_selected_tests}) selected_tests') \
            .selectExpr('selected_tests.test selected_tests')

        selected_tests = selected_test_df.collect()[0].selected_tests
        print(json.dumps(selected_tests, indent=2))
    finally:
        spark.stop()


def main() -> None:
    # Checks if a training mode enabled or not first
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', dest='train_mode_enabled', action='store_true')
    args, rest_argv = parser.parse_known_args()

    if args.train_mode_enabled:
        train_main(rest_argv)
    else:
        predict_main(rest_argv)


if __name__ == '__main__':
    main()
