"""Backend unit tests for the classical-ML pipeline (no server, no network).

Covers: dataset parsing, schema detection, target validation, preprocessing
(fit-on-train-only), split correctness, stratification, seed reproducibility,
classification + regression training, metric validity, prediction, artifact
creation, experiment serialization/reload, CV, and search limits.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import dataset_manager as dm
from ml import evaluation as ev
from ml import models as mm
from ml import preprocessing as pp
from ml import serialization as se
from ml import split as sp
from ml.experiment_runner import (
    _evaluate_split,
    _expand_search_space,
    _fit_pipeline,
)


def make_frame(n=200, seed=0, missing=True):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'age': rng.randint(18, 70, n).astype(float),
        'income': rng.normal(50000, 15000, n),
        'city': rng.choice(['a', 'b', 'c'], n),
        'member': rng.choice(['yes', 'no'], n),
    })
    df['bought'] = ((df['income'] > 52000) ^ (df['city'] == 'a')).astype(int).astype(str)
    df['spend'] = df['income'] * 0.1 + rng.normal(0, 500, n)
    if missing:
        df.loc[::17, 'age'] = np.nan
        df.loc[::23, 'city'] = None
    return df


class DatasetTest(unittest.TestCase):
    def test_csv_tsv_json_parse(self):
        df, _ = dm.load_frame_from_text('a.csv', 'x,y\n1,2\n3,4\n', 10**6, 10**5)
        self.assertEqual(list(df.columns), ['x', 'y'])
        df, _ = dm.load_frame_from_text('a.tsv', 'x\ty\n1\t2\n', 10**6, 10**5)
        self.assertEqual(df.iloc[0]['y'], 2)
        df, _ = dm.load_frame_from_text('a.json', '[{"x": 1}, {"x": 2}]', 10**6, 10**5)
        self.assertEqual(len(df), 2)

    def test_parse_errors(self):
        with self.assertRaises(dm.DatasetError):
            dm.load_frame_from_text('a.csv', '', 10**6, 10**5)
        with self.assertRaises(dm.DatasetError):
            dm.load_frame_from_text('a.csv', 'x\n', 10**6, 10**5)  # header only, no rows
        with self.assertRaises(dm.DatasetError):
            dm.load_frame_from_text('a.json', 'not json{{{', 10**6, 10**5)
        big = 'x\n' + '1\n' * 10
        with self.assertRaises(dm.DatasetError):
            dm.load_frame_from_text('a.csv', big, 5, 10**5)

    def test_row_cap_and_truncation_flag(self):
        df, info = dm.load_frame_from_text('a.csv', 'x\n' + '1\n' * 10, 10**6, 4)
        self.assertEqual(len(df), 4)
        self.assertTrue(info['truncated'])

    def test_schema_detection(self):
        df = make_frame()
        kinds = {m['name']: m['kind'] for m in dm.column_metadata(df)}
        self.assertEqual(kinds['age'], 'numeric')
        self.assertEqual(kinds['city'], 'categorical')
        self.assertEqual(kinds['spend'], 'numeric')
        meta = {m['name']: m for m in dm.column_metadata(df)}
        self.assertGreater(meta['age']['missing'], 0)
        self.assertEqual(meta['city']['unique'], 3)

    def test_profile_shape(self):
        df = make_frame()
        p = dm.profile_dataframe(df, target='bought')
        self.assertEqual(p['row_count'], 200)
        self.assertEqual(p['duplicate_rows'], df.duplicated().sum())
        self.assertIn('age', p['numeric_summary'])
        self.assertIn('histogram', p['numeric_summary']['age'])
        self.assertIsNotNone(p['correlation'])
        self.assertIsNotNone(p['scatter'])
        self.assertIsNotNone(p['target_distribution'])
        self.assertTrue(len(p['preview']) <= 100)

    def test_view_drop_filter_sample(self):
        df = make_frame()
        out, applied = dm.apply_view(df, {'drop_columns': ['member'],
                                          'filters': [{'column': 'city', 'op': '==', 'value': 'a'}],
                                          'sample': {'n': 10, 'seed': 1}})
        self.assertNotIn('member', out.columns)
        self.assertTrue((out['city'] == 'a').all())
        self.assertEqual(len(out), 10)
        self.assertEqual(applied['dropped'], ['member'])


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.df = make_frame().reset_index(drop=True)

    def test_counts_and_coverage(self):
        parts, _ = sp.make_splits(self.df, {'strategy': 'train_val_test', 'train_frac': 0.7,
                                            'val_frac': 0.15, 'test_frac': 0.15, 'seed': 42,
                                            'shuffle': True, 'stratify': False}, target='bought')
        total = sum(len(v) for v in parts.values())
        self.assertEqual(total, len(self.df))
        all_idx = np.concatenate([np.asarray(parts[k]) for k in parts])
        self.assertEqual(len(set(all_idx.tolist())), len(self.df))

    def test_stratification_preserves_ratio(self):
        parts, notes = sp.make_splits(self.df, {'strategy': 'train_val_test', 'train_frac': 0.7,
                                                'val_frac': 0.15, 'test_frac': 0.15, 'seed': 7,
                                                'shuffle': True, 'stratify': True}, target='bought')
        self.assertIn('stratified', notes)
        overall = (self.df['bought'] == 'True').mean()
        for k in ('train', 'validation', 'test'):
            r = (self.df.iloc[np.asarray(parts[k])]['bought'] == 'True').mean()
            self.assertAlmostEqual(r, overall, delta=0.15)

    def test_seed_reproducibility(self):
        cfg = {'strategy': 'train_test', 'train_frac': 0.8, 'test_frac': 0.2,
               'seed': 123, 'shuffle': True, 'stratify': True}
        a, _ = sp.make_splits(self.df, cfg, target='bought')
        b, _ = sp.make_splits(self.df, cfg, target='bought')
        for k in a:
            self.assertTrue(np.array_equal(np.asarray(a[k]), np.asarray(b[k])))
        c, _ = sp.make_splits(self.df, dict(cfg, seed=999), target='bought')
        self.assertFalse(np.array_equal(np.asarray(a['test']), np.asarray(c['test'])))

    def test_group_no_leak(self):
        parts, _ = sp.make_splits(self.df, {'strategy': 'train_test', 'train_frac': 0.8,
                                            'test_frac': 0.2, 'seed': 1, 'shuffle': True,
                                            'stratify': False, 'group_column': 'city'}, target='bought')
        tr_cities = set(self.df.iloc[np.asarray(parts['train'])]['city'])
        te_cities = set(self.df.iloc[np.asarray(parts['test'])]['city'])
        self.assertTrue(tr_cities.isdisjoint(te_cities))

    def test_time_order(self):
        df = make_frame().reset_index(drop=True)
        df['day'] = pd.date_range('2020-01-01', periods=len(df))
        parts, notes = sp.make_splits(df, {'strategy': 'time', 'time_column': 'day',
                                           'train_frac': 0.7, 'val_frac': 0.15, 'test_frac': 0.15})
        self.assertTrue('time-ordered' in notes[0])
        self.assertLess(df.iloc[np.asarray(parts['train'])]['day'].max(),
                        df.iloc[np.asarray(parts['test'])]['day'].min())

    def test_official_passthrough(self):
        df = make_frame().reset_index(drop=True)
        df['split'] = np.where(np.arange(len(df)) % 10 < 7, 'train',
                               np.where(np.arange(len(df)) % 10 < 8, 'validation', 'test'))
        parts, notes = sp.make_splits(df, {'strategy': 'official', 'split_column': 'split'})
        self.assertEqual(len(parts['train']) + len(parts['validation']) + len(parts['test']), len(df))
        self.assertIn('official', notes[0])

    def test_bad_fractions_rejected(self):
        with self.assertRaises(sp.SplitError):
            sp.validate_split_config({'strategy': 'train_val_test', 'train_frac': 0.5,
                                      'val_frac': 0.5, 'test_frac': 0.5})


class PreprocessingTest(unittest.TestCase):
    def test_fit_only_on_train(self):
        # Shifted test distribution must not leak into imputer statistics.
        train = pd.DataFrame({'age': [20.0, 30.0, 40.0], 'city': ['a', 'b', 'a']})
        pre, _ = pp.build_preprocessor({}, ['age'], ['city'])
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('preprocess', pre),
                         ('estimator', mm.build_estimator('LogisticRegression', {}, 0)[0])])
        pipe.fit(train, pd.Series(['yes', 'no', 'yes']))
        med = pipe.named_steps['preprocess'].named_transformers_['num'].named_steps['impute'].statistics_
        self.assertAlmostEqual(float(med[0]), 30.0)  # train median, unaffected by test
        test = pd.DataFrame({'age': [1000.0], 'city': ['zzz-unknown']})
        out = pipe.named_steps['preprocess'].transform(test)
        self.assertFalse(np.isnan(np.asarray(out, dtype=float)).any())
        self.assertEqual(out.shape[1], 1 + 2)  # 1 numeric + 2 one-hot (a,b; unknown ignored)

    def test_unknown_categories_ignored(self):
        train = pd.DataFrame({'city': ['a', 'b']})
        pre, _ = pp.build_preprocessor({'categorical_columns': ['city']}, [], ['city'])
        pre.fit(train)
        out = pre.transform(pd.DataFrame({'city': ['never-seen']}).astype(object))
        self.assertEqual(np.asarray(out, dtype=float).shape, (1, 2))

    def test_summary_and_validation(self):
        _, spec = pp.build_preprocessor({}, ['age'], ['city'])
        s = pp.summarize_preprocessor(spec)
        self.assertIn('numeric', s)
        self.assertIn('categorical', s)
        with self.assertRaises(pp.PreprocessingError):
            pp.build_preprocessor({'numeric_columns': ['city']}, ['age'], ['city'])
        with self.assertRaises(pp.PreprocessingError):
            pp.build_preprocessor({}, ['age'], ['city'],
                                  ) if False else pp.validate_preprocessing_config(
                {'numeric': {'scaler': 'bogus'}}, ['age'], ['city'])


class ModelsTest(unittest.TestCase):
    def test_catalog_coverage(self):
        from ml.models import CATALOG
        for fam in ('LogisticRegression', 'RandomForestClassifier', 'SVC',
                    'LinearRegression', 'RandomForestRegressor', 'SVR'):
            self.assertIn(fam, CATALOG)

    def test_unknown_family_and_param(self):
        with self.assertRaises(mm.ModelError):
            mm.build_estimator('Nope', {}, 0)
        with self.assertRaises(mm.ModelError):
            mm.build_estimator('Ridge', {'bogus': 1}, 0)

    def test_max_depth_sentinel(self):
        est, merged = mm.build_estimator('RandomForestClassifier', {'max_depth': 0}, 1)
        self.assertIsNone(est.max_depth)
        self.assertIsNone(merged['max_depth'])


class MetricsTest(unittest.TestCase):
    def test_classification_full(self):
        y = np.array(['a', 'a', 'b', 'b'])
        p = np.array(['a', 'b', 'b', 'b'])
        proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
        m = ev.classification_metrics(y, p, proba, labels=['a', 'b'])
        self.assertAlmostEqual(m['accuracy'], 0.75)
        self.assertIsNotNone(m['roc_auc'])
        self.assertIsNotNone(m['pr_auc'])
        self.assertIsNotNone(m['log_loss'])
        self.assertEqual(len(m['per_class']), 2)
        self.assertEqual(len(m['confusion_matrix']), 2)

    def test_no_proba_means_no_probability_metrics(self):
        y = np.array(['a', 'b', 'b'])
        m = ev.classification_metrics(y, y, None, labels=['a', 'b'])
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['pr_auc'])
        self.assertIsNone(m['log_loss'])
        self.assertIsNone(m['roc_curve'])
        self.assertAlmostEqual(m['accuracy'], 1.0)

    def test_regression(self):
        m = ev.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(m['mae'], 0.0)
        self.assertAlmostEqual(m['rmse'], 0.0)
        self.assertAlmostEqual(m['r2'], 1.0)


class EndToEndTest(unittest.TestCase):
    def test_full_run_and_artifacts(self):
        import subprocess
        tmp = tempfile.mkdtemp(prefix='mltest_')
        try:
            df = make_frame(n=120, seed=3)
            df.to_csv(os.path.join(tmp, 'data.csv'), index=False)
            ws = 'mlw_aaaaaaaaaaaa'
            os.makedirs(os.path.join(tmp, 'ws', ws))
            import shutil as _sh
            _sh.copy(os.path.join(tmp, 'data.csv'), os.path.join(tmp, 'ws', ws, 'data.csv'))
            cfg = {
                'experiment_id': 'exp_bbbbbbbbbbbb',
                'workspace_id': ws,
                'workspace_base': os.path.join(tmp, 'wsbase'),
                'task': 'classification',
                'target_column': 'bought',
                'feature_columns': None,
                'dataset_view': {},
                'split': {'strategy': 'train_val_test', 'train_frac': 0.7, 'val_frac': 0.15,
                          'test_frac': 0.15, 'seed': 42, 'shuffle': True, 'stratify': True},
                'preprocessing': {},
                'model': {'family': 'LogisticRegression', 'params': {}},
                'seed': 42,
                'cv': {'method': 'stratified', 'folds': 3, 'scoring': 'f1_macro', 'seed': 42},
                'search': None,
            }
            # point the runner at our scratch workspace layout
            os.makedirs(os.path.join(tmp, 'wsbase', 'ml_workspaces', ws))
            _sh.copy(os.path.join(tmp, 'data.csv'),
                     os.path.join(tmp, 'wsbase', 'ml_workspaces', ws, 'data.csv'))
            with open(os.path.join(tmp, 'wsbase', 'ml_workspaces', ws, 'manifest.json'), 'w') as f:
                json.dump({'workspace_id': ws, 'source_type': 'upload',
                           'source_name': 'test.csv', 'sha256': 'test'}, f)
            workdir = os.path.join(tmp, 'exp')
            os.makedirs(workdir)
            cfg_path = os.path.join(workdir, 'config.json')
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f)
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            r = subprocess.run([sys.executable, os.path.join(repo, 'ml', 'experiment_runner.py'),
                                'run', '--workdir', workdir, '--config', cfg_path],
                               capture_output=True, text=True, timeout=300, cwd=repo)
            last = None
            for line in r.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        last = json.loads(line)
                    except Exception:
                        pass
            self.assertIsNotNone(last, 'runner produced no result line; stderr: %s' % r.stderr[-2000:])
            self.assertEqual(last.get('status'), 'finished', 'stderr: %s' % r.stderr[-2000:])
            for fn in ('config.json', 'metrics.json', 'model.joblib', 'split.joblib', 'experiment.json'):
                self.assertTrue(os.path.exists(os.path.join(workdir, fn)), fn)
            rec = json.load(open(os.path.join(workdir, 'experiment.json')))
            self.assertEqual(rec['status'], 'finished')
            self.assertIn('test', rec['metrics'])
            self.assertIn('cv', rec['metrics'])
            # reload + predict from the artifact
            bundle = se.load_joblib(os.path.join(workdir, 'model.joblib'))
            self.assertIn('pipeline', bundle)
            X = df[bundle['feature_columns']].head(5)
            pred = bundle['pipeline'].predict(X)
            self.assertEqual(len(pred), 5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_search_limits(self):
        from ml.experiment_runner import _expand_search_space
        self.assertEqual(_expand_search_space({}), {})
        space = _expand_search_space({'a': [1, 2], 'b': {'low': 0, 'high': 1, 'steps': 3}})
        self.assertEqual(space['a'], [1, 2])
        self.assertEqual(len(space['b']), 3)


class HfConfigTest(unittest.TestCase):
    """Config/subset passthrough for HF ingestion (no network: the datasets
    library is stubbed; only argument routing is exercised)."""

    def _fake_datasets(self, record):
        import types
        fake = types.ModuleType('datasets')

        class FakeDS:
            def to_pandas(self):
                import pandas as pd
                return pd.DataFrame({'x': [1, 2], 'y': ['a', 'b']})

        def load_dataset(dataset_id, name=None, split=None, streaming=False):
            record['dataset_id'] = dataset_id
            record['name'] = name
            record['split'] = split
            return FakeDS()

        fake.load_dataset = load_dataset
        return fake

    def test_config_reaches_loader(self):
        import sys
        record = {}
        sys.modules['datasets'] = self._fake_datasets(record)
        try:
            df, _ = dm.load_frame_from_hf('owner/ds', split='train', max_rows=10, config='plain_text')
            self.assertEqual(len(df), 2)
            self.assertEqual(record['name'], 'plain_text')
            self.assertEqual(record['split'], 'train')
        finally:
            sys.modules.pop('datasets', None)

    def test_no_config_passes_none(self):
        import sys
        record = {}
        sys.modules['datasets'] = self._fake_datasets(record)
        try:
            dm.load_frame_from_hf('owner/ds', split=None, max_rows=10)
            self.assertIsNone(record['name'])
        finally:
            sys.modules.pop('datasets', None)

    def test_invalid_config_rejected_without_network(self):
        import sys
        record = {}
        sys.modules['datasets'] = self._fake_datasets(record)
        try:
            with self.assertRaises(dm.DatasetError):
                dm.load_frame_from_hf('owner/ds', split='train', max_rows=10, config='   ')
        finally:
            sys.modules.pop('datasets', None)

    def test_chained_cause_preserved_and_first_failure_logged(self):
        # Regression: DatasetGenerationError("An error occurred while
        # generating the dataset") hides the real cause in `from e`.
        # The loader must keep the chain in the error and log the first
        # failure's traceback server-side (stderr), without leaking paths
        # beyond the library's own message.
        import io
        import sys
        import types
        from contextlib import redirect_stderr
        fake = types.ModuleType('datasets')

        def load_dataset(dataset_id, name=None, split=None, streaming=False):
            try:
                raise OSError('disk full writing arrow file')
            except OSError as cause:
                raise Exception('An error occurred while generating the dataset') from cause

        fake.load_dataset = load_dataset
        sys.modules['datasets'] = fake
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                with self.assertRaises(dm.DatasetError) as ctx:
                    dm.load_frame_from_hf('owner/ds', split='train', max_rows=10)
            msg = str(ctx.exception)
            self.assertIn('Could not load Hugging Face dataset', msg)
            self.assertIn('caused by', msg)
            self.assertIn('OSError', msg)
            self.assertIn('Traceback', buf.getvalue())
        finally:
            sys.modules.pop('datasets', None)


class HfIngestErrorsTest(unittest.TestCase):
    """Ingestion failure taxonomy (no network except where noted).

    The classifier maps real exception shapes to user-facing codes without
    tracebacks; the split fallback is exercised with a stubbed loader.
    """

    def _classifier(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ml.experiment_runner import _classify_ingest_error
        return _classify_ingest_error

    def test_missing_dependency(self):
        code, msg = self._classifier()(ModuleNotFoundError("No module named 'datasets'"))
        self.assertEqual(code, 'hf_dependency_error')
        self.assertNotIn('Traceback', msg)

    def test_dataset_not_found_shapes(self):
        for text in ("Dataset 'x' doesn't exist on the Hub or cannot be accessed.",
                     "Couldn't find dataset 'x'", '404 Client Error'):
            code, msg = self._classifier()(ValueError(text))
            self.assertEqual(code, 'hf_dataset_not_found', text)
            self.assertEqual(msg, 'Dataset could not be found.')

    def test_gated_shapes(self):
        for text in ('GatedRepoError: You must be authenticated (401)',
                     '403 Forbidden for repo'):
            code, _ = self._classifier()(ValueError(text))
            self.assertEqual(code, 'hf_auth_error', text)

    def test_config_split_shapes(self):
        code, msg = self._classifier()(
            ValueError("BuilderConfig 'nope' not found. Available: ['default']"))
        self.assertEqual(code, 'hf_config_not_found')
        code, msg = self._classifier()(
            ValueError('Unknown split "zzz". Should be one of ["train", "test"]'))
        self.assertEqual(code, 'hf_split_not_found')
        self.assertIn('split', msg)

    def test_network_and_rate_shapes(self):
        code, _ = self._classifier()(ConnectionError('Connection aborted by peer'))
        self.assertEqual(code, 'hf_network_error')
        code, _ = self._classifier()(ValueError('429 Too Many Requests'))
        self.assertEqual(code, 'hf_rate_limited')

    def test_conversion_shape(self):
        code, msg = self._classifier()(
            ValueError('to_pandas failed: pyarrow.lib.ArrowInvalid: struct arrays'))
        self.assertEqual(code, 'hf_conversion_error')
        self.assertIn('tabular', msg)

    def test_split_fallback_uses_dataset_dict(self):
        # Direct split load fails -> loader falls back to DatasetDict select.
        import sys
        import types
        import pandas as pd
        calls = {}

        class FakeSplit:
            def __init__(self, rows):
                self._rows = rows

            def to_pandas(self):
                return pd.DataFrame(self._rows)

        class FakeDict(dict):
            pass

        fake = types.ModuleType('datasets')

        def load_dataset(dataset_id, name=None, split=None, streaming=False):
            calls.setdefault('calls', []).append((name, split))
            if split is not None:
                raise ValueError('Unknown split "%s". Should be one of ["train", "test"]' % split)
            d = FakeDict()
            d['train'] = FakeSplit([{'x': 1}, {'x': 2}])
            d['test'] = FakeSplit([{'x': 3}])
            return d

        fake.load_dataset = load_dataset
        sys.modules['datasets'] = fake
        try:
            df, info = dm.load_frame_from_hf('owner/ds', split='test', max_rows=10)
            self.assertEqual(len(df), 1)
            self.assertEqual(calls['calls'][0], (None, 'test'))
        finally:
            sys.modules.pop('datasets', None)

    def test_invalid_split_message(self):
        import sys
        import types
        fake = types.ModuleType('datasets')

        def load_dataset(dataset_id, name=None, split=None, streaming=False):
            if split is not None:
                raise ValueError('Unknown split "%s".' % split)
            return {'train': None}

        fake.load_dataset = load_dataset
        sys.modules['datasets'] = fake
        try:
            with self.assertRaises(dm.DatasetError) as ctx:
                dm.load_frame_from_hf('owner/ds', split='zzz', max_rows=10)
            self.assertIn('zzz', str(ctx.exception))
        finally:
            sys.modules.pop('datasets', None)

    def test_tabular_conversion_shapes(self):
        # Real datasets objects, no network: ClassLabel, text, list columns.
        from datasets import Dataset, Features, Value, ClassLabel, Sequence
        feats = Features({'text': Value('string'),
                          'label': ClassLabel(names=['neg', 'pos']),
                          'tags': Sequence(Value('int32'))})
        ds = Dataset.from_dict({'text': ['good', 'bad'], 'label': [1, 0],
                                'tags': [[1, 2], [3]]}, features=feats)
        df = ds.to_pandas()
        # Loader flattening keeps the frame tabular (lists/arrays -> JSON strings).
        import json as _json
        import numpy as _np
        for c in list(df.columns):
            if not len(df):
                break
            v0 = df[c].iloc[0]
            if isinstance(v0, _np.ndarray):
                df[c] = df[c].map(lambda v: _json.dumps(
                    v.tolist() if isinstance(v, _np.ndarray) else v, default=str))
            elif isinstance(v0, (list, dict)):
                df[c] = df[c].map(lambda v: _json.dumps(v, default=str))
        self.assertEqual(list(df.columns), ['text', 'label', 'tags'])
        self.assertEqual(df['tags'].iloc[0], '[1, 2]')
        self.assertTrue(set(df['label'].tolist()) <= {0, 1})

    def test_loader_flattens_ndarray_columns(self):
        # End-to-end through load_frame_from_hf with a stubbed loader whose
        # to_pandas yields numpy arrays (the previously missed case).
        import sys
        import types
        import numpy as _np
        import pandas as _pd

        class FakeSplit:
            def to_pandas(self):
                return _pd.DataFrame({'x': [1, 2],
                                      'tags': [_np.array([1, 2]), _np.array([3])]})

        fake = types.ModuleType('datasets')

        def load_dataset(dataset_id, name=None, split=None, streaming=False):
            return FakeSplit()

        fake.load_dataset = load_dataset
        sys.modules['datasets'] = fake
        try:
            df, _ = dm.load_frame_from_hf('owner/ds', split='train', max_rows=10)
            self.assertEqual(df['tags'].iloc[0], '[1, 2]')
            self.assertEqual(df['tags'].iloc[1], '[3]')
        finally:
            sys.modules.pop('datasets', None)


class RunnerImportRobustnessTest(unittest.TestCase):
    """experiment_runner must import without heavy deps, and every command
    must emit a structured dependency error (never a bare traceback) when
    a package is missing. Uses `python -S` (no site-packages) as the
    missing-dependency environment — no network needed."""

    def _run_nosite(self, *argv):
        import subprocess
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env['PYTHONPATH'] = repo + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        return subprocess.run(
            [sys.executable, '-S', os.path.join(repo, 'ml', 'experiment_runner.py')] + list(argv),
            capture_output=True, text=True, timeout=120, cwd=repo, env=env)

    def test_module_import_needs_no_heavy_deps(self):
        import subprocess
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env['PYTHONPATH'] = repo + (os.pathsep + env.get('PYTHONPATH', ''))
        r = subprocess.run(
            [sys.executable, '-S', '-c', 'import ml.experiment_runner; print("import OK")'],
            capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        self.assertIn('import OK', r.stdout)

    def test_missing_dep_emits_structured_error(self):
        r = self._run_nosite('dataset-ingest-hf', '--ws', 'mlw_aaaaaaaaaaaa',
                             '--workspace-base', tempfile.gettempdir(),
                             '--dataset-id', 'scikit-learn/iris',
                             '--config', 'default', '--split', 'train',
                             '--max-rows', '50000')
        self.assertNotEqual(r.returncode, 0)
        payload = json.loads(r.stdout.strip().split('\n')[-1])
        self.assertEqual(payload.get('type'), 'error')
        self.assertEqual(payload.get('code'), 'hf_dependency_error')
        self.assertIn('joblib', payload.get('error', ''))
        self.assertNotIn('Traceback', payload.get('error', ''))


if __name__ == '__main__':
    unittest.main(verbosity=2)
