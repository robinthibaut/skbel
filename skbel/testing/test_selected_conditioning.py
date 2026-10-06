"""Regression tests for selected-observation KDE/TM conditioning."""

import unittest
from unittest import mock

import numpy as np
from sklearn.linear_model import LinearRegression

from skbel import BEL

X_OBS = np.array([[10.0], [20.0]])
N_POSTS = 3


def _linear(offset):
    x = np.array([[0.0], [1.0], [2.0]])
    return {"kind": "linear", "function": LinearRegression().fit(x, x + offset), "bandwidth": 0}


def _pdf(tag):
    return {"kind": "pdf", "function": mock.Mock(x=np.array([0.0, 1.0]), tag=tag), "bandwidth": 1}


def _bel(mode, functions):
    bel = BEL(mode=mode)
    bel.n_posts = N_POSTS
    bel.seed = 1
    arr = np.zeros((len(functions), len(functions[0])), dtype=object)
    for i, row in enumerate(functions):
        for j, f in enumerate(row):
            arr[i, j] = f
    setattr(bel, "kde_functions" if mode == "kde" else "tm_functions", arr)
    return bel


def _run(bel, *args, **kwargs):
    with mock.patch("skbel.learning.bel.check_is_fitted"):
        return bel.random_sample(*args, **kwargs)


class _FakeTM:
    def __init__(self, shift):
        self.shift = shift
        self.calls = []

    def map(self, X):
        return np.zeros((X.shape[0], 1))

    def inverse_map(self, X_precalc, Y):
        self.calls.append((X_precalc.copy(), Y.copy()))
        return X_precalc + self.shift


class SelectedLinearKDETest(unittest.TestCase):
    def setUp(self):
        self.bel = _bel("kde", [[_linear(100)], [_linear(200)]])

    def test_batch_unchanged(self):
        out = _run(self.bel, X_OBS)
        self.assertEqual(out.shape, (2, N_POSTS, 1))
        np.testing.assert_allclose(out[:, 0, 0], [110, 220])

    def test_selected_indices(self):
        for idx, expected in ((0, 110), (1, 220), (-1, 220), (-2, 110)):
            out = _run(self.bel, X_OBS, obs_n=idx)
            self.assertEqual(out.shape, (1, N_POSTS, 1))
            np.testing.assert_allclose(out, expected)

    def test_caller_array_unchanged(self):
        x = X_OBS.copy()
        _run(self.bel, x, obs_n=1)
        np.testing.assert_array_equal(x, X_OBS)

    def test_kde_init_selected(self):
        for idx, expected in ((0, 110), (1, 220), (-1, 220)):
            init = self.bel.kde_init(X_OBS, obs_n=idx)
            self.assertEqual(init.shape, (1, 1))
            np.testing.assert_allclose(init[0, 0], expected)
        self.assertEqual(self.bel.kde_init(X_OBS).shape, (2, 1))

    def test_cached_selected(self):
        full = self.bel.kde_init(X_OBS)
        for idx, expected in ((0, 110), (1, 220), (-1, 220)):
            for cache in (full, self.bel.kde_init(X_OBS, obs_n=idx)):
                out = _run(self.bel, X_OBS, obs_n=idx, init_kde=cache)
                self.assertEqual(out.shape, (1, N_POSTS, 1))
                np.testing.assert_allclose(out, expected)

    def test_cached_batch(self):
        out = _run(self.bel, X_OBS, init_kde=self.bel.kde_init(X_OBS))
        np.testing.assert_allclose(out[:, :, 0], [[110] * 3, [220] * 3])

    def test_index_errors(self):
        for bad in (2, -3, 1.0, True):
            with self.assertRaises(IndexError):
                _run(self.bel, X_OBS, obs_n=bad)
            with self.assertRaises(IndexError):
                self.bel.kde_init(X_OBS, obs_n=bad)

    def test_two_components(self):
        bel = _bel("kde", [[_linear(100), _linear(300)], [_linear(200), _linear(400)]])
        x = np.array([[10.0, 1.0], [20.0, 2.0]])
        out = _run(bel, x, obs_n=1)
        self.assertEqual(out.shape, (1, N_POSTS, 2))
        np.testing.assert_allclose(out[0, 0], [220, 402])
        self.assertEqual(bel.kde_init(x, obs_n=-1).shape, (1, 2))


class SelectedPdfKDETest(unittest.TestCase):
    def setUp(self):
        self.fns = [[_pdf("a")], [_pdf("b")]]
        self.bel = _bel("kde", self.fns)

    def _sampler(self, pdf, num_samples=None, **kw):
        return np.full(num_samples, 1.0 if pdf.tag == "a" else 2.0)

    def test_selected_uses_selected_pdf(self):
        with mock.patch("skbel.learning.bel.it_sampling", side_effect=self._sampler) as m:
            out = _run(self.bel, X_OBS, obs_n=1)
        self.assertEqual(out.shape, (1, N_POSTS, 1))
        np.testing.assert_allclose(out, 2.0)
        self.assertEqual(m.call_count, 1)
        self.assertEqual(m.call_args.kwargs["pdf"].tag, "b")

    def test_cached_cdf_aligned(self):
        init = np.zeros((2, 1), dtype=object)
        init[0, 0], init[1, 0] = "cdf_a", "cdf_b"
        with mock.patch("skbel.learning.bel.it_sampling", side_effect=self._sampler) as m:
            _run(self.bel, X_OBS, obs_n=-1, init_kde=init)
            self.assertEqual(m.call_args.kwargs["cdf_y"], "cdf_b")
            _run(self.bel, X_OBS, obs_n=0, init_kde=init[:1])
            self.assertEqual(m.call_args.kwargs["cdf_y"], "cdf_a")

    def test_kde_init_selected(self):
        with mock.patch("skbel.learning.bel.it_sampling", return_value="cdf") as m:
            init = self.bel.kde_init(X_OBS, obs_n=1)
        self.assertEqual(init.shape, (1, 1))
        self.assertEqual(m.call_count, 1)
        self.assertEqual(m.call_args.kwargs["pdf"].tag, "b")

    def test_two_components(self):
        bel = _bel("kde", [[_pdf("a"), _pdf("b")], [_pdf("b"), _pdf("a")]])
        with mock.patch("skbel.learning.bel.it_sampling", side_effect=self._sampler):
            out = _run(bel, np.zeros((2, 2)), obs_n=1)
        self.assertEqual(out.shape, (1, N_POSTS, 2))
        np.testing.assert_allclose(out[0, 0], [2.0, 1.0])


class SelectedTMTest(unittest.TestCase):
    def setUp(self):
        self.tms = [_FakeTM(100), _FakeTM(200)]
        X = np.zeros((N_POSTS, 2))
        funs = [[{"kind": "tm", "function": tm, "X": X}] for tm in self.tms]
        self.bel = _bel("tm", funs)

    def test_batch(self):
        out = _run(self.bel, X_OBS)
        self.assertEqual(out.shape, (2, N_POSTS, 1))
        np.testing.assert_allclose(out[:, 0, 0], [110, 220])

    def test_selected(self):
        for idx, which, expected in ((0, 0, 110), (1, 1, 220), (-1, 1, 220)):
            for tm in self.tms:
                tm.calls.clear()
            out = _run(self.bel, X_OBS, obs_n=idx)
            self.assertEqual(out.shape, (1, N_POSTS, 1))
            np.testing.assert_allclose(out, expected)
            self.assertEqual(len(self.tms[which].calls), 1)
            self.assertEqual(len(self.tms[1 - which].calls), 0)
            pre, y = self.tms[which].calls[0]
            np.testing.assert_allclose(pre, X_OBS[idx, 0])
            self.assertEqual(y.shape, (N_POSTS, 1))

    def test_index_errors(self):
        for bad in (2, -3):
            with self.assertRaises(IndexError):
                _run(self.bel, X_OBS, obs_n=bad)


if __name__ == "__main__":
    unittest.main()
