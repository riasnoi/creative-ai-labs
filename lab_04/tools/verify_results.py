"""Независимая проверка точных массивов Float64, усреднения, PSNR и контраста."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.results_dir
    results = tomllib.loads((root / 'results.toml').read_text(encoding='utf-8'))
    expectations = tomllib.loads((root / 'expectations.toml').read_text(encoding='utf-8'))
    assert results['variant'] == expectations['variant'] == 13
    assert results['seeds'] == [101, 102, 103, 104, 105]
    assert expectations['recorded_at_msk'] <= results['started_at_msk'] <= results['finished_at_msk']
    assert expectations['practical_threshold_db'] == results['threshold_db'] == 1
    dtype = '<f8' if results['environment']['endian'] == 'little' else '>f8'

    def read_array(name):
        data = (root / name).read_bytes()
        assert len(data) == 128 * 128 * 8, name
        return np.frombuffer(data, dtype=dtype).reshape((128, 128), order='F')

    def close(a, b):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)

    def metrics(a, ref):
        mse = np.mean((a - ref) ** 2)
        return {'mse': float(mse), 'psnr': float(-10 * np.log10(mse)),
                'contrast': float(np.std(a, ddof=1))}

    reference = read_array('reference.f64')
    i, j = np.mgrid[1:129, 1:129]
    theory = 0.10 + 0.30 * (i + j) / 256
    theory[(i - 64) ** 2 + (j - 64) ** 2 <= 32 ** 2] = 0.75
    close(reference, theory)
    assert hashlib.sha256((root / 'reference.f64').read_bytes()).hexdigest() == results['reference']['sha256']
    records = results['per_seed']
    assert [r['seed'] for r in records] == results['seeds']
    before_metrics, after_metrics, filter_errors = [], [], []
    for record in records:
        seed = record['seed']
        arrays = {}
        for name, collection in [('before', before_metrics), ('after', after_metrics)]:
            filename = f'{name}_{seed}.f64'
            data = (root / filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == record[f'{name}_sha256']
            a = read_array(filename)
            assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
            calculated = metrics(a, reference)
            for key, value in calculated.items():
                close(value, record[name][key])
            collection.append(calculated)
            arrays[name] = a
            if seed == 101:
                png = np.asarray(Image.open(root / f'{name}.png'))
                np.testing.assert_array_equal(png, np.rint(a * 255).astype(np.uint8))
        padded = np.pad(arrays['before'], 2, mode='edge')
        filtered = np.lib.stride_tricks.sliding_window_view(padded, (5, 5)).mean(axis=(-1, -2))
        error = float(np.max(np.abs(filtered - arrays['after'])))
        assert error < 1e-12
        filter_errors.append(error)
        close(after_metrics[-1]['psnr'] - before_metrics[-1]['psnr'], record['delta_psnr'])
    for row in results['rows']:
        metric = row['metric']
        # Энергия Собеля остаётся предметом проверки в Julia; здесь независимы PSNR и контраст.
        if metric not in {'psnr', 'contrast'}:
            continue
        b = np.array([r[metric] for r in before_metrics])
        a = np.array([r[metric] for r in after_metrics])
        delta = a.mean() - b.mean()
        std = b.std(ddof=1)
        for value, key in [(b.mean(), 'base'), (std, 's'), (a.mean(), 'after'),
                           (delta, 'delta'), (abs(delta) / std, 'ratio')]:
            close(value, row[key])
        assert bool(abs(delta) > 2 * std) == row['exceeds_2s']
    np.testing.assert_array_equal(np.asarray(Image.open(root / 'reference.png')),
                                  np.rint(reference * 255).astype(np.uint8))
    delta = np.mean([r['psnr'] for r in after_metrics]) - np.mean([r['psnr'] for r in before_metrics])
    expected_class = 'полезна' if delta >= 1 else 'вредна' if delta <= -1 else 'нейтральна'
    assert results['classification'] == expected_class
    report = {'status': 'passed', 'seeds': results['seeds'],
              'checks': ['reference geometry', 'array SHA-256', 'range and finite values',
                         'replicate box filter', 'MSE and PSNR', 'sample contrast',
                         'summary PSNR and contrast', 'PNG quantization',
                         'recording order', 'classification'],
              'filter_max_abs_error': max(filter_errors), 'delta_psnr_db': float(delta),
              'classification': expected_class,
              'limitation': 'Независимая проверка генератора Julia и энергии Собеля здесь не выполняется.'}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
