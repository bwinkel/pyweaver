#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from astropy.utils.misc import NumpyRNGContext


__all__ = ['create_mock_data', ]


def create_mock_data(
        map_size=(5, 5),
        beam_fwhm=10 / 60,
        grid_kernel_fwhm=None,
        pixel_size=None,
        num_scans1=None,
        num_scans2=None,
        samples_per_scan1=None,
        samples_per_scan2=None,
        map_rms=1,
        offset_rms=1,
        poly_order=2,
        # baseline_coeffs=((0, ), (0, )),
        rng_seed=0,
        ):
    '''
    TODO
    '''

    lon_size, lat_size = map_size

    if grid_kernel_fwhm is None:
        grid_kernel_fwhm = beam_fwhm / 2

    if pixel_size is None:
        pixel_size = beam_fwhm / 3

    naxis1 = int(lon_size / pixel_size + 0.5)
    naxis2 = int(lat_size / pixel_size + 0.5)

    if num_scans1 is None:
        num_scans1 = naxis1

    if num_scans2 is None:
        num_scans2 = naxis2

    if samples_per_scan1 is None:
        samples_per_scan1 = naxis1

    if samples_per_scan2 is None:
        samples_per_scan2 = naxis2

    # generating WCS header
    map_header = {
        'NAXIS': 3,
        'NAXIS1': naxis1,
        'NAXIS2': naxis2,
        'NAXIS3': 1,
        'CDELT1': -pixel_size,
        'CRPIX1': (naxis1 + 1) / 2,
        'CRVAL1': 0.,
        'CTYPE1': 'RA---CAR',
        'CUNIT1': 'deg',
        'CDELT2': pixel_size,
        'CRPIX2': (naxis2 + 1) / 2,
        'CRVAL2': 0.,
        'CTYPE2': 'DEC--CAR',
        'CUNIT2': 'deg',
        }

    # lon and lat coordinates for each of the scan lines
    # these are stored as lists (of arrays per scan line), because
    # in the real world each scan line could have a different number
    # of samples; want to test that this works!

    # assume 1st coverage has scans in longitudinal direction
    lons1, lats1 = [], []
    for i, _lat in enumerate(
            np.linspace(-lat_size / 2, lat_size / 2, num_scans1)
            ):
        # increase sample num by 1, if odd scan line
        tmp = np.linspace(
            -lon_size / 2, lon_size / 2, samples_per_scan1 + i % 2
            )
        lons1.append(tmp)
        lats1.append(np.full_like(tmp, _lat))

    # assume 2nd coverage has scans in latitudinal direction
    lons2, lats2 = [], []
    for i, _lon in enumerate(
            np.linspace(-lon_size / 2, lon_size / 2, num_scans2)
            ):
        # increase sample num by 1, if odd scan line
        tmp = np.linspace(
            -lat_size / 2, lat_size / 2, samples_per_scan2 + i % 2
            )
        lons2.append(np.full_like(tmp, _lon))
        lats2.append(tmp)

    # generating artificial raw data

    # TODO: allow to add a (different) baseline to both data sets
    # (which would mimic different ground contributions etc.)

    def generate_data(lons, lats, num_scans):

        def gauss2d(l, b, A, l0, b0, s):
            return (
                A * np.exp(-((l - l0) ** 2 + (b - b0) ** 2) / 2 / s ** 2) /
                2 / np.pi / s ** 2
                )

        gauss_coeffs = [
            (25, 0, 0, 15 * beam_fwhm),
            (0.125, 1, 0, beam_fwhm),
            (0.25, 0, 1, beam_fwhm),
            (2.5, -1, -1, 5 * beam_fwhm),
            ]

        MockData = namedtuple(
            'MockData', [
                'lons', 'lats', 'coeffs',
                'tvecs', 'offsets', 'model', 'noise',
                # 'baseline',
                'clean', 'dirty'
                ])
        mockdata = []

        with NumpyRNGContext(rng_seed):

            # prepare scan-line offset coefficients
            coeffs = np.random.normal(0, offset_rms, (num_scans, poly_order))

            for _lons, _lats, _coeffs in zip(lons, lats, coeffs):

                _len = len(_lons)
                # (1) noise
                _noise = np.random.normal(0, map_rms, _len)

                # (2) polynomial offsets (per scan line); also store tvecs
                #     (the tvecs define the polynomial basis)
                _tvec = np.linspace(-1, 1, _len)
                # TODO: allow other types of polynomials?
                _tvecs = np.array([
                    np.power(_tvec, p)
                    for p in range(len(_coeffs))
                    ])
                _offsets = np.polyval(_coeffs, _tvec)

                # (3) generating source signal (a combination of some
                #     gaussians, and a 2D polynomial)

                _model = np.sum(
                    (gauss2d(_lons, _lats, *_c) for _c in gauss_coeffs),
                    axis=1,
                    )
                _model += (
                    1 +
                    0.1 * _lons + 0.05 * (_lons - 1) ** 2 +
                    0.2 * _lats + 0.02 * (_lats + 0.5) ** 2
                    )

                # (4) put everything together for clean and dirty "maps"

                mockdata.append(MockData(
                    lons=_lons,
                    lats=_lats,
                    coeffs=_coeffs,
                    tvecs=_tvecs,
                    offsets=_offsets,
                    model=_model,
                    noise=_noise,
                    clean=_model + _noise,
                    dirty=_model + _offsets + _noise
                    ))

            return mockdata

    mockdata1 = generate_data(lons1, lats1, num_scans1)
    mockdata2 = generate_data(lons2, lats2, num_scans2)

    return map_header, mockdata1, mockdata2


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    map_header, mockdata1, mockdata2 = create_mock_data(
        map_size=(5, 5),
        # map_size=(3, 3),
        beam_fwhm=10 / 60,
        grid_kernel_fwhm=None,
        pixel_size=None,
        num_scans1=None,
        num_scans2=None,
        samples_per_scan1=None,
        samples_per_scan2=None,
        map_rms=1,
        offset_rms=1,
        poly_order=2,
        # baseline_coeffs=((0, ), (0, )),
        rng_seed=0,
        )

    print(map_header)

    plt.scatter(
        np.hstack((m1.lons for m1 in mockdata1)),
        np.hstack((m1.lats for m1 in mockdata1)),
        c=np.hstack((m1.dirty for m1 in mockdata1)),
        )
    plt.show()
