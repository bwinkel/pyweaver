#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from cygrid import WcsGrid


__all__ = ['compute_bw_arrays', ]


def compute_bw_arrays(
        lons1, lats1, tvecs1, data1,
        lons2, lats2, tvecs2, data2,
        map_header,
        kernel_params,
        poly_order=1,
        ):
    '''
    Compute all basket-weaving helper arrays

    (lon/lat-dir and diff images, scan-line maps, BW matrix)
    TODO

    '''

    num_scans1 = len(lons1)
    num_scans2 = len(lons2)
    # TODO: add checks (i.e., all arrays have compatible dimension)

    # will grid each scan line separately (which is slower, but needs less
    # memory); may want to change this in the future!

    # three gridders are needed, one each for the lon- and lat-maps,
    # and one for the scan-line helper maps
    # TODO: it could be better (performance-wise) to only use one gridder
    # because the caches would not need to be re-computed

    gridders = [WcsGrid(map_header) for i in range(3)]
    for gridder in gridders:
        gridder.set_kernel(*kernel_params)

    for lons, lats, data in zip(lons1, lats1, data1):
        gridders[0].grid(lons, lats, data[:, np.newaxis])

    for lons, lats, data in zip(lons2, lats2, data2):
        gridders[1].grid(lons, lats, data[:, np.newaxis])

    zi1d = gridders[0].get_datacube().squeeze()
    zi2d = gridders[1].get_datacube().squeeze()
    wi1d = gridders[0].get_weights().squeeze()
    wi2d = gridders[1].get_weights().squeeze()

    bw_maps1 = np.empty((poly_order * num_scans1, ) + zi1d.shape)
    bw_maps2 = np.empty((poly_order * num_scans2, ) + zi1d.shape)

    for idx, (lons, lats, tvecs) in enumerate(zip(lons1, lats1, tvecs1)):
        for p in range(poly_order):

            gridders[2].clear_data_and_weights()
            gridders[2].grid(lons, lats, tvecs[p, :, np.newaxis])

            bw_map = gridders[2].get_unweighted_datacube().squeeze()
            bw_maps1[idx * poly_order + p] = bw_map

    for idx, (lons, lats, tvecs) in enumerate(zip(lons2, lats2, tvecs2)):
        for p in range(poly_order):

            gridders[2].clear_data_and_weights()
            gridders[2].grid(lons, lats, tvecs[p, :, np.newaxis])

            bw_map = gridders[2].get_unweighted_datacube().squeeze()
            # minus is important!
            bw_maps2[idx * poly_order + p] = -bw_map

    # note: BW matrix is effectively created by:
    # bw_mat = np.vstack([
    #     bw_maps1.reshape((poly_order * num_scans1, -1)),
    #     bw_maps2.reshape((poly_order * num_scans2, -1)),
    #     ]).T
    # if you want to look at the poly orders separately, do, e.g.
    # bw_p0 = bw_mat[:, 0::poly_order]
    # bw_p1 = bw_mat[:, 1::poly_order]
    # etc.

    return zi1d, wi1d, bw_maps1, zi2d, wi2d, bw_maps2


def compute_bw_matrices_and_vectors(
        zi1d, wi1d, bw_maps1,
        zi2d, wi2d, bw_maps2,
        poly_order=1,
        dampening=0.1,
        ):
    '''
    Produce both, reconstruction and solving matrix
    '''

    zid = zi1d - zi2d
    num_params1 = bw_maps1.shape[0]
    num_params2 = bw_maps2.shape[0]
    num_scans1 = num_params1 // poly_order
    num_scans2 = num_params2 // poly_order
    num_params = num_params1 + num_params2

    # result vector (containing the flattened difference map)
    dirty_vec = np.hstack([
        zid.flat,
        np.zeros((num_params, ), dtype=np.float64)
        ])

    _bw1 = bw_maps1.reshape((poly_order * num_scans1, -1)).T
    _bw2 = bw_maps2.reshape((poly_order * num_scans2, -1)).T

    # BW solving matrix; essentially one needs to correct the weighting,
    # because we didn't yet divide by the overall weightmaps!
    bw_mat = np.hstack([
        _bw1 / poly_order / np.abs(np.sum(_bw1, axis=1))[:, np.newaxis],
        _bw2 / poly_order / np.abs(np.sum(_bw2, axis=1))[:, np.newaxis],
        ])

    # BW reconstruction matrix
    bw_mat_r = np.hstack([_bw1, _bw2])

    # need to add the regularization submatrix to this
    reg_mat = np.eye(num_params, dtype=np.float64) * dampening
    A = np.vstack([bw_mat, reg_mat])
    rA = np.vstack([bw_mat_r, reg_mat])

    return dirty_vec, A, rA


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from helper import create_mock_data

    poly_order = 1
    beam_fwhm = 10 / 60
    kernel_sigma = beam_fwhm / 2 / np.sqrt(8 * np.log(2))
    kernel_params = (
        'gauss1d', (kernel_sigma, ), 3 * kernel_sigma, kernel_sigma / 2
        )

    map_header, mockdata1, mockdata2 = create_mock_data(
        map_size=(5, 5),
        beam_fwhm=beam_fwhm,
        map_rms=0.01,
        poly_order=poly_order,
        )

    pvec_input = np.hstack([
        np.array(m1.coeffs).flat,
        np.array(m2.coeffs).flat,
        ])

    zi1d, wi1d, bw_maps1, zi2d, wi2d, bw_maps2 = compute_bw_arrays(
        m1.lons, m1.lats, m1.tvecs, m1.dirty,
        m2.lons, m2.lats, m2.tvecs, m2.dirty,
        map_header,
        kernel_params,
        poly_order=poly_order,
        )

    # grid some of the aux maps, for testing purposes
    tmp_header = map_header.copy()
    tmp_header['NAXIS3'] = 3
    gridder = WcsGrid(tmp_header)
    gridder.set_kernel(*kernel_params)
    gridder.grid(
        np.hstack(m1.lons + m2.lons),
        np.hstack(m1.lats + m2.lats),
        np.vstack([
            np.hstack(m1.offsets + m2.offsets),
            np.hstack(m1.model + m2.model),
            np.hstack(m1.clean + m2.clean),
            ]).T
        )
    offset_map, model_map, clean_map = gridder.get_datacube()

    dirty_vec, A, rA = compute_bw_matrices_and_vectors(
        zi1d, wi1d, bw_maps1,
        zi2d, wi2d, bw_maps2,
        poly_order=poly_order,
        dampening=0.1,
        )

    # plt.imshow(zi1d, origin='lower', interpolation='nearest')
    # plt.show()

    presult = np.linalg.lstsq(A, dirty_vec, rcond=None)  # future behaviour
    # presult = np.linalg.lstsq(A, dirty_vec, rcond=-1)  # old behaviour

    pvec_final = presult[0]
    pvec_final = pvec_input
    # plt.plot(pvec_input, pvec_final, 'bx')
    # plt.show()

    # plt.plot(pvec_input, 'bx')
    # plt.plot(pvec_final, 'rx')
    # plt.show()

    num_rows = zi1d.size
    correction_map = (
        np.dot(np.abs(rA[:num_rows]), pvec_final) /
        poly_order /
        np.sum(np.abs(rA[:num_rows, ::poly_order]), axis=1)
        ).reshape(zi1d.shape)

    diff_map = zi1d - zi2d
    dirty_map = (zi1d * wi1d + zi2d * wi2d) / (wi1d + wi2d)
    cleaned_map = dirty_map - correction_map

    plt.close()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ims = [
        axes[idx // 4, idx % 4].imshow(
            dat, origin='lower', interpolation='nearest'
            )
        for idx, dat in enumerate([
            model_map, clean_map, dirty_map, offset_map,
            diff_map, correction_map, cleaned_map
            ])
        ]
    axes[-1, -1].plot(pvec_input, 'bx')
    axes[-1, -1].plot(pvec_final, 'rx')
    for im, ax, title in zip(
            ims, axes.flat, [
                'model_map', 'clean_map', 'dirty_map', 'offset_map',
                'diff_map', 'correction_map', 'rz'
                ]):
        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    fig.tight_layout()
    plt.show()
