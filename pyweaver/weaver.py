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
        porder=1,
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

    bw_maps = np.empty((porder * (num_scans1 + num_scans2), ) + zi1d.shape)

    for idx, (lons, lats, tvecs) in enumerate(zip(lons1, lats1, tvecs1)):
        for p in range(porder):

            gridders[2].clear_data_and_weights()
            gridders[2].grid(lons, lats, tvecs[p, :, np.newaxis])

            bw_map = gridders[2].get_unweighted_datacube().squeeze()
            bw_maps[idx * porder + p] = bw_map

    for idx, (lons, lats, tvecs) in enumerate(zip(lons2, lats2, tvecs2)):
        for p in range(porder):

            gridders[2].clear_data_and_weights()
            gridders[2].grid(lons, lats, tvecs[p, :, np.newaxis])

            bw_map = gridders[2].get_unweighted_datacube().squeeze()
            # minus is important!
            bw_maps[(idx + num_scans1) * porder + p] = -bw_map

    # note: BW matrix is effectively created by:
    # bw_mat = bw_maps.reshape((porder * (num_scans1 + num_scans2), -1)).T
    # if you want to look at the poly orders separately, do, e.g.
    # bw_p0 = bw_maps[0::2].reshape((num_scans1 + num_scans2, -1)).T
    # bw_p1 = bw_maps[1::2].reshape((num_scans1 + num_scans2, -1)).T

    return zi1d, wi1d, zi2d, wi2d, bw_maps


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    beam_fwhm = 10 / 60
    kernel_sigma = beam_fwhm / 2 / np.sqrt(8 * np.log(2))
    kernel_params = (
        'gauss1d', (kernel_sigma, ), 3 * kernel_sigma, kernel_sigma / 2
        )

    map_header, mockdata1, mockdata2 = create_mock_data(
        map_size=(5, 5),
        beam_fwhm=beam_fwhm,
        )

    lons1 = [m1.lons for m1 in mockdata1]
    lats1 = [m1.lats for m1 in mockdata1]
    tvecs1 = [m1.tvecs for m1 in mockdata1]
    data1 = [m1.dirty for m1 in mockdata1]

    lons2 = [m2.lons for m2 in mockdata2]
    lats2 = [m2.lats for m2 in mockdata2]
    tvecs2 = [m2.tvecs for m2 in mockdata2]
    data2 = [m2.dirty for m2 in mockdata2]

    zi1d, wi1d, zi2d, wi2d, bw_maps = compute_bw_arrays(
        lons1, lats1, tvecs1, data1,
        lons2, lats2, tvecs2, data2,
        map_header,
        kernel_params,
        porder=2,
        )
