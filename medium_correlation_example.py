
# %%

import glob, os
import numpy as np
from timeit import default_timer
import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()" +
    "\t%(levelname)s\t%(message)s")

from obspy import UTCDateTime, read
from obspy.core import Stream
from eqcorrscan.utils.pre_processing import (
    dayproc, _prep_data_for_correlation, shortproc)
from eqcorrscan.utils.correlate import (
    _time_threaded_normxcorr, _get_array_dicts)
from eqcorrscan.core.match_filter import Tribe
from fast_matched_filter import matched_filter as fmf
#from fmf2 import matched_filter as fmf2

# Load example streams (24 hours of continuous seismograms for ca. 330 channels)
# Logger.info('Preparing data')
# t1 = UTCDateTime("2022-04-25T00:00:00.000000Z")
# t2 = t1 + 24 * 60 * 60
# mseed_files = glob.glob(os.path.join('Data', 'NS.*115'))
# stream = Stream()
# for mseed_file in mseed_files:
#     stream += read(mseed_file)
# # Preprocess, filter data
# stream.detrend()
# stream.merge(method=1, fill_value=0, interpolation_samples=-1)
# stream = dayproc(
#     stream, lowcut=3.0, highcut=9.0, filt_order=4, samp_rate=20.0,
#     starttime=t1)
# # Rename traces so they fit with the templates
# for tr in stream:
#     tr.stats.network = 'NS'
#     tr.stats.location = '00'
#     tr.stats.channel = 'BH' + tr.stats.channel[-1]

# Load the preprocessed stream to quickly start up FMF
stream = read('sample_stream_20220426.mseed')
stream_seed_ids = list(set([tr.id for tr in stream]))

# Load example tribe of templates, for 16 ridge earthquakes on 2022-04-25
tribe = Tribe().read('data/mohn_cluster_tribe_20220425.tgz')
template_streams = [template.st for template in tribe]
new_template_streams = []
for template_stream in template_streams:
    template_stream = Stream([
        tr for tr in template_stream if tr.id in stream_seed_ids])
    if len(template_stream) > 0:
        new_template_streams.append(template_stream)
template_names = [template.name for template in tribe]


# %% Reshape data and directly invoce cross-correlation computation backend
stream, templates, _ = _prep_data_for_correlation(
    stream=stream, templates=new_template_streams,
    template_names=template_names)

# Get dicts that describe which template and data are available for CC
array_dict_tuple = _get_array_dicts(templates, stream, stack=True)

stream_dict, template_dict, pad_dict, seed_ids = array_dict_tuple
# Reshape templates into [templates x traces x time]
template_arr = np.array([template_dict[seed_id]
                         for seed_id in seed_ids]).swapaxes(0, 1)
# Reshape stream into [traces x time]
data_arr = np.array([stream_dict[seed_id] for seed_id in seed_ids])
# Moveouts should be [templates x traces]
pads = np.array([pad_dict[seed_id] for seed_id in seed_ids]).swapaxes(0, 1)
# Weights should be shaped like pads
weights = np.ones_like(pads)

# Demean
template_arr -= template_arr.mean(axis=-1, keepdims=True)
data_arr -= data_arr.mean(axis=-1, keepdims=True)

# Compute correlation coefficients direclty by invoking FMF
# arch can be 'precise' (runs on CPU) or 'gpu' (runs on CUDA GPU)
Logger.info('Starting FMF correlation run')
outtic = default_timer()
fmf_cccsums = fmf(
    templates=template_arr, weights=weights, moveouts=pads, check_zeros='all',
    data=data_arr, step=1,
    arch='gpu', normalize="full")
    #arch='precise', normalize="full")
    # arch='cpu')

# cc_out = _fmf_gpu(templates, stream)  # can be called as wrapper instead
outtoc = default_timer()
Logger.info('FastMatchedFilter correlations took: {0:.4f}s'.format(
    outtoc - outtic))

# test FMF2:
Logger.info('Starting FMF2 correlation run')
outtic = default_timer()
fmf2_cccsums = fmf(
    templates=template_arr, weights=weights, moveouts=pads, check_zeros='all',
    data=data_arr, step=1,
    # arch='sycl', normalize="full")
    arch='gpu', normalize="full")
    # arch='cpu')

# cc_out = _fmf_gpu(templates, stream)  # can be called as wrapper instead
outtoc = default_timer()
Logger.info('FMF2 correlations took: {0:.4f}s'.format(
    outtoc - outtic))

for atol in np.arange(0.05, 1.0, 0.05):
    try:
        assert np.allclose(fmf_cccsums, fmf2_cccsums, atol=atol)
    except AssertionError:
        Logger.error(
            'FMF and FMF2 results not similar enough within atol=%s', atol)


# %%

# For comparing results, you can load in time-domain results from
# numpy-file:
td_cccsums = np.load('time_domain_cccsums.npy')
# Or you can recompute with the EQcorrscan- time domain backend:
# outtic = default_timer()
# cc_ref = _time_threaded_normxcorr(templates, stream)
# outtoc = default_timer()
# Logger.info('EQcorrscan time domain correlations took: {0:.4f}s'.format(
#     outtoc - outtic))
# td_cccsums = cc_ref[0]
# np.save('time_domain_cccsums.npy', td_cccsums)

# Compare outputs
for atol in np.arange(0.05, 1.0, 0.05):
    try:
        assert np.allclose(fmf_cccsums, td_cccsums, atol=atol)
    except AssertionError:
        Logger.error(
            'FMF and TD results not similar enough within atol=%s', atol)
    try:
        assert np.allclose(fmf2_cccsums, td_cccsums, atol=atol)
    except AssertionError:
        Logger.error(
            'FMF2 and TD results not similar enough within atol=%s', atol)



# %%
