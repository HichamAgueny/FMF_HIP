import numpy as np
import pickle
from timeit import default_timer
from fast_matched_filter import matched_filter as fmf
from eqcorrscan.utils.correlate import _get_array_dicts
import gc
import logging
#from memory_profiler import profile
NUM_TEMPLATES = 200
#@profile
def read_data():
    with open('templates_5269.pickle', "rb") as template_file:
        template_data = pickle.load(template_file)
    with open('stream_2006-01-01_181.pickle', "rb") as stream_file:
        stream = pickle.load(stream_file)
    return template_data, stream
#@profile
def setup_templates(template_data, stream):
    # Get dicts that describe which template and data are available for CC
    array_dict_tuple = _get_array_dicts(template_data, stream, stack=True)
    del template_data, stream
    stream_dict, template_dict, pad_dict, seed_ids = array_dict_tuple
    del array_dict_tuple
    # Reshape templates into [templates x traces x time]
    template_arr = np.array([template_dict[seed_id] for seed_id in seed_ids], dtype=np.float32).swapaxes(0, 1)
    # Reshape stream into [traces x time]
    data_arr = np.array([stream_dict[seed_id] for seed_id in seed_ids], dtype=np.float32)
    # Moveouts should be [templates x traces]
    pads = np.array([pad_dict[seed_id] for seed_id in seed_ids], dtype=np.float32).swapaxes(0, 1)
    # Weights should be shaped like pads
    weights = np.ones_like(pads)
    template_arr -= template_arr.mean(axis=-1, keepdims=True)
    data_arr -= data_arr.mean(axis=-1, keepdims=True)

    del stream_dict, template_dict, pad_dict, seed_ids
    return template_arr, data_arr, weights, pads
#@profile
def main():
    Logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()" +
        "\t%(levelname)s\t%(message)s")
    
    Logger.info('Reading data')
    template_data, stream = read_data()
    Logger.info('Data is read')
    
    del template_data[NUM_TEMPLATES:]
    Logger.info('Setting up templates')
    template_arr, data_arr, weights, pads = setup_templates(template_data, stream)
    del template_data, stream
    gc.collect()
    Logger.info(f"Starting FMF correlation run with {NUM_TEMPLATES} templates")

    outtic = default_timer()
    cccsums = fmf(templates=template_arr, weights=weights, moveouts=pads, check_zeros='all',
                data=data_arr, step=1,
                arch='gpu', normalize="full")
    print(len(cccsums[0]))
    print(len(cccsums))
    np.savetxt('ccsum_test.out', cccsums[0:2])
    del template_arr, weights, pads, data_arr
    gc.collect()
    outtoc = default_timer()

    Logger.info('FMF spent: {0:.4f}s'.format(outtoc - outtic))

if __name__ == "__main__":
    main()
