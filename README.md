HOW TO INSTALL AND RUN FMF HIP ON LUMI:

Replace original Makefile with Makefile-HIP. Use make python_gpu (python_cpu if you want the cpu implementation too)

Install conda environment with the following:
```
ml LUMI/23.03  partition/G
ml lumi-container-wrapper

mkdir FMF_env
conda-containerize new --prefix FMF_env env.yaml
```
env.yaml:
```
channels:
  - conda-forge
dependencies:
  - numpy=1.24.4
  - eqcorrscan=0.4.4
```
Working conda environment is also located in:
```
export PATH="/project/project_465000096/FMF-HIP/fmf_env/bin:$PATH"
```

We have 3 ways to test the code:

Built-in tests that are quick to run and will only test the FMF module located in /fast_matched_filter/fast_matched_filter/tests

Medium data example: Fast running example that requires EQcorrscan from conda environment.

Big data example: The challenge is to speed this up.

All data and code implementations are located in project directory:

Big data:
```
/project/project_465000096/FMF_test_big_share-SYCL
```
Medium data:
```
/project/project_465000096/FMF_test_medium_share
```
Big/medium/tests ready to run with HIP:
```
/project/project_465000096/FMF-HIP
````
If you are running the big data example I suggest using this implementation:

```
import numpy as np
import pickle
from timeit import default_timer
from fast_matched_filter import matched_filter as fmf
from eqcorrscan.utils.correlate import _get_array_dicts
import logging
import gc

NUM_TEMPLATES = 5269
def read_data():
    with open('templates_5269.pickle', "rb") as template_file:
        templates = pickle.load(template_file)
    with open('stream_2006-01-01_181.pickle', "rb") as stream_file:
        stream = pickle.load(stream_file)
    return templates, stream

def setup_templates(templates, stream):
    # Get dicts that describe which template and data are available for CC
    array_dict_tuple = _get_array_dicts(templates, stream, stack=True)
    del templates, stream
    gc.collect()
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

def match_filter(template_array, weight, pad, data_array):
    return fmf(templates=template_array, weights=weight, moveouts=pad, check_zeros='all',
                data=data_array, step=1,
                arch='gpu', normalize="full")
def main():
    Logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()" +
        "\t%(levelname)s\t%(message)s")

    Logger.info('Reading data')
    
    templates, stream = read_data()
    del templates[NUM_TEMPLATES:]
    template_arr, data_arr, weights, pads = setup_templates(templates, stream)
    del templates, stream
    outtic = default_timer()
    gc.collect()
    Logger.info(f"Starting FMF correlation run with {NUM_TEMPLATES} templates")
    cccsums = match_filter(template_arr, weights, pads, data_arr)
    del template_arr, weights, pads, data_arr
    gc.collect()

    outtoc = default_timer()

    Logger.info('FMF spent: {0:.4f}s'.format(outtoc - outtic))

if __name__ == "__main__":
    main()
```

Here you can adjust the number of templates you want to run by editing the NUM_TEMPLATES variable.


Sample run script:
```
#!/bin/bash -e
#SBATCH --job-name=fmf.hip
#SBATCH --account=project_465000096
#SBATCH --time=00:45:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --mem=124G
#SBATCH --gpus-per-node=8 
#SBATCH --cpus-per-task=8
#SBATCH -o %x-%j.out

ml rocm
ml LUMI/23.03  partition/G
ml lumi-container-wrapper

export PATH="/project/project_465000096/FMF-HIP/fmf_env/bin:$PATH"

time python big_correlation_example.py
```

Best performance using full big data set is currently achieved by using 8 gpus, 1 node and it also requires at least 105 GB of RAM to not run out of memory.





||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# fast_matched_filter (FMF)
An efficient seismic matched-filter search for both CPU and GPU architectures. Documentation at https://ebeauce.github.io/FMF_documentation/.

If you use FMF in research to be published, please reference the following article: Beaucé, Eric, W. B. Frank, and Alexey Romanenko (2017). Fast matched-filter (FMF): an efficient seismic matched-filter search for both CPU and GPU architectures. _Seismological Research Letters_, doi: [10.1785/0220170181](https://doi.org/10.1785/0220170181)

FMF is available at https://github.com/beridel/fast_matched_filter and can be downloaded with:<br>

    git clone https://github.com/beridel/fast_matched_filter.git

## Required software/hardware
- A C compiler that supports OpenMP (default Mac OS compiler clang does not support OpenMP; gcc can be easily downloaded via homebrew)
- CPU version: either Python (v2.7 or 3.x) or Matlab
- GPU version: Python (v2.7 or 3.x) and a discrete Nvidia graphics card that supports CUDA C with CUDA toolkit installed

## Installation

### From source
A simple make + whichever implementation does the trick. Possible make commands are:<br>

    make python_cpu
    make python_gpu
    make matlab

NB: Matlab compiles via mex, which needs to be setup before running. Any compiler can be chosen during the setup of mex, because it will be bypassed by the CC environment variable in the Makefile. Therefore CC must be set to an OpenMP-compatible compiler.


### Using pip

Installation as a Python module is possible via pip (which supports clean uninstalling):<br>

    python setup.py build_ext
    pip install .

or simply:<br>

    pip install git+https://github.com/beridel/fast_matched_filter

## Running

Python: Both CPU and GPU versions are called with the matched_filter function.<br>
If FMF was installed with pip:
```python
    import fast_matched_filter as fmf
```

Matlab: The CPU version is called with the fast_matched_filter function

## To do:

- [ ] Update Matlab wrapper.
