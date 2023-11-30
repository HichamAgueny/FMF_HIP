/*
:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

#include <hip/hip_runtime.h>
// #include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

#define BLOCKSIZE 512
#define WARPSIZE 32
#define NCHUNKS 20
#define STABILITY_THRESHOLD 0.000001f
#define MEGABYTES pow(1024, 2)

extern "C"
{ // needed for C-style symbols in shared object compiled by nvcc
#include "matched_filter_GPU.h"

    //-------------------------------------------------------------------------
    inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
    {

        if (code != hipSuccess)
        {
            fprintf(stderr, "An error occured in the kernel: %s %s %d\n", hipGetErrorString(code), file, line);
            if (abort)
                exit(code);
        }
    }

    //-------------------------------------------------------------------------
    __global__ void network_corr(float *templates, float *sum_square_template,
                                 int *moveout, float *data, float *weights,
                                 size_t step, size_t n_samples_template, size_t n_samples_data,
                                 size_t n_stations, size_t n_components,
                                 size_t chunk_offset, size_t chunk_size,
                                 float *cc_mat, int normalize) {

        // each thread matches the template to one time in the data
        size_t idx, first_sample_block, first_sample_trace, last_sample_trace; // sample's index
        size_t s;                                                              // counters
        size_t data_offset, templates_offset, sum_square_template_offset, cc_mat_offset;
        float numerator, denominator, sum_square_data, mean_data, sum_square;
        float data_sample;
        size_t t_idx;

        //------------------------------------------------
        size_t count_template = (n_samples_template / WARPSIZE + 1) * WARPSIZE;
        extern __shared__ float shared[];
        //float *ss_template = &shared[0];
        float *templates_s = &shared[sizeof(float)];
        float *data_s = &shared[count_template + sizeof(float)];

        // 1 block processes one channel to blockDim.x / step different positions in time
        // idx is in units of correlation time
        // first_samples_block is in units of waveform time
        idx = blockIdx.x / n_stations * blockDim.x + chunk_offset;
        first_sample_block = idx * step;
        s = blockIdx.x % n_stations;

        for (size_t c = 0; c < n_components; c++)
        {
            if (weights[s * n_components + c] != 0.)
            {
                // compute offsets for input variables
                cc_mat_offset = (first_sample_block / step + threadIdx.x - chunk_offset) * n_stations * n_components + s * n_components + c;
                templates_offset = s * n_samples_template * n_components + c * n_samples_template;
                sum_square_template_offset = s * n_components + c;
                first_sample_trace = first_sample_block + moveout[s * n_components + c];
                last_sample_trace = first_sample_trace + n_samples_template + threadIdx.x * step;
                data_offset = s * n_samples_data * n_components + c * n_samples_data + first_sample_trace;

                // initialize sums
                sum_square_data = 0.0f;
                mean_data = 0.0f;
                numerator = 0.0f;
                sum_square = 0.0f;

                // load template and data into shared memory
                t_idx = threadIdx.x;

                while (t_idx < n_samples_template)
                {
                    templates_s[t_idx] = templates[templates_offset + t_idx];
                    if ((first_sample_trace + t_idx) < n_samples_data)
                        data_s[t_idx] = data[data_offset + t_idx];
                    t_idx += blockDim.x;
                }
                while (t_idx < (blockDim.x * step + n_samples_template))
                {
                    if ((first_sample_trace + t_idx) < n_samples_data)
                        data_s[t_idx] = data[data_offset + t_idx];
                    t_idx += blockDim.x;
                }

                __syncthreads(); // make sure the waveforms are read before keep going

                // calculate correlation coefficient
                if (last_sample_trace <= n_samples_data)
                {
                    // if not, corresponds to an ill-defined CC with some samples out of the bounds
                    // Calculate the mean if fully normalising
                    if (normalize > 0)
                    {
                        for (size_t i = 0; i < n_samples_template; i++)
                        {
                            mean_data += data_s[i + threadIdx.x * step];
                        }
                        mean_data /= n_samples_template;
                    }

                    for (size_t i = 0; i < n_samples_template; i++)
                    {
                        data_sample = data_s[i + threadIdx.x * step] - mean_data;
                        numerator += data_sample * templates_s[i];
                        sum_square_data += data_sample * data_sample;
                        sum_square += templates[templates_offset + i] * templates[templates_offset + i];
                    }
                    sum_square_template[sum_square_template_offset] = sum_square;
                    denominator = sum_square_data * sum_square_template[sum_square_template_offset];
                    //denominator = sum_square_data * ss_template[0];
                    if (cc_mat_offset < (chunk_size * n_stations * n_components))
                    {
                        // check that this thread is not ouf of the chunk's bounds
                        if (denominator > STABILITY_THRESHOLD)
                        {
                            cc_mat[cc_mat_offset] = numerator * rsqrtf(denominator);
                        }
                    }
                }
            }
            __syncthreads(); // wait for every thread to finish before leaving the kernel
        }
    }

    //-------------------------------------------------------------------------
    __global__ void sum_cc(float *cc_mat, float *cc_sum, float *weights,
                           size_t n_stations, size_t n_components, size_t n_corr,
                           size_t chunk_offset, size_t chunk_size)
    {

        size_t i;

        i = blockIdx.x * blockDim.x + threadIdx.x;
        if (((i + chunk_offset) < n_corr) & (i < chunk_size))
        {
            // first condition: check if we are not outside cc_sum's length
            // second condition: check if we are not outside the chunk's size
            float *cc_mat_offset;

            cc_mat_offset = cc_mat + i * n_stations * n_components;
            for (size_t ch = 0; ch < (n_stations * n_components); ch++)
            {
                cc_sum[i] += cc_mat_offset[ch] * weights[ch];
            }
        }
    }

void checkSharedMemory(size_t sharedMem, size_t count_data,
                       size_t n_samples_template, int maxSharedMem,
                       size_t step, size_t count_template) {

    if (sharedMem > maxSharedMem) {
        size_t new_step = (maxSharedMem / sizeof(float) - 2 * n_samples_template - 2 * WARPSIZE) / BLOCKSIZE;
        int new_length = maxSharedMem / sizeof(float) - count_data - WARPSIZE;
        if (new_length < 0) {
            new_length = 0;
        }
        printf("The maximum shared memory available on this card is %zu bytes "
               "(%zu bytes required). You should consider the different options:\n"
               "  - Change the temporal step to %zu without changing the template length.\n"
               "  - Change the template length to %d without changing the temporal step.\n"
               "  - Try to decrease both of these parameters.\n",
               size_t(maxSharedMem), sharedMem, new_step, new_length);
        exit(EXIT_FAILURE);
    }
}

inline size_t calculate_chunk_size(size_t chunk_offset, size_t chunk_size, size_t n_corr_t) {
    size_t cs; // Variable to hold the calculated chunk size

    // Calculate cs based on provided conditions
    if (chunk_offset + chunk_size > n_corr_t) {
        cs = n_corr_t - chunk_offset;
        // If cs is not positive, the function can't return size_t(-1) since size_t is unsigned
        if (cs == 0)
            return 0; // Use 0 or some special return value to indicate continuation
    } else {
        cs = chunk_size;
    }

    return cs; // Return the calculated chunk size
}

    //-------------------------------------------------------------------------
    void matched_filter(float *templates,
                        int *moveouts, float *data, float *weights, size_t step,
                        size_t n_samples_template, size_t n_samples_data,
                        size_t n_templates, size_t n_stations,
                        size_t n_components, size_t n_corr,
                        float *cc_out, int normalize, int sum_cc_mode) {

        int t_global = -1;
        size_t Mb = MEGABYTES;
        size_t sizeof_cc_out = 0;
        size_t sizeof_cc_out_chunk = 0;

        size_t chunk_size = n_corr / NCHUNKS + 1;

        // Size of variables to create on the device (GPU)
        size_t sizeof_templates = sizeof(float) * n_samples_template * n_stations * n_components * n_templates;
        size_t sizeof_moveouts = sizeof(int) * n_components * n_stations * n_templates;
        size_t sizeof_data = sizeof(float) * n_samples_data * n_stations * n_components;
        size_t sizeof_cc_mat = sizeof(float) * chunk_size * n_stations * n_components; // cc matrix for one template (and one chunk of data)
        if (sum_cc_mode > 0)
        {
            // return summed CC time series (memory efficient)
            sizeof_cc_out = sizeof(float) * chunk_size; // cc sums for one template (and one chunk of data)
        }
        else
        {
            // return one CC time series per channel
            sizeof_cc_out = sizeof_cc_mat;
        }
        size_t sizeof_sum_square_templates = sizeof(float) * n_templates * n_stations * n_components;
        size_t sizeof_weights = sizeof(float) * n_templates * n_stations * n_components;
        size_t sizeof_total = sizeof_templates + sizeof_moveouts + sizeof_data + sizeof_cc_mat + sizeof_cc_out + sizeof_sum_square_templates + sizeof_weights;

            float *templates_d = nullptr;
            float *data_d = nullptr;
            int *moveouts_d = nullptr;
            float *sum_square_templates_d = nullptr;
            float *weights_d = nullptr;
            int id;

            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, 0);

            // Card-dependent settings: prefer L1 cache or shared memory
            hipDeviceSetCacheConfig(hipFuncCachePreferShared);
            // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

            // check if enough memory is available
            size_t freeMem = 0;
            size_t totalMem = 0;
            hipMemGetInfo(&freeMem, &totalMem);
            if (sizeof_total > freeMem)
            {
                printf("%zu Mb are requested on GPU #%i whereas it has only %zu free Mb.\n",
                       sizeof_total / Mb, id, freeMem / Mb);
                printf("Reduce the number of templates or stations processed in one batch.\n");
                exit(0);
            }



            // allocate GPU memory
            hipMalloc((void **)&templates_d, sizeof_templates);
            hipMalloc((void **)&moveouts_d, sizeof_moveouts);
            hipMalloc((void **)&data_d, sizeof_data);
            hipMalloc((void **)&sum_square_templates_d, sizeof_sum_square_templates);
            hipMalloc((void **)&weights_d, sizeof_weights);

            // transfer the inputs from host to the GPU
            hipMemcpy(data_d, data, sizeof_data, hipMemcpyHostToDevice);

            /* TODO: per stream*/
            hipMemcpy(templates_d, templates, sizeof_templates, hipMemcpyHostToDevice);
            hipMemcpy(moveouts_d, moveouts, sizeof_moveouts, hipMemcpyHostToDevice);
            hipMemcpy(weights_d, weights, sizeof_weights, hipMemcpyHostToDevice);


            int maxSharedMem = props.sharedMemPerBlock;

            size_t count_template = (n_samples_template / WARPSIZE + 1) * WARPSIZE;
            size_t count_data = ((n_samples_template + BLOCKSIZE * step) / WARPSIZE + 1) * WARPSIZE;
            size_t sharedMem = (count_template + count_data + 1) * sizeof(float);
            checkSharedMemory(sharedMem, count_data, n_samples_template, maxSharedMem, step, count_template);


            hipStream_t *streams = (hipStream_t*) malloc(n_templates * sizeof(hipStream_t));
            // loop over templates
            for (size_t t = 0; t < n_templates; t++) {
                // Create streams
                hipStreamCreateWithFlags(&streams[t], hipStreamNonBlocking);

                // Allocate result
                float *cc_mat_d = nullptr;
                float *cc_out_d = nullptr;
                hipMallocAsync((void **)&cc_mat_d, sizeof_cc_mat * n_templates, streams[t]);
                hipMallocAsync((void **)&cc_out_d, sizeof_cc_out * n_templates, streams[t]);
                int max_moveout;
                float *cc_out_t = nullptr;

                // compute the number of correlation steps for this template
                int *moveouts_t = moveouts + t * n_stations * n_components;
                max_moveout = 0;

                for (size_t i = 0; i < (n_stations * n_components); i++)
                    max_moveout = (moveouts_t[i] > max_moveout) ? moveouts_t[i] : max_moveout;

                size_t n_corr_t = (n_samples_data - n_samples_template - max_moveout) / step + 1;

                // local pointers on the device
                float *templates_d_t = templates_d + t * n_samples_template * n_stations * n_components;
                float *sum_square_templates_d_t = sum_square_templates_d + t * n_stations * n_components;
                int *moveouts_d_t = moveouts_d + t * n_stations * n_components;
                float *weights_d_t = weights_d + t * n_stations * n_components;

                for (size_t ch = 0; ch < NCHUNKS; ch++) {
                    size_t chunk_offset = ch * chunk_size;

                    // make sure the chunk is not going out of bounds
                    size_t cs = calculate_chunk_size(chunk_offset, chunk_size, n_corr_t);
                    if (cs == 0)
                        continue;

                    // define block and grid sizes for kernels
                    dim3 BS(BLOCKSIZE);
                    dim3 GS(ceilf(cs / (float)BS.x) * n_stations);

                   // process
                    hipMemsetAsync(&cc_mat_d[sizeof_cc_mat * t], 0, sizeof_cc_mat, streams[t]); // initialize cc_mat to 0
                    hipLaunchKernelGGL(network_corr, dim3(GS), dim3(BS), sharedMem, streams[t], templates_d_t,
                                                        sum_square_templates_d_t,
                                                        moveouts_d_t,
                                                        data_d,
                                                        weights_d_t,
                                                        step,
                                                        n_samples_template,
                                                        n_samples_data,
                                                        n_stations,
                                                        n_components,
                                                        chunk_offset,
                                                        cs,
                                                        cc_mat_d,
                                                        normalize);

                    if (sum_cc_mode > 0) {
                        // weighted sum of correlation coefficients
                        hipMemsetAsync(cc_out_d, 0, sizeof_cc_out, streams[t]);

                        // using a small block size seems to improve the speed of sum_cc
                        dim3 BS_sum(32);
                        dim3 GS_sum(ceilf(cs / (float)BS_sum.x));
                        hipLaunchKernelGGL(sum_cc, dim3(GS_sum), dim3(BS_sum), 0, streams[t], cc_mat_d, cc_out_d, weights_d_t,
                                                   n_stations, n_components,
                                                   n_corr_t, chunk_offset, cs);

                        // xfer cc_sum back to host
                        sizeof_cc_out_chunk = sizeof(float) * cs;
                        cc_out_t = cc_out + t * n_corr + chunk_offset;
                        hipMemcpyAsync(cc_out_t, cc_out_d, sizeof_cc_out_chunk, hipMemcpyDeviceToHost, streams[t]);
                    } else {
                        // xfer cc_mat back to host
                        sizeof_cc_out_chunk = sizeof(float) * cs * n_stations * n_components;
                        cc_out_t = cc_out + (t * n_corr + chunk_offset) * n_stations * n_components;
                        hipMemcpyAsync(cc_out_t, cc_mat_d, sizeof_cc_out_chunk, hipMemcpyDeviceToHost, streams[t]);
                    }
                }

                hipFreeAsync(cc_mat_d, streams[t]);
                hipFreeAsync(cc_out_d, streams[t]);
            }

            hipDeviceSynchronize();

            // free device memory
            hipFree(templates_d);
            hipFree(moveouts_d);
            hipFree(data_d);

            hipFree(sum_square_templates_d);
            hipFree(weights_d);

            free(streams);
    }     //  matched_filter
} // extern C
