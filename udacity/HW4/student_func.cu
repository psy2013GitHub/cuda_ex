
//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <math.h>
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__
void gen_hist(unsigned int * const d_inputVals, 
              unsigned int * d_hist, 
              unsigned int pass, 
              unsigned int numElems) {
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= numElems)
       return;
   unsigned int b = d_inputVals[idx] & (1u << pass);
   if (b) 
      atomicAdd(&(d_hist[1]),1);
   else
      atomicAdd(&(d_hist[0]),1); 
}    

__global__
void scan_ele(unsigned int* d_inputVals, unsigned int* d_scaned, unsigned int base, unsigned int pass, unsigned int numElems, unsigned int threadSize) {
    unsigned int idx = base*threadSize + threadIdx.x;
    if (idx >= numElems)
        return;
    unsigned int b = (d_inputVals[idx] & (1u << pass))?1:0;
    int spot, val;
    
    d_scaned[idx] = b;
    __syncthreads();
    for (unsigned int s=threadSize>>1; s>0; s=s>>1) {
        spot = idx - s;
        if (spot >= 0 && spot>=base*threadSize)
            val = d_scaned[spot];
        __syncthreads();
        if (spot >= 0 && spot>=base*threadSize)
            d_scaned[idx] += val;
        __syncthreads();
    }
    if (base>0)
          d_scaned[idx] +=  d_scaned[base*threadSize-1];
}    

/* scan_ele0, 错误方法 */
__global__
void scan_ele0(unsigned int* d_inputVals, unsigned int* d_scaned, unsigned int pass, unsigned int numElems) {
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= numElems)
        return;
    unsigned int b = (d_inputVals[idx] & (1u << pass))?1:0;
    int spot, val;
    
    d_scaned[idx] = b;
    __syncthreads();
    for (unsigned int s=blockDim.x>>1; s>0; s=s>>1) {
        spot = idx - s;
        if (spot >= 0)
            val = d_scaned[spot];
        __syncthreads();
        if (spot >= 0)
            d_scaned[idx] += val;
        __syncthreads();
    }
    /* if (base>0)
          d_scaned[idx] +=  d_scaned[base*threadSize-1]; */
}    


__global__
void move_ele(unsigned int* const d_inputVals, 
              unsigned int* const d_inputPos,
              unsigned int* const d_outputVals,
              unsigned int* const d_outputPos,
              unsigned int* const d_scaned,
              unsigned int* const d_hist,
              unsigned int pass,
              unsigned int numElems) {
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= numElems)
       return;
    
   unsigned int b = d_inputVals[idx] & (1u << pass); 
   
   unsigned int base=0;
   unsigned int offset=0;
   unsigned int p=0; 
   if (b) {
      base = d_hist[0]; 
      offset = idx?d_scaned[idx-1]:0; // !!!
      // offset = d_scaned[idx]; // !!! 
   } else {
      base = 0;
      offset = idx - d_scaned[idx];
   }
   p = base + offset; 
   d_outputVals[p] = d_inputVals[idx];
   d_outputPos[p]  = d_inputPos[idx];  
}    

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    
 dim3 gridSize(ceil((float)(numElems)/1024.0f)+1);
 dim3 blockSize(1024);
 
 unsigned int *d_hist;
 cudaMalloc((void **)(&d_hist), sizeof(unsigned int) * 2);
 
 size_t bytes = sizeof(unsigned int) * numElems;   
    
 /* unsigned int *d_inputVals_tmp;
 cudaMalloc((void **)(&d_inputVals_tmp), bytes);
 cudaMemcpy(d_inputVals_tmp, d_inputVals, bytes, cudaMemcpyDeviceToDevice);   
 unsigned int *d_inputPos_tmp;
 cudaMalloc((void **)(&d_inputPos_tmp), bytes);
 cudaMemcpy(d_inputPos_tmp,  d_inputPos, bytes, cudaMemcpyDeviceToDevice);     
 */
    
 /* unsigned int *d_scan;
 checkCudaErrors(cudaMalloc((void **)(&d_scan), bytes)); */
 unsigned int *d_scaned;
 checkCudaErrors(cudaMalloc((void **)(&d_scaned), bytes));
 // unsigned int h_scaned[numElems]; 
    
 for (int pass=0; pass<32; pass++) {
     
     checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) << 1));
     checkCudaErrors(cudaMemset(d_scaned, 0, bytes));
     checkCudaErrors(cudaMemset(d_outputVals, 0, bytes));
     checkCudaErrors(cudaMemset(d_outputPos, 0, bytes));
     
     gen_hist<<<gridSize, blockSize>>>(d_inputVals, d_hist, pass, numElems);
     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
     
     // 为了对整个数组进行排序，只能一个block一个block处理，因为scan调用了__syncthreads()，只能保证在block内有效
     for (unsigned base=0; base < gridSize.x; ++base) { // loop each block
         scan_ele<<<dim3(1), blockSize>>>(d_inputVals, d_scaned, base, pass, numElems, blockSize.x);
         cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
     }
     /* 以下方法错误，无法对整个进行数组排序
     scan_ele0<<<gridSize, blockSize>>>(d_inputVals, d_scaned, pass, numElems);
     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
     */
     move_ele<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scaned, d_hist, pass, numElems);
     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
     
     checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, bytes, cudaMemcpyDeviceToDevice));    
     checkCudaErrors(cudaMemcpy(d_inputPos,  d_outputPos,  bytes, cudaMemcpyDeviceToDevice));
     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 }    
 /*
 checkCudaErrors(cudaMemcpy(d_inputVals, d_inputVals_tmp, bytes, cudaMemcpyDeviceToDevice));
 checkCudaErrors(cudaMemcpy(d_inputPos,  d_inputPos_tmp,  bytes, cudaMemcpyDeviceToDevice));   
 */   
 checkCudaErrors(cudaFree((void *)(d_hist)));
 /*
 checkCudaErrors(cudaFree((void *)(d_inputVals_tmp)));
 checkCudaErrors(cudaFree((void *)(d_inputPos_tmp)));   
 */   
}

