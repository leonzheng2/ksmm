// -*- c -*-

#ifndef KERNEL_BS_FIRST_FLOAT4
#define KERNEL_BS_FIRST_FLOAT4

#include "template_kernels_float4.cuh"

void best_kernel_bs_first_float4(float *input, float *values, float *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
        if (batch_size == 25088 && a == 1 && b == 64 && c == 768 && d == 24) {
            threadsPerBlock.x = 32;
            blockGrid.x = 24;
            blockGrid.y = 196;
            kernel_bs_first_float4<64,8,128,16,16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
          }
          if (batch_size == 25088 && a == 96 && b == 192 && c == 64 && d == 1) {
            threadsPerBlock.x = 32;
            blockGrid.x = 288;
            blockGrid.y = 196;
            kernel_bs_first_float4<64,8,128,16,16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
          }
        if (batch_size == 25088 && a == 1 && b == 64 && c == 192 && d == 96) {
            threadsPerBlock.x = 32;
            blockGrid.x = 96;
            blockGrid.y = 196;
            kernel_bs_first_float4<64,8,128,16,16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
          }
          if (batch_size == 25088 && a == 24 && b == 768 && c == 64 && d == 1) {
            threadsPerBlock.x = 32;
            blockGrid.x = 288;
            blockGrid.y = 196;
            kernel_bs_first_float4<64,8,128,16,16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
          }
//         if (batch_size == 784 && a == 1 && b == 512 && c == 3584 && d == 8) {
//                         threadsPerBlock.x = 64;
//                         blockGrid.x = 32;
//                         blockGrid.y = 49;
//                         kernel_bs_first_float4<128,16,16,4,8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
//                         break;
//                       }
//
//         if (batch_size == 784 && a == 896 && b == 32 && c == 16 && d == 1) {
//                         threadsPerBlock.x = 32;
//                         blockGrid.x = 896;
//                         blockGrid.y = 49;
//                         kernel_bs_first_float4<32,16,16,4,4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
//                         break;
//                       }
//
//         if (batch_size == 784 && a == 1 && b == 16 && c == 32 && d == 896) {
//                         threadsPerBlock.x = 16;
//                         blockGrid.x = 896;
//                         blockGrid.y = 49;
//                         kernel_bs_first_float4<16,16,16,4,4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
//                         break;
//                       }
//
//
//         if (batch_size == 784 && a == 8 && b == 3584 && c == 512 && d == 1) {
//                         threadsPerBlock.x = 32;
//                         blockGrid.x = 224;
//                         blockGrid.y = 49;
//                         kernel_bs_first_float4<128,16,16,8,8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
//                         break;
//                       }


		assert(1 == 0);
		break;
	}
}

#endif
