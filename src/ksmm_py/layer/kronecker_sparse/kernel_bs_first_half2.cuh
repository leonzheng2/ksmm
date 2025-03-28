// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF2
#define KERNEL_BS_FIRST_HALF2

#include "template_kernels_half2.cuh"

void best_kernel_bs_first_half2(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
        if (batch_size == 12544 && a == 1 && b == 512 && c == 3584 && d == 8) {
            threadsPerBlock.x = 64;
            blockGrid.x = 64;
            blockGrid.y = 196;
            kernel_bs_first_half2<64,16,64,8,8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
        }
        if (batch_size == 12544 && a == 896 && b == 32 && c == 16 && d == 1) {
            threadsPerBlock.x = 32;
            blockGrid.x = 896;
            blockGrid.y = 196;
            kernel_bs_first_half2<32,8,64,8,8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
        }
        if (batch_size == 12544 && a == 1 && b == 16 && c == 32 && d == 896) {
            threadsPerBlock.x = 32;
            blockGrid.x = 896;
            blockGrid.y = 196;
            kernel_bs_first_half2<16,16,64,8,4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
        }
        if (batch_size == 12544 && a == 8 && b == 3584 && c == 512 && d == 1) {
            threadsPerBlock.x = 64;
            blockGrid.x = 448;
            blockGrid.y = 196;
            kernel_bs_first_half2<64,16,64,8,8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
            break;
        }
		assert(1 == 0);
		break;
	}
}

#endif
