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
		assert(1 == 0);
		break;
	}
}

#endif

// // -*- c -*-

// #ifndef KERNEL_BS_FIRST_FLOAT4
// #define KERNEL_BS_FIRST_FLOAT4

// #include "template_kernels_float4.cuh"

// void best_kernel_bs_first_float4(float *input, float *values, float *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
// 	while (1) {
// 		threadsPerBlock.y = 1;
// 		if (batch_size == 25088 && a == 6 && b == 64 && c == 64 && d == 1) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = 6;
// 			blockGrid.y = 784;
// 			kernel_bs_first_float4<64, 32, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 4, 128, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 4, 128, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 8, 128, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 4, 128, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 8, 128, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 16, 128, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 8, 128, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 4, 128, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 8, 128, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 4, 128, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 8, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 16, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 8, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 16, 128, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<64, 8, 128, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 8, 128, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<32, 16, 128, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 16, 128, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 8, 128, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 128) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 128 - 1) / 128;
// 			kernel_bs_first_float4<16, 16, 128, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 16, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 4, 64, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 4, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 8, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 4, 64, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 4, 64, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 4, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 8, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 4, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 8, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 4, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 8, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 256;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 32, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 256;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 4, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 8, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 32, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<32, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 8, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 4, 64, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 8, 64, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 64) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 64 - 1) / 64;
// 			kernel_bs_first_float4<16, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 4, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 4, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 8, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 4, 32, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 4, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 4, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 16, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 4, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 4, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 8, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 32, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 128;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 4, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 32 && ((c * d) % (d * 32)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 32, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<32, 8, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 4, 32, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 4, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 32) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 32 - 1) / 32;
// 			kernel_bs_first_float4<16, 8, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 8, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 4, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 4, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 4, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 8, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 4, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 8, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 4, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 4, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 8, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 4, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 8, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 64 && ((b * d) % (d * 64)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 64;
// 			blockGrid.x = (a * b * d + 64 - 1) / 64;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<64, 16, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 4, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 4, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 16, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 32 && ((b * d) % (d * 32)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 32;
// 			blockGrid.x = (a * b * d + 32 - 1) / 32;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<32, 8, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 4;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 16, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 8;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 16 && ((c * d) % (d * 16)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 16, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 8 && ((c * d) % (d * 8)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 8, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		if ((batch_size % 16) == 0 && b > 16 && ((b * d) % (d * 16)) == 0 && c > 4 && ((c * d) % (d * 4)) == 0) {
// 			threadsPerBlock.x = 16;
// 			blockGrid.x = (a * b * d + 16 - 1) / 16;
// 			blockGrid.y = (batch_size + 16 - 1) / 16;
// 			kernel_bs_first_float4<16, 4, 16, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
// 			break;
// 		}
// 		assert(1 == 0);
// 		break;
// 	}
// }

// #endif
