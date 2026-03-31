#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "hpc.h"

#define BLKDIM 16

typedef struct {
    int width;
    int height;
    int maxgrey;
    unsigned char *bmap;
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    assert(img != NULL);
    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (int i=0; i<height; i++)
        for (int j=0; j<width; j++)
            img->bmap[i*width + j] = col;
}

void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    char *s;
    int nread;
    assert(f != NULL && img != NULL);
    s = fgets(buf, sizeof(buf), f);
    if (0 != strcmp(s, "P5\n")) { fprintf(stderr, "Wrong file type %s\n", buf); exit(EXIT_FAILURE); }
    do { s = fgets(buf, sizeof(buf), f); } while (s[0] == '#');
    sscanf(s, "%d %d", &img->width, &img->height);
    s = fgets(buf, sizeof(buf), f);
    sscanf(s, "%d", &img->maxgrey);
    if (img->maxgrey > 255) { fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey); exit(EXIT_FAILURE); }
    img->bmap = (unsigned char*)malloc(img->width * img->height);
    assert(img->bmap != NULL);
    nread = fread(img->bmap, 1, img->width * img->height, f);
    if (img->width * img->height != nread) { fprintf(stderr, "FATAL: error reading input\n"); exit(EXIT_FAILURE); }
}

void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL && img != NULL);
    fprintf(f, "P5\n# %s\n%d %d\n%d\n", comment ? comment : "", img->width, img->height, img->maxgrey);
    fwrite(img->bmap, 1, img->width * img->height, f);
}

void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL;
    img->width = img->height = img->maxgrey = -1;
}

int IDX(int i, int j, int width) { return i*width + j; }

//CPU version
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width, height = in->height;
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            const int Gx =
                in->bmap[IDX(i-1,j-1,width)] - in->bmap[IDX(i-1,j+1,width)]
                + 2*in->bmap[IDX(i,j-1,width)] - 2*in->bmap[IDX(i,j+1,width)]
                + in->bmap[IDX(i+1,j-1,width)] - in->bmap[IDX(i+1,j+1,width)];
            const int Gy =
                in->bmap[IDX(i-1,j-1,width)] + 2*in->bmap[IDX(i-1,j,width)] + in->bmap[IDX(i-1,j+1,width)]
                - in->bmap[IDX(i+1,j-1,width)] - 2*in->bmap[IDX(i+1,j,width)] - in->bmap[IDX(i+1,j+1,width)];
            const int magnitude = Gx*Gx + Gy*Gy;
            edges->bmap[IDX(i,j,width)] = (magnitude > threshold*threshold) ? WHITE : BLACK;
        }
    }
}

// GPU kernel
__global__ void edge_detect_kernel( const unsigned char *in, unsigned char *out,
                                    int width, int height, int threshold )
{
    // Shared tile ; interior + 1-pixel halo on each side
    __shared__ unsigned char tile[BLKDIM+2][BLKDIM+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x*BLKDIM + tx;
    int row = blockIdx.y*BLKDIM + ty;
    int tile_x = tx + 1, tile_y = ty + 1;

    // Load interior
    if (row < height && col < width)
        tile[tile_y][tile_x] = in[row*width + col];

    // Load halo edges
    if (tx == 0)
        tile[tile_y][0] = (col > 0 && row < height) ? in[row*width + (col-1)] : 0;
    if (tx == BLKDIM-1 || col == width-1)
        tile[tile_y][BLKDIM+1] = (col < width-1 && row < height) ? in[row*width + (col+1)] : 0;
    if (ty == 0)
        tile[0][tile_x] = (row > 0 && col < width) ? in[(row-1)*width + col] : 0;
    if (ty == BLKDIM-1 || row == height-1)
        tile[BLKDIM+1][tile_x] = (row < height-1 && col < width) ? in[(row+1)*width + col] : 0;

    // Load halo corners (needed for Sobel diagonals)
    if (tx == 0 && ty == 0)
        tile[0][0] = (row > 0 && col > 0) ? in[(row-1)*width + (col-1)] : 0;
    if ((tx == BLKDIM-1 || col == width-1) && ty == 0)
        tile[0][BLKDIM+1] = (row > 0 && col < width-1) ? in[(row-1)*width + (col+1)] : 0;
    if (tx == 0 && (ty == BLKDIM-1 || row == height-1))
        tile[BLKDIM+1][0] = (row < height-1 && col > 0) ? in[(row+1)*width + (col-1)] : 0;
    if ((tx == BLKDIM-1 || col == width-1) && (ty == BLKDIM-1 || row == height-1))
        tile[BLKDIM+1][BLKDIM+1] = (row < height-1 && col < width-1) ? in[(row+1)*width + (col+1)] : 0;

    __syncthreads();

    // Compute Sobel from shared memory for interior pixels
    if (row > 0 && row < height-1 && col > 0 && col < width-1) {
        const int Gx =
            tile[tile_y-1][tile_x-1] - tile[tile_y-1][tile_x+1]
            + 2*tile[tile_y][tile_x-1] - 2*tile[tile_y][tile_x+1]
            + tile[tile_y+1][tile_x-1] - tile[tile_y+1][tile_x+1];
        const int Gy =
            tile[tile_y-1][tile_x-1] + 2*tile[tile_y-1][tile_x] + tile[tile_y-1][tile_x+1]
            - tile[tile_y+1][tile_x-1] - 2*tile[tile_y+1][tile_x] - tile[tile_y+1][tile_x+1];
        const int magnitude = Gx*Gx + Gy*Gy;
        out[row*width + col] = (magnitude > threshold*threshold) ? WHITE : BLACK;
    } else if (row < height && col < width) {
        out[row*width + col] = BLACK;
    }
}

int cdiv(int a, int b) { return (a+b-1)/b; }

void edge_detect_gpu( const PGM_image* in, PGM_image* edges, int threshold )
{
    unsigned char *d_in, *d_out;
    size_t sz = in->width * in->height * sizeof(unsigned char);

    cudaMalloc(&d_in, sz);
    cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, in->bmap, sz, cudaMemcpyHostToDevice);

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid(cdiv(in->width, BLKDIM), cdiv(in->height, BLKDIM));
    edge_detect_kernel<<<grid, block>>>(d_in, d_out, in->width, in->height, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(edges->bmap, d_out, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out_cpu, out_gpu;
    int threshold = 70;

    if (argc > 2) { fprintf(stderr, "Usage: %s [threshold] < in.pgm > out.pgm\n", argv[0]); return EXIT_FAILURE; }
    if (argc > 1) threshold = atoi(argv[1]);

    read_pgm(stdin, &bmap);
    init_pgm(&out_cpu, bmap.width, bmap.height, BLACK);
    init_pgm(&out_gpu, bmap.width, bmap.height, BLACK);

    //CPU
    double tstart = hpc_gettime();
    edge_detect(&bmap, &out_cpu, threshold);
    fprintf(stderr, "CPU Execution time %.3f\n", hpc_gettime() - tstart);

    // GPU
    tstart = hpc_gettime();
    edge_detect_gpu(&bmap, &out_gpu, threshold);
    fprintf(stderr, "GPU Execution time %.3f\n", hpc_gettime() - tstart);

    write_pgm(stdout, &out_gpu, "produced by cuda-edge-detect.cu");

    free_pgm(&bmap);
    free_pgm(&out_cpu);
    free_pgm(&out_gpu);
    return EXIT_SUCCESS;
}

