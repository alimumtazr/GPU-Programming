#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "hpc.h"

#define CHECK_CUDA(code) \
  do { if ((code) != cudaSuccess) { \
    fprintf(stderr, "GPU ERROR in %s:%d := %s\n", __FILE__, __LINE__, cudaGetErrorString(code)); exit(code); \
  } } while(0)

#define BLKDIM 16  // smaller block for better occupancy
#define FILTER_RADIUS 1

typedef struct {
    int width;
    int height;
    int maxcol;
    unsigned char *r, *g, *b;
} PPM_image;

void read_ppm(FILE *f, PPM_image* img) {
    char buf[1024];
    char *s;
    int nread;
    assert(f != NULL && img != NULL);
    s = fgets(buf, sizeof(buf), f);
    if (0 != strcmp(s, "P6\n")) { fprintf(stderr,"wrong file type\n"); exit(EXIT_FAILURE); }
    do { s = fgets(buf, sizeof(buf), f); } while (s[0] == '#');
    sscanf(s, "%d %d", &img->width, &img->height);
    s = fgets(buf, sizeof(buf), f);
    sscanf(s, "%d", &img->maxcol);
    if (img->maxcol > 255) { fprintf(stderr,"maxcol>255\n"); exit(EXIT_FAILURE); }
    img->r = (unsigned char*)malloc(img->width*img->height);
    img->g = (unsigned char*)malloc(img->width*img->height);
    img->b = (unsigned char*)malloc(img->width*img->height);
    for (int k=0; k<img->width*img->height; k++) {
        nread = fscanf(f,"%c%c%c", img->r+k, img->g+k, img->b+k);
        if (nread != 3) { fprintf(stderr,"error reading pixel data\n"); exit(EXIT_FAILURE); }
    }
}

void write_ppm(FILE *f, const PPM_image* img, const char *comment) {
    fprintf(f,"P6\n# %s\n%d %d\n%d\n", comment?comment:"", img->width,img->height,img->maxcol);
    for (int k=0;k<img->width*img->height;k++)
        fprintf(f,"%c%c%c", img->r[k], img->g[k], img->b[k]);
}

void free_ppm(PPM_image* img) {
    free(img->r); free(img->g); free(img->b);
    img->r=img->g=img->b=NULL;
    img->width=img->height=img->maxcol=-1;
}

__host__ __device__ void compare_and_swap(unsigned char *a,unsigned char *b) {
    if(*a>*b){ unsigned char t=*a; *a=*b; *b=t; }
}

__host__ __device__ unsigned char median_of_five(unsigned char v[5]) {
    compare_and_swap(v+3,v+4); compare_and_swap(v+2,v+3); compare_and_swap(v+1,v+2); compare_and_swap(v,v+1);
    compare_and_swap(v+3,v+4); compare_and_swap(v+2,v+3); compare_and_swap(v+1,v+2);
    compare_and_swap(v+3,v+4); compare_and_swap(v+2,v+3);
    return v[2];
}

__global__ void denoise_kernel(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in,
                               unsigned char *r_out, unsigned char *g_out, unsigned char *b_out,
                               int width, int height) {
    __shared__ unsigned char tile_r[BLKDIM+2][BLKDIM+2];
    __shared__ unsigned char tile_g[BLKDIM+2][BLKDIM+2];
    __shared__ unsigned char tile_b[BLKDIM+2][BLKDIM+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x*BLKDIM + tx;
    int row = blockIdx.y*BLKDIM + ty;

    int tile_x = tx + 1, tile_y = ty + 1;

    if(row<height && col<width){
        tile_r[tile_y][tile_x] = r_in[row*width+col];
        tile_g[tile_y][tile_x] = g_in[row*width+col];
        tile_b[tile_y][tile_x] = b_in[row*width+col];

        // loading halo
        if(tx==0 && col>0){
            tile_r[tile_y][0] = r_in[row*width+(col-1)];
            tile_g[tile_y][0] = g_in[row*width+(col-1)];
            tile_b[tile_y][0] = b_in[row*width+(col-1)];
        }
        if(tx==BLKDIM-1 && col<width-1){
            tile_r[tile_y][BLKDIM+1] = r_in[row*width+(col+1)];
            tile_g[tile_y][BLKDIM+1] = g_in[row*width+(col+1)];
            tile_b[tile_y][BLKDIM+1] = b_in[row*width+(col+1)];
        }
        if(ty==0 && row>0){
            tile_r[0][tile_x] = r_in[(row-1)*width+col];
            tile_g[0][tile_x] = g_in[(row-1)*width+col];
            tile_b[0][tile_x] = b_in[(row-1)*width+col];
        }
        if(ty==BLKDIM-1 && row<height-1){
            tile_r[BLKDIM+1][tile_x] = r_in[(row+1)*width+col];
            tile_g[BLKDIM+1][tile_x] = g_in[(row+1)*width+col];
            tile_b[BLKDIM+1][tile_x] = b_in[(row+1)*width+col];
        }
    }
    __syncthreads();

    if(row>0 && row<height-1 && col>0 && col<width-1){
        unsigned char v[5];
        // Red
        v[0]=tile_r[tile_y][tile_x]; v[1]=tile_r[tile_y][tile_x-1]; v[2]=tile_r[tile_y][tile_x+1]; 
        v[3]=tile_r[tile_y-1][tile_x]; v[4]=tile_r[tile_y+1][tile_x]; 
        r_out[row*width+col] = median_of_five(v);
        // Green
        v[0]=tile_g[tile_y][tile_x]; v[1]=tile_g[tile_y][tile_x-1]; v[2]=tile_g[tile_y][tile_x+1]; 
        v[3]=tile_g[tile_y-1][tile_x]; v[4]=tile_g[tile_y+1][tile_x]; 
        g_out[row*width+col] = median_of_five(v);
        // Blue
        v[0]=tile_b[tile_y][tile_x]; v[1]=tile_b[tile_y][tile_x-1]; v[2]=tile_b[tile_y][tile_x+1]; 
        v[3]=tile_b[tile_y-1][tile_x]; v[4]=tile_b[tile_y+1][tile_x]; 
        b_out[row*width+col] = median_of_five(v);
    }
}

int cdiv(int a, int b){ return (a+b-1)/b; }

void denoise_gpu(PPM_image *img) {
    unsigned char *d_r, *d_g, *d_b, *d_r_out, *d_g_out, *d_b_out;
    size_t sz = img->width*img->height*sizeof(unsigned char);
    CHECK_CUDA(cudaMalloc(&d_r, sz)); CHECK_CUDA(cudaMalloc(&d_g, sz)); CHECK_CUDA(cudaMalloc(&d_b, sz));
    CHECK_CUDA(cudaMalloc(&d_r_out, sz)); CHECK_CUDA(cudaMalloc(&d_g_out, sz)); CHECK_CUDA(cudaMalloc(&d_b_out, sz));
    CHECK_CUDA(cudaMemcpy(d_r,img->r,sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_g,img->g,sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b,img->b,sz,cudaMemcpyHostToDevice));

    dim3 block(BLKDIM,BLKDIM);
    dim3 grid(cdiv(img->width,BLKDIM),cdiv(img->height,BLKDIM));
    denoise_kernel<<<grid,block>>>(d_r,d_g,d_b,d_r_out,d_g_out,d_b_out,img->width,img->height);
    CHECK_CUDA(cudaMemcpy(img->r,d_r_out,sz,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(img->g,d_g_out,sz,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(img->b,d_b_out,sz,cudaMemcpyDeviceToHost));

    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_r_out); cudaFree(d_g_out); cudaFree(d_b_out);
}

int main(void){
    PPM_image img;
    read_ppm(stdin,&img);
    double tstart=hpc_gettime();
    denoise_gpu(&img);
    double elapsed=hpc_gettime()-tstart;
    fprintf(stderr,"GPU Execution time %.3f\n",elapsed);
    write_ppm(stdout,&img,"produced by cuda-denoise.cu");
    free_ppm(&img);
    return 0;
}
