#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0)
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255)
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uint8_t * pixels, int width, int height, 
		char * fileName, int numChannels=1)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
}

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < height && c < width)
	{
		int i = r * width + c;
		uint8_t red = inPixels[i].x;
		uint8_t green = inPixels[i].y;
		uint8_t blue = inPixels[i].z;
		outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
	}
}

void convertRgb2Gray(uchar3 * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
				int i = r * width + c;
				uint8_t red = inPixels[i].x;
				uint8_t green = inPixels[i].y;
				uint8_t blue = inPixels[i].z;
				outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
            }
        }
	}
	else 
	{
		uchar3 * d_inPixels;
		uint8_t * d_outPixels;
		size_t nBytes = 3 * width * height * sizeof(uint8_t);
		CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_outPixels, nBytes));
		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

		cudaError_t errSync  = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess) 
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
		CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time convertRgb2Gray (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

__global__ void convertGray2SobelKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels, int8_t * x_Sobel, int8_t * y_Sobel, uint8_t filterWidth)
{
	int outPixelsR = blockIdx.y * blockDim.y + threadIdx.y;
	int outPixelsC = blockIdx.x * blockDim.x + threadIdx.x;

	if(outPixelsR < height && outPixelsC < width)
	{
		int outPixel_x = 0;       
		int outPixel_y = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					int8_t filterVal_x = x_Sobel[filterR * filterWidth + filterC];
					int8_t filterVal_y = y_Sobel[filterR * filterWidth + filterC];
					int inPixelsR = outPixelsR - filterWidth/2 + filterR;
					int inPixelsC = outPixelsC - filterWidth/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
					outPixel_x += filterVal_x * inPixel;
					outPixel_y += filterVal_y * inPixel;
				}
			}
			outPixels[outPixelsR * width + outPixelsC] = abs(outPixel_x) + abs(outPixel_y);
	}

}

void convertGray2Sobel(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels,
		int8_t * x_Sobel, int8_t * y_Sobel, uint8_t filterWidth, bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				int outPixel_x = 0;       
				int outPixel_y = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						int8_t filterVal_x = x_Sobel[filterR * filterWidth + filterC];
						int8_t filterVal_y = y_Sobel[filterR * filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
						outPixel_x += filterVal_x * inPixel;
						outPixel_y += filterVal_y * inPixel;
					}
				}
				outPixels[outPixelsR * width + outPixelsC] = abs(outPixel_x) + abs(outPixel_y);
			}
		}
	}
	else
	{
		uint8_t * d_inPixels;
		uint8_t * d_outPixels;
		int8_t * d_x_Sobel;
		int8_t * d_y_Sobel;
		size_t nBytes = height * width * sizeof(uint8_t);
		size_t nBytesFilter = filterWidth  * filterWidth  * sizeof(int8_t);
		CHECK(cudaMalloc(&d_inPixels, nBytes));
		CHECK(cudaMalloc(&d_outPixels, nBytes));
		CHECK(cudaMalloc(&d_x_Sobel, nBytesFilter));
		CHECK(cudaMalloc(&d_y_Sobel, nBytesFilter));
		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_x_Sobel, x_Sobel, nBytesFilter, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_y_Sobel, y_Sobel, nBytesFilter, cudaMemcpyHostToDevice));
		dim3 gridSize((width - 1) / blockSize.x + 1, 
                (height - 1) / blockSize.y + 1);
		convertGray2SobelKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels, d_x_Sobel, d_y_Sobel, filterWidth);
		cudaError_t errSync  = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess) 
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
		CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));
		CHECK(cudaFree(d_inPixels));
        CHECK(cudaFree(d_outPixels));
        CHECK(cudaFree(d_x_Sobel));
		CHECK(cudaFree(d_y_Sobel));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time convertGray2Sobel (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

void computeEnergy(uint8_t * inPixels, int width, int height, int * outPixels)
{
	for (int c = 0; c < width; c++)
	{
		outPixels[(height - 1) * width + c] = inPixels[(height - 1) * width + c];
	}
}
__global__ void computeEnergyKernel(uint8_t * inPixels, int width, int height, int * outPixels)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width)
	{
		outPixels[(height - 1) * width + c] = inPixels[(height - 1) * width + c];
	}
}

void computeSumEnergy(uint8_t * inPixels, int width, int height,
		int * outPixels, int8_t * trace)
{
	for (int outPixelsR = height - 2; outPixelsR >= 0; outPixelsR--)
	{
		int outPixel_left, outPixel_mid, outPixel_right, temp, temp_sum;
		uint8_t inPixel_cur;
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			if (outPixelsC == 0)
			{
				inPixel_cur = inPixels[outPixelsR * width];
				outPixel_mid = outPixels[(outPixelsR + 1) * width];
				outPixel_right = outPixels[(outPixelsR + 1) * width + 1];
				if (outPixel_mid < outPixel_right)
				{
					temp = outPixel_mid;
					trace[outPixelsR * width] = 0;
				}
				else
				{
					temp = outPixel_right;
					trace[outPixelsR * width] = 1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[outPixelsR * width] = temp_sum;
			}
			else if (outPixelsC == width - 1)
			{
				inPixel_cur = inPixels[(outPixelsR + 1) * width - 1];
				outPixel_mid = outPixels[(outPixelsR + 2) * width - 1];
				outPixel_left = outPixels[(outPixelsR + 2) * width - 2];
				if (outPixel_mid < outPixel_left)
				{
					temp = outPixel_mid;
					trace[(outPixelsR + 1) * width - 1] = 0;
				}
				else
				{
					temp = outPixel_left;
					trace[(outPixelsR + 1) * width - 1] = -1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[(outPixelsR + 1) * width - 1] = temp_sum;
			}
			else
			{
				inPixel_cur = inPixels[outPixelsR * width + outPixelsC];
				outPixel_left = outPixels[(outPixelsR + 1) * width + outPixelsC - 1];
				outPixel_mid = outPixels[(outPixelsR + 1) * width + outPixelsC];
				outPixel_right = outPixels[(outPixelsR + 1) * width + outPixelsC + 1];
				if (outPixel_mid < outPixel_right)
				{
					temp = outPixel_mid;
					trace[outPixelsR * width + outPixelsC] = 0;
				}
				else if (outPixel_left < outPixel_right)
				{
					temp = outPixel_left;
					trace[outPixelsR * width + outPixelsC] = -1;
				}
				else
				{
					temp = outPixel_right;
					trace[outPixelsR * width + outPixelsC] = 1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[outPixelsR * width + outPixelsC] = temp_sum;
			}
		}
	}
}

__global__ void computeSumEnergyKernel(uint8_t * inPixels, int width, int height, int * outPixels, int8_t * trace)
{
	int outPixelsC = blockIdx.x * blockDim.x + threadIdx.x;
	for (int outPixelsR = height - 2; outPixelsR >= 0; outPixelsR--)
	{
		int outPixel_left, outPixel_mid, outPixel_right, temp, temp_sum;
		uint8_t inPixel_cur;
		if (outPixelsC < width)
		{
			if (outPixelsC == 0)
			{
				inPixel_cur = inPixels[outPixelsR * width];
				outPixel_mid = outPixels[(outPixelsR + 1) * width];
				outPixel_right = outPixels[(outPixelsR + 1) * width + 1];
				if (outPixel_mid < outPixel_right)
				{
					temp = outPixel_mid;
					trace[outPixelsR * width] = 0;
				}
				else
				{
					temp = outPixel_right;
					trace[outPixelsR * width] = 1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[outPixelsR * width] = temp_sum;
			}
			else if (outPixelsC == width - 1)
			{
				inPixel_cur = inPixels[(outPixelsR + 1) * width - 1];
				outPixel_mid = outPixels[(outPixelsR + 2) * width - 1];
				outPixel_left = outPixels[(outPixelsR + 2) * width - 2];
				if (outPixel_mid < outPixel_left)
				{
					temp = outPixel_mid;
					trace[(outPixelsR + 1) * width - 1] = 0;
				}
				else
				{
					temp = outPixel_left;
					trace[(outPixelsR + 1) * width - 1] = -1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[(outPixelsR + 1) * width - 1] = temp_sum;
			}
			else
			{
				inPixel_cur = inPixels[outPixelsR * width + outPixelsC];
				outPixel_left = outPixels[(outPixelsR + 1) * width + outPixelsC - 1];
				outPixel_mid = outPixels[(outPixelsR + 1) * width + outPixelsC];
				outPixel_right = outPixels[(outPixelsR + 1) * width + outPixelsC + 1];
				if (outPixel_mid < outPixel_right)
				{
					temp = outPixel_mid;
					trace[outPixelsR * width + outPixelsC] = 0;
				}
				else if (outPixel_left < outPixel_right)
				{
					temp = outPixel_left;
					trace[outPixelsR * width + outPixelsC] = -1;
				}
				else
				{
					temp = outPixel_right;
					trace[outPixelsR * width + outPixelsC] = 1;
				}
				temp_sum = inPixel_cur + temp;
				outPixels[outPixelsR * width + outPixelsC] = temp_sum;
			}
			__syncthreads();
		}
	}
}

void findSeam(int * inPixels, int8_t * trace, int width, int height,
		int * seam)
{
	for (int i = 0; i < height; i++)
	{
		if (i == 0)
		{
			int inPixel_idx = 0;
			for (int c = 1; c < width; c++)
			{
				if (inPixels[c] < inPixels[inPixel_idx])
				{
					inPixel_idx = c;
				}
			}
			seam[i] = inPixel_idx;
		}
		else
		{
			seam[i] = width + seam[i - 1] + trace[seam[i - 1]];
		}
	}
}


void removeSeam(uchar3 * inPixels, uint8_t * inPixels_Sobel, int * seam, int width, int height)
{
	int length = width * height;
	for (int i = height - 1; i >= 0; i--)
	{
		int j = 0;
		memcpy(&inPixels_Sobel[seam[i]], &inPixels_Sobel[seam[i] + 1], length - seam[i] - 1 - j);
		memcpy(&inPixels[seam[i]], &inPixels[seam[i] + 1], (length - seam[i] - 1 - j) * sizeof(uchar3));
		j++;
	}
}

__global__ void removeSeamKernel(uchar3 * inPixels, uint8_t * inPixels_Sobel, int * seam,
									uchar3 * inPixels_temp, uint8_t * inPixels_Sobel_temp,
									int width, int height)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < seam[0])
	{
		inPixels_Sobel_temp[c] = inPixels_Sobel[c];
		inPixels_temp[c] = inPixels[c];
	}
	__syncthreads();

	int j = 1;
	uint8_t t_inS;
	uchar3 t_in;
	for (int i = 1; i < height; i++)
	{
		t_inS = inPixels_Sobel[seam[i - 1] + 1 + c];
	 	t_in = inPixels[seam[i - 1] + 1 + c];
	 	__syncthreads();
		if (seam[i - 1] + 1 + c < seam[i])
		{
			inPixels_Sobel_temp[seam[i - 1] + 1 + c - j] = t_inS;
			inPixels_temp[seam[i - 1] + 1 + c - j] = t_in;
		}
		__syncthreads();
		j++;
	}

	int length = width * height;
	t_inS = inPixels_Sobel[seam[height - 1] + 1 + c];
	t_in = inPixels[seam[height - 1] + 1 + c];
	if (seam[height - 1] + 1 + c < length)
	{
		inPixels_Sobel_temp[seam[height - 1] + 1 + c - j] = t_inS;
		inPixels_temp[seam[height - 1] + 1 + c - j] = t_in;
	}
}

void find2removeSeam(int new_width, int &i, uint8_t * correctOutSobelPixels, int * correctSumEnergy, int * correctSeam, int8_t * trace, uchar3 * inPixels, int width, int height, bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		for (i; i > new_width; i--)
		{
			computeEnergy(correctOutSobelPixels, i, height, correctSumEnergy);
			computeSumEnergy(correctOutSobelPixels, i, height, correctSumEnergy, trace);
			findSeam(correctSumEnergy, trace, i, height, correctSeam);
			removeSeam(inPixels, correctOutSobelPixels, correctSeam, i, height);
		}
	}
	else
	{
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		uchar3 * d_inPixels;
		uint8_t * d_correctOutSobelPixels;
		int * d_correctSumEnergy;
		int * d_correctSeam;
		int8_t * d_trace;
		CHECK(cudaMalloc(&d_inPixels, height * width * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_correctOutSobelPixels, height * width * sizeof(uint8_t)));
		CHECK(cudaMalloc(&d_correctSumEnergy, height * width * sizeof(int)));
		CHECK(cudaMalloc(&d_correctSeam, height * sizeof(int)));
		CHECK(cudaMalloc(&d_trace, height * width * sizeof(int8_t)));

		uchar3 * d_inPixels_temp;
		uint8_t * d_correctOutSobelPixels_temp;
		CHECK(cudaMalloc(&d_inPixels_temp, height * width * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_correctOutSobelPixels_temp, height * width * sizeof(uint8_t)));

		CHECK(cudaMemcpy(d_correctOutSobelPixels, correctOutSobelPixels, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_inPixels, inPixels, height * width * sizeof(uchar3), cudaMemcpyHostToDevice));
		
		dim3 newBlockSize(blockSize.x * blockSize.y);
		dim3 newGridSizeX((width - 1) / newBlockSize.x + 1);

		for (i; i > new_width; i--)
		{
			computeEnergyKernel<<<newGridSizeX, newBlockSize>>>(d_correctOutSobelPixels, i, height, d_correctSumEnergy);
			computeSumEnergyKernel<<<newGridSizeX, newBlockSize>>>(d_correctOutSobelPixels, i, height, d_correctSumEnergy, d_trace);
			cudaError_t errSync  = cudaGetLastError();
			cudaError_t errAsync = cudaDeviceSynchronize();
			if (errSync != cudaSuccess) 
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
			if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
			
			CHECK(cudaMemcpy(trace, d_trace, height * i * sizeof(int8_t), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(correctSumEnergy, d_correctSumEnergy, height * i * sizeof(int), cudaMemcpyDeviceToHost));
			findSeam(correctSumEnergy, trace, i, height, correctSeam);
			CHECK(cudaMemcpy(d_correctSeam, correctSeam, height * sizeof(int), cudaMemcpyHostToDevice));
			
			removeSeamKernel<<<newGridSizeX, newBlockSize>>>(d_inPixels, d_correctOutSobelPixels, d_correctSeam, d_inPixels_temp, d_correctOutSobelPixels_temp, i, height);
			errSync  = cudaGetLastError();
			errAsync = cudaDeviceSynchronize();
			if (errSync != cudaSuccess) 
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
			if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
			CHECK(cudaMemcpy(d_correctOutSobelPixels, d_correctOutSobelPixels_temp, height * i * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(d_inPixels, d_inPixels_temp, height * i * sizeof(uchar3), cudaMemcpyDeviceToDevice));
			
		}
		CHECK(cudaMemcpy(inPixels, d_inPixels, height * i * sizeof(uchar3), cudaMemcpyDeviceToHost));
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_correctOutSobelPixels));
		CHECK(cudaFree(d_correctSumEnergy));
		CHECK(cudaFree(d_correctSeam));
		CHECK(cudaFree(d_trace));
		CHECK(cudaFree(d_inPixels_temp));
		CHECK(cudaFree(d_correctOutSobelPixels_temp));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time find2removeSeam (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}


int main(int argc, char ** argv)
{	
	if (argc != 3 && argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	uchar3 * inPixelsDevice;
	readPnm(argv[1], width, height, inPixels);
	readPnm(argv[1], width, height, inPixelsDevice);
	printf("Image size (width x height): %i x %i\n\n", width, height);
	char * outFileNameBase = strtok(argv[2], ".");

	// Set up Sobel filters
	uint8_t filterWidth = 3;
	int8_t * x_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	int8_t * y_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	x_Sobel[0] = x_Sobel[6] = y_Sobel[0] = y_Sobel[2] = 1;
	x_Sobel[1] = x_Sobel[4] = x_Sobel[7] = y_Sobel[3] = y_Sobel[4] = y_Sobel[5] = 0;
	x_Sobel[2] = x_Sobel[8] = y_Sobel[6] = y_Sobel[8] = -1;
	x_Sobel[3] = y_Sobel[1] = 2;
	x_Sobel[5] = y_Sobel[7] = -2;

	// Convert RGB to grayscale
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "_ck2_gray_host.pnm"));

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}  
	convertRgb2Gray(inPixelsDevice, width, height, outPixels, true, blockSize);
	writePnm(outPixels, width, height, concatStr(outFileNameBase, "_ck2_gray_device.pnm"));

	// Convert grayscale to sobel-grayscale (energy)
	uint8_t * correctOutSobelPixels= (uint8_t *)malloc(width * height);
	convertGray2Sobel(correctOutPixels, width, height, correctOutSobelPixels, x_Sobel, y_Sobel, filterWidth);
	writePnm(correctOutSobelPixels, width, height, concatStr(outFileNameBase, "_ck2_sobel_host.pnm"));

	// Convert grayscale to sobel-grayscale (energy) using device 
	uint8_t * correctOutSobelPixelsDevice= (uint8_t *)malloc(width * height);
	convertGray2Sobel(outPixels, width, height, correctOutSobelPixelsDevice, x_Sobel, y_Sobel, filterWidth, true, blockSize);
	writePnm(correctOutSobelPixelsDevice, width, height, concatStr(outFileNameBase, "_ck2_sobel_device.pnm"));

	int new_width = 2 * width / 3; //Default
	if (argc == 4 || argc == 6)
	{
		new_width = atoi(argv[3]);
	}  
	int i = width;
	int k = width;

	// Find and remove seam using host
	int * correctSumEnergy = (int *)malloc(width * height * sizeof(int));
	int8_t * trace = (int8_t *)malloc(width * height * sizeof(int8_t));
	int * correctSeam = (int *)malloc(height * sizeof(int));
	find2removeSeam(new_width, i, correctOutSobelPixels, correctSumEnergy, correctSeam, trace, inPixels, width, height);
	writePnm(inPixels, i, height, concatStr(outFileNameBase, "_ck2_seam_host.pnm"));

	// Find and remove seam using device
	int * correctSumEnergyDevice = (int *)malloc(width * height * sizeof(int));
	int8_t * traceDevice = (int8_t *)malloc(width * height* sizeof(int8_t));
	int * correctSeamDevice = (int *)malloc(height * sizeof(int));
	find2removeSeam(new_width,k, correctOutSobelPixelsDevice, correctSumEnergyDevice, correctSeamDevice, traceDevice, inPixelsDevice, width, height, true, blockSize);
	writePnm(inPixelsDevice, k, height, concatStr(outFileNameBase, "_ck2_seam_device.pnm"));

	// Free memories
	free(inPixels);
	free(outPixels);
	free(inPixels);
	free(inPixelsDevice);
	free(x_Sobel);
	free(y_Sobel);
	free(correctOutPixels);
	free(correctOutSobelPixels);
	free(correctSumEnergy);
	free(trace);
	free(correctSeam);
	free(correctSumEnergyDevice);
	free(traceDevice);
	free(correctSeamDevice);
}
