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
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
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
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
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
	else // use device
	{
		// TODO: Allocate device memories
		uchar3 * d_inPixels;
		uint8_t * d_outPixels;
		size_t nBytes = 3 * width * height * sizeof(uint8_t); // 3 phần tử liên tiếp lần lượt là r, g, b nên x3
		CHECK(cudaMalloc(&d_inPixels, 3 * width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_outPixels, nBytes));

		// TODO: Copy data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel (remember to check kernel error)
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

		cudaError_t errSync  = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess) 
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
		// TODO: Copy result from device memories
		CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));

		// TODO: Free device memories
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
		// TODO
		uint8_t * d_inPixels;
		uint8_t * d_outPixels;
		int8_t * d_x_Sobel;
		int8_t * d_y_Sobel;
		size_t nBytes = height * width * sizeof(uint8_t);
		size_t nBytesFilter = height * width * sizeof(int8_t);
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
char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{	
	// if (argc != 3 && argc != 5)
	// {
	// 	printf("The number of arguments is invalid\n");
	// 	return EXIT_FAILURE;
	// }

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
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
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "_gray_host.pnm"));

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize);
	writePnm(outPixels, width, height, concatStr(outFileNameBase, "_gray_device.pnm"));

	// Compute mean absolute error between host result and device result
	// float err = computeError(outPixels, correctOutPixels, width * height);
	// printf("Error between device result and host result: %f\n", err);


	// Convert grayscale to sobel-grayscale (energy)
	uint8_t * correctOutSobelPixels= (uint8_t *)malloc(width * height);
	convertGray2Sobel(correctOutPixels, width, height, correctOutSobelPixels, x_Sobel, y_Sobel, filterWidth);
	writePnm(correctOutSobelPixels, width, height, concatStr(outFileNameBase, "_sobel_host.pnm"));

	// Convert grayscale to sobel-grayscale (energy) using device 
	uint8_t * correctOutSobelPixelsDevice= (uint8_t *)malloc(width * height);
	convertGray2Sobel(correctOutPixels, width, height, correctOutSobelPixelsDevice, x_Sobel, y_Sobel, filterWidth, true, blockSize);
	writePnm(correctOutSobelPixelsDevice, width, height, concatStr(outFileNameBase, "_sobel_device.pnm"));


	// Free memories
	free(inPixels);
	free(outPixels);
}
