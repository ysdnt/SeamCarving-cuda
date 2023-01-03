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
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
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

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
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

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < height && c < width)
	{
		int i = r * width + c;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299*red + 0.587*green + 0.114*blue;  
	}

}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}
	else // use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
		uint8_t * d_inPixels;
		uint8_t * d_outPixels;
		size_t nBytes = 3 * width * height * sizeof(uint8_t); // 3 phần tử liên tiếp lần lượt là r, g, b nên x3
		CHECK(cudaMalloc(&d_inPixels, nBytes));
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
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

__global__ void convertGray2SobelKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * d_outPixels, int8_t * x_Sobel, int8_t * y_Sobel, uint8_t filterWidth)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	int radius = filterWidth / 2;

	if(r < height && c < width)
	{
		int i = r * width + c;
		int outPixel_x = 0;
		int outPixel_y = 0;

		for(int filterR = 0; filterR < filterWidth; filterR++)
		{
			for(int filterC = 0; filterC < filterWidth; filterC++)
			{
				int filter_idx = filterR * filterWidth + filterC;
				int8_t filterVal_x = x_Sobel[filter_idx]; 
				int8_t filterVal_y = y_Sobel[filter_idx];
				int inPixelsR = r - radius + filterR;
				int inPixelsC = c - radius + filterC;
				inPixelsR = min(max(0, inPixelsR), height - 1);
				inPixelsC = min(max(0, inPixelsC), width - 1);
				uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
				outPixel_x += filterVal_x * inPixel;
				outPixel_y += filterVal_y * inPixel;
			}
		}
		d_outPixels[i] = abs(outPixel_x) + abs(outPixel_y);
	}
}

void convertGray2Sobel(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, int8_t * x_Sobel, int8_t * y_Sobel, 
		uint8_t filterWidth, bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer2;
	timer2.Start();
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
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		uint8_t * d_inPixels;
		uint8_t * d_outPixels;
		int8_t * d_x_Sobel;
		int8_t * d_y_Sobel;
		size_t nBytes = width * height * sizeof(uint8_t);
		size_t filter_nBytes = filterWidth * filterWidth * sizeof(int8_t);

		CHECK(cudaMalloc(&d_inPixels, nBytes));
		CHECK(cudaMalloc(&d_outPixels, nBytes));
		CHECK(cudaMalloc(&d_x_Sobel, filter_nBytes));
		CHECK(cudaMalloc(&d_y_Sobel, filter_nBytes));

		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_x_Sobel, x_Sobel, filter_nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_y_Sobel, y_Sobel, filter_nBytes, cudaMemcpyHostToDevice));

		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		convertGray2SobelKernel<<<gridSize,blockSize>>>(d_inPixels, width, height, d_outPixels, d_x_Sobel, d_y_Sobel, filterWidth);

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
	timer2.Stop();
	float res = timer2.Elapsed();
	printf("Processing time (%s): %f ms\n", 
    		useDevice == true? "use device" : "use host", res);
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
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
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	} 
	convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize); 

	// Compute mean absolute error between host result and device result
	float err = computeError(outPixels, correctOutPixels, width * height);
	printf("Error between device result and host result: %f\n", err);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_gray_host.pnm"));
	writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_gray_device.pnm"));

	// Set up Sobel filters
	uint8_t filterWidth = 3;
	int8_t * x_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	int8_t * y_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	x_Sobel[0] = x_Sobel[6] = y_Sobel[0] = y_Sobel[2] = 1;
	x_Sobel[1] = x_Sobel[4] = x_Sobel[7] = y_Sobel[3] = y_Sobel[4] = y_Sobel[5] = 0;
	x_Sobel[2] = x_Sobel[8] = y_Sobel[6] = y_Sobel[8] = -1;
	x_Sobel[3] = y_Sobel[1] = 2;
	x_Sobel[5] = y_Sobel[7] = -2;

	// Convert grayscale to sobel-grayscale not using device
	uint8_t * OutSobelPixels= (uint8_t *)malloc(width * height);
	uint8_t * correctOutSobelPixels= (uint8_t *)malloc(width * height);
	convertGray2Sobel(outPixels, width, height, correctOutSobelPixels, x_Sobel, y_Sobel, filterWidth);
	writePnm(correctOutSobelPixels, 1, width, height, concatStr(outFileNameBase, "_sobel_host.pnm"));

	convertGray2Sobel(outPixels, width, height, OutSobelPixels, x_Sobel, y_Sobel, filterWidth, true, blockSize);
	writePnm(OutSobelPixels, 1, width, height, concatStr(outFileNameBase, "_sobel_device.pnm"));

	// Free memories
	free(inPixels);
	free(outPixels);
}
