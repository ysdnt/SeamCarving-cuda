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

// __global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
// 		uint8_t * outPixels)
// {
// 	// TODO
//     // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  

// }

void convertRgb2Gray(uchar3 * inPixels, int width, int height,
		uint8_t * outPixels)
{
	GpuTimer timer;
	timer.Start();
	// if (useDevice == false)
	// {
          
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
	//}
	// else // use device
	// {
	// 	cudaDeviceProp devProp;
	// 	cudaGetDeviceProperties(&devProp, 0);
	// 	printf("GPU name: %s\n", devProp.name);
	// 	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	// 	// TODO: Allocate device memories

	// 	// TODO: Copy data to device memories

	// 	// TODO: Set grid size and call kernel (remember to check kernel error)

	// 	// TODO: Copy result from device memories

	// 	// TODO: Free device memories

	// }
	timer.Stop();
	float time = timer.Elapsed();
	// printf("Processing time (%s): %f ms\n\n", 
	// 		useDevice == true? "use device" : "use host", time);
	printf("Processing time: %f ms\n\n", time);
}

void convertRgb2GraySobel(uchar3 * inPixels, int width, int height,
		uint8_t * outPixels,
		int8_t * x_Sobel, int8_t * y_Sobel, uint8_t filterWidth)
{
	GpuTimer timer;
	timer.Start();
	// if (useDevice == false)
	// {
          
		// for (int r = 0; r < height; r++)
		// {
		// 	for (int c = 0; c < width; c++)
		// 	{
		// 		int i = r * width + c;
		// 		uint8_t red = inPixels[3 * i];
		// 		uint8_t green = inPixels[3 * i + 1];
		// 		uint8_t blue = inPixels[3 * i + 2];
		// 		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
		// 	}
		// }
	// for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	// {
	// 	for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
	// 	{
	// 		uint8_t outPixel = 0;
	// 		for (int filterR = 0; filterR < filterWidth; filterR++)
	// 		{
	// 			for (int filterC = 0; filterC < filterWidth; filterC++)
	// 			{
	// 				int8_t filterVal_x = x_Sobel[filterR * filterWidth + filterC];
	// 				int8_t filterVal_y = y_Sobel[filterR * filterWidth + filterC];
	// 				int inPixelsR = outPixelsR - filterWidth/2 + filterR;
	// 				int inPixelsC = outPixelsC - filterWidth/2 + filterC;
	// 				inPixelsR = min(max(0, inPixelsR), height - 1);
	// 				inPixelsC = min(max(0, inPixelsC), width - 1);
	// 				uchar3 inPixel = inPixels[inPixelsR * width + inPixelsC];
	// 				outPixel.x += filterVal * inPixel.x;
	// 				outPixel.y += filterVal * inPixel.y;
	// 				outPixel.z += filterVal * inPixel.z;
	// 			}
	// 		}
	// 		outPixels[outPixelsR*width + outPixelsC] = make_uchar3(outPixel.x, outPixel.y, outPixel.z); 
	// 	}
	// }
	//}
	// else // use device
	// {
	// 	cudaDeviceProp devProp;
	// 	cudaGetDeviceProperties(&devProp, 0);
	// 	printf("GPU name: %s\n", devProp.name);
	// 	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	// 	// TODO: Allocate device memories

	// 	// TODO: Copy data to device memories

	// 	// TODO: Set grid size and call kernel (remember to check kernel error)

	// 	// TODO: Copy result from device memories

	// 	// TODO: Free device memories

	// }
	timer.Stop();
	float time = timer.Elapsed();
	// printf("Processing time (%s): %f ms\n\n", 
	// 		useDevice == true? "use device" : "use host", time);
	printf("Processing time: %f ms\n\n", time);
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

	// Set up Sobel filters
	uint8_t filterWidth = 3;
	int8_t * x_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	int8_t * y_Sobel= (int8_t *)malloc(filterWidth * filterWidth);
	x_Sobel[0] = x_Sobel[6] = y_Sobel[0] = y_Sobel[2] = 1;
	x_Sobel[1] = x_Sobel[4] = x_Sobel[7] = y_Sobel[3] = y_Sobel[4] = y_Sobel[5] = 0;
	x_Sobel[2] = x_Sobel[8] = y_Sobel[6] = y_Sobel[8] = -1;
	x_Sobel[3] = y_Sobel[1] = 2;
	x_Sobel[5] = y_Sobel[7] = -2;
	
	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);
	// convertRgb2GraySobel(inPixels, width, height, correctOutPixels, x_Sobel, y_Sobel, filterWidth);

	// Convert RGB to grayscale using device
	// uint8_t * outPixels= (uint8_t *)malloc(width * height);
	// dim3 blockSize(32, 32); // Default
	// if (argc == 5)
	// {
	// 	blockSize.x = atoi(argv[3]);
	// 	blockSize.y = atoi(argv[4]);
	// } 
	// convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize); 

	// // Compute mean absolute error between host result and device result
	// float err = computeError(outPixels, correctOutPixels, width * height);
	// printf("Error between device result and host result: %f\n", err);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	//writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(x_Sobel);
	free(y_Sobel);
	free(correctOutPixels);
	free(outPixels);
}
