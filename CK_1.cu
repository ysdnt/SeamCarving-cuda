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

void convertRgb2Gray(uchar3 * inPixels, int width, int height,
		uint8_t * outPixels)
{
	GpuTimer timer;
	timer.Start();    
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
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (convertRgb2Gray) (%s): %f ms\n\n", 
		 "use device" , time);
}

void convertGray2Sobel(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels,
		int8_t * x_Sobel, int8_t * y_Sobel, uint8_t filterWidth)
{
	GpuTimer timer;
	timer.Start();
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
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (convertGray2Sobel) (%s): %f ms\n\n", 
		 "use device" , time);
}

void computeEnergy(uint8_t * inPixels, int width, int height,
		int * outPixels)
{
	for (int c = 0; c < width; c++)
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

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void find2removeSeam(int new_width, int &i, uint8_t * correctOutSobelPixels, int * correctSumEnergy, int * correctSeam, int8_t * trace, uchar3 * inPixels, int width, int height)
{
	GpuTimer timer;
	timer.Start();
	for (i; i > new_width; i--)
	{
		computeEnergy(correctOutSobelPixels, i, height, correctSumEnergy);
		computeSumEnergy(correctOutSobelPixels, i, height, correctSumEnergy, trace);
		findSeam(correctSumEnergy, trace, i, height, correctSeam);
		removeSeam(inPixels, correctOutSobelPixels, correctSeam, i, height);
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (find2removeSeam) (%s): %f ms\n\n", 
		 "use device" , time);
}
int main(int argc, char ** argv)
{	
	if (argc != 3 && argc != 4)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

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
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "ck1_gray_host.pnm"));

	// Convert grayscale to sobel-grayscale (energy)
	uint8_t * correctOutSobelPixels= (uint8_t *)malloc(width * height);
	convertGray2Sobel(correctOutPixels, width, height, correctOutSobelPixels, x_Sobel, y_Sobel, filterWidth);
	writePnm(correctOutSobelPixels, width, height, concatStr(outFileNameBase, "ck1_sobel_host.pnm"));

	int new_width = 2 * width / 3; //Default
	if (argc == 4)
	{
		new_width = atoi(argv[3]);
	}  
	int i = width;

	// Find and remove seam using host
	int * correctSumEnergy = (int *)malloc(width * height * sizeof(int));
	int8_t * trace = (int8_t *)malloc(width * height * sizeof(int8_t));
	int * correctSeam = (int *)malloc(height * sizeof(int));
	find2removeSeam(new_width, i, correctOutSobelPixels, correctSumEnergy, correctSeam, trace, inPixels, width, height);
	writePnm(inPixels, i, height, concatStr(outFileNameBase, "ck1_seam_host.pnm"));

	// Free memories
	free(inPixels);
	free(x_Sobel);
	free(y_Sobel);
	free(correctOutPixels);
	free(correctOutSobelPixels);
	free(correctSumEnergy);
	free(trace);
	free(correctSeam);
}
