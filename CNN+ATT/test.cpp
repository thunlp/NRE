#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <cmath>
#include <pthread.h>
#include <iostream>

#include<assert.h>
#include<ctime>
#include<sys/time.h>

#include "init.h"
#include "test.h"

void preprocess()
{

	matrixRelation = (float *)calloc(dimensionC * relationTotal, sizeof(float));
	matrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
	matrixRelationPrDao = (float *)calloc(relationTotal, sizeof(float));
	wordVecDao = (float *)calloc(dimension * wordTotal, sizeof(float));
	positionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
	positionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
	
	matrixW1 = (float*)calloc(dimensionC * dimension * window, sizeof(float));
	matrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixB1 = (float*)calloc(dimensionC, sizeof(float));
	
	att_W.resize(relationTotal);
	for (int i=0; i<relationTotal; i++)
	{
		att_W[i].resize(dimensionC);
		for (int j=0; j<dimensionC; j++)
			att_W[i][j].resize(dimensionC);
	}
	version = "";
	
	FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d%d", &dimensionC, &dimension, &window, &dimensionWPE);
	for (int i = 0; i < dimensionC; i++) {
		for (int j = 0; j < dimension * window; j++)
			fscanf(fout, "%f", &matrixW1[i* dimension*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fscanf(fout, "%f", &matrixW1PositionE1[i* dimensionWPE*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fscanf(fout, "%f", &matrixW1PositionE2[i* dimensionWPE*window+j]);
		fscanf(fout, "%f", &matrixB1[i]);
	}
	fclose(fout);

	fout = fopen(("./out/matrixRl.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d", &relationTotal, &dimensionC);
	for (int i = 0; i < relationTotal; i++) {
		for (int j = 0; j < dimensionC; j++)
			fscanf(fout, "%f", &matrixRelation[i * dimensionC + j]);
	}
	for (int i = 0; i < relationTotal; i++) 
		fscanf(fout, "%f", &matrixRelationPr[i]);
	fclose(fout);

	fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d%d", &PositionTotalE1, &PositionTotalE2, &dimensionWPE);
	for (int i = 0; i < PositionTotalE1; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fscanf(fout, "%f", &positionVecE1[i * dimensionWPE + j]);
	}
	for (int i = 0; i < PositionTotalE2; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fscanf(fout, "%f", &positionVecE2[i * dimensionWPE + j]);
	}
	fclose(fout);

	fout = fopen(("./out/word2vec.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d",&wordTotal,&dimension);
	for (int i = 0; i < wordTotal; i++)
	{
		for (int j=0; j<dimension; j++)
			fscanf(fout,"%f", &wordVec[i*dimension+j]);
	}
	fclose(fout);
	fout = fopen(("./out/att_W.txt"+version).c_str(), "r");
	fscanf(fout,"%d%d", &relationTotal, &dimensionC);
	for (int r1 = 0; r1 < relationTotal; r1++) {
		for (int i = 0; i < dimensionC; i++)
		{
			for (int j = 0; j < dimensionC; j++)
				fscanf(fout, "%f", &att_W[r1][i][j]);
		}
	}
	fclose(fout);
}

int main()
{
	init();
	preprocess();
	test();
	return 0;
}