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

using namespace std;

double score = 0;
float alpha1;

struct timeval t_start,t_end; 
long start,end;

void time_begin()
{
  
  gettimeofday(&t_start, NULL); 
  start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 
}
void time_end()
{
  gettimeofday(&t_end, NULL); 
  end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
  cout<<"time(s):\t"<<(double(end)-double(start))/1000<<endl;
}




double train(int flag, int *sentence, int *trainPositionE1, int *trainPositionE2, int len, int e1, int e2, int r1, float &res, float &res1, float *matrixW1Dao, float *matrixB1Dao, float *r, float *matrixRelationDao,
	float *positionVecDaoE1, float *positionVecDaoE2, float*matrixW1PositionE1Dao, float*matrixW1PositionE2Dao,  float alpha) {
		int tip[dimensionC];
			
		for (int i = 0; i < dimensionC; i++) {
			int last = i * dimension * window;
			int lastt = i * dimensionWPE * window;
			float mx = -FLT_MAX;
			for (int i1 = 0; i1 <= len - window; i1++) {
				float res = 0;
				int tot = 0;
				int tot1 = 0;
				for (int j = i1; j < i1 + window; j++)  
				if (j>=0&&j<len){
					int last1 = sentence[j] * dimension;
				 	for (int k = 0; k < dimension; k++) {
				 		res += matrixW1Dao[last + tot] * wordVecDao[last1+k];
				 		tot++;
				 	}
				 	int last2 = trainPositionE1[j] * dimensionWPE;
				 	int last3 = trainPositionE2[j] * dimensionWPE;
				 	for (int k = 0; k < dimensionWPE; k++) {
				 		res += matrixW1PositionE1Dao[lastt + tot1] * positionVecDaoE1[last2+k];
				 		res += matrixW1PositionE2Dao[lastt + tot1] * positionVecDaoE2[last3+k];
				 		tot1++;
				 	}
				}
				else
				{
					tot+=dimension;
					tot1+=dimensionWPE;
				}
				if (res > mx) {
					mx = res;
					tip[i] = i1;
				}
			}
			r[i] = mx + matrixB1Dao[i];
		}

		for (int i = 0; i < dimensionC; i++) {
			r[i] = CalcTanh(r[i]);
		}
		
		vector<int> dropout;
		for (int i = 0; i < dimensionC; i++) 
		//	dropout.push_back(1);
			dropout.push_back(rand()%2);
		
		vector<double> f_r;	
		double sum = 0;
		for (int j = 0; j < relationTotal; j++) {
			float s = 0;
			for (int i = 0; i < dimensionC; i++) {
				s += dropout[i] * r[i] * matrixRelationDao[j * dimensionC + i];
			}
			s += matrixRelationPrDao[j];
			f_r.push_back(exp(s));
			sum+=f_r[j];
		}
		double rt = log(f_r[r1]) - log(sum);
		if (flag)
		{
			float s1, g, s2;
			for (int i = 0; i < dimensionC; i++) {
				if (dropout[i]==0)
					continue;
				int last = i * dimension * window;
				int tot = 0;
				int lastt = i * dimensionWPE * window;
				int tot1 = 0;
				float g1 = 0;
				for (int r2 = 0; r2<relationTotal; r2++)
				{
					g = f_r[r2]/sum*alpha;//sigmod(margin*(marginNegative + s2)) * margin * alpha;
					if (r2 == r1)
						g -= alpha;
					g1 += g * matrixRelationDao[r2 * dimensionC + i] * (1 -  r[i] * r[i]);
					matrixRelation[r2 * dimensionC + i] -= g * r[i];
					if (i==0)
						matrixRelationPr[r2] -= g;
				}
				for (int j = 0; j < window; j++)  
				if (tip[i]+j>=0&&tip[i]+j<len){
					int last1 = sentence[tip[i] + j] * dimension;
					for (int k = 0; k < dimension; k++) {
						matrixW1[last + tot] -= g1 * wordVecDao[last1+k];
						wordVec[last1 + k] -= g1 * matrixW1Dao[last + tot];
						tot++;
					}
					int last2 = trainPositionE1[tip[i] + j] * dimensionWPE;
					int last3 = trainPositionE2[tip[i] + j] * dimensionWPE;
					for (int k = 0; k < dimensionWPE; k++) {
						matrixW1PositionE1[lastt + tot1] -= g1 * positionVecDaoE1[last2 + k];
						matrixW1PositionE2[lastt + tot1] -= g1 * positionVecDaoE2[last3 + k];
						positionVecE1[last2 + k] -= g1 * matrixW1PositionE1Dao[lastt + tot1];
						positionVecE2[last3 + k] -= g1 * matrixW1PositionE2Dao[lastt + tot1];
						tot1++;
					}
				}
				matrixB1[i] -= g1;
			}

			for (int i = 0; i < dimensionC; i++) {
				int last = dimension * window * i;
				res1+=Belt * matrixB1Dao[i] * matrixB1Dao[i];

				for (int j = dimension * window -1; j >= 0; j--) {
					res1+= Belt * matrixW1Dao[last + j] * matrixW1Dao[last + j];
					matrixW1[last + j] += - Belt * matrixW1Dao[last + j] * alpha * 2; 
				}

				last = dimensionWPE * window * i;
				for (int j = dimensionWPE * window -1; j>=0; j--) {
					matrixW1PositionE1[last + j] += -Belt * matrixW1PositionE1Dao[last + j] * alpha * 2;
					matrixW1PositionE2[last + j] += -Belt * matrixW1PositionE2Dao[last + j] * alpha * 2;
				}

				matrixB1[i] += -Belt * matrixB1Dao[i] *alpha * 2;
			}
		}
		return rt;
}

int turn;

vector<string> b_train;
vector<int> c_train;
double score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;
void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		float *r = (float *)calloc(dimensionC, sizeof(float));
		{
				float res = 0;
				float res1 = 0;
				for (int k1 = batch; k1 > 0; k1--)
				{
					int j = getRand(0, c_train.size());
					j = c_train[j];
					double tmp1 = -1e8;
					int tmp2 = -1;
					vector<double> s_tmp;
					for (int k=0; k<bags_train[b_train[j]].size(); k++)
					{
						int i = bags_train[b_train[j]][k];
						double tmp = train(0,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
						positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
					//	score+=tmp;
						if (tmp1<tmp)
						{
							tmp1 = tmp;
							tmp2 = i;
						}
						if (k==0)
							s_tmp.push_back(exp(tmp));
						else
							s_tmp.push_back(exp(tmp)+s_tmp[k-1]);
					}
					score+= train(1,trainLists[tmp2], trainPositionE1[tmp2], trainPositionE2[tmp2], trainLength[tmp2], headList[tmp2], tailList[tmp2], relationList[tmp2], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
						positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
				}
		}
		free(r);
}

void train() {
	int tmp = 0;
	b_train.clear();
	c_train.clear();
	for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		for (int i=0; i<max(1,1); i++)
			c_train.push_back(b_train.size());
		b_train.push_back(it->first);
		tmp+=it->second.size();
	}
	cout<<c_train.size()<<endl;

	float con = sqrt(6.0/(dimensionC+relationTotal));
	float con1 = sqrt(6.0/((dimensionWPE+dimension)*window));
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

	for (int i = 0; i < dimensionC; i++) {
		int last = i * window * dimension;
		for (int j = dimension * window - 1; j >=0; j--)
			matrixW1[last + j] = getRandU(-con1, con1);
		last = i * window * dimensionWPE;
		float tmp1 = 0;
		float tmp2 = 0;
		for (int j = dimensionWPE * window - 1; j >=0; j--) {
			matrixW1PositionE1[last + j] = getRandU(-con1, con1);
			tmp1 += matrixW1PositionE1[last + j]  * matrixW1PositionE1[last + j] ;
			matrixW1PositionE2[last + j] = getRandU(-con1, con1);
			tmp2 += matrixW1PositionE2[last + j]  * matrixW1PositionE2[last + j] ;
		}
		matrixB1[i] = getRandU(-con1, con1);
	}

	for (int i = 0; i < relationTotal; i++) 
	{
		matrixRelationPr[i] = getRandU(-con, con);				//add
		for (int j = 0; j < dimensionC; j++)
			matrixRelation[i * dimensionC + j] = getRandU(-con, con);
	}

	for (int i = 0; i < PositionTotalE1; i++) {
		float tmp = 0;
		for (int j = 0; j < dimensionWPE; j++) {
			positionVecE1[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE1[i * dimensionWPE + j] * positionVecE1[i * dimensionWPE + j];
		}
	}

	for (int i = 0; i < PositionTotalE2; i++) {
		float tmp = 0;
		for (int j = 0; j < dimensionWPE; j++) {
			positionVecE2[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE2[i * dimensionWPE + j] * positionVecE2[i * dimensionWPE + j];
		}
	}

	matrixRelationDao = (float *)calloc(dimensionC*relationTotal, sizeof(float));
	matrixW1Dao =  (float*)calloc(dimensionC * dimension * window, sizeof(float));
	matrixB1Dao =  (float*)calloc(dimensionC, sizeof(float));
	
	positionVecDaoE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
	positionVecDaoE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
	matrixW1PositionE1Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixW1PositionE2Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	/*time_begin();
	test();
	time_end();*/
//	return;
	for (turn = 0; turn < trainTimes; turn ++) {
		len = c_train.size();
		npoch  =  len / (batch * num_threads);
		alpha1 = alpha*rate/batch;

		score = 0;
		score_max = 0;
		score_tmp = 0;
		double score1 = score;
		time_begin();
		for (int k = 1; k <= npoch; k++) {
			score_max += batch * num_threads;
			memcpy(positionVecDaoE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
			memcpy(positionVecDaoE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
			memcpy(matrixW1PositionE1Dao, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(matrixW1PositionE2Dao, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(wordVecDao, wordVec, dimension * wordTotal * sizeof(float));

			memcpy(matrixW1Dao, matrixW1, sizeof(float) * dimensionC * dimension * window);
			memcpy(matrixB1Dao, matrixB1, sizeof(float) * dimensionC);
			memcpy(matrixRelationPrDao, matrixRelationPr, relationTotal * sizeof(float));				//add
			memcpy(matrixRelationDao, matrixRelation, dimensionC*relationTotal * sizeof(float));
			pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
			for (int a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void *)a);
			for (int a = 0; a < num_threads; a++)
			pthread_join(pt[a], NULL);
			free(pt);
			if (k%(npoch/5)==0)
			{
				cout<<"npoch:\t"<<k<<'/'<<npoch<<endl;
				time_end();
				time_begin();
				cout<<"score:\t"<<score-score1<<' '<<score_tmp<<endl;
				score1 = score;
			}
		}
		printf("Total Score:\t%f\n",score);
		printf("test\n");
		test();
		//if ((turn+1)%1==0) 
		//	rate=rate*reduce;
	}
	test();
	cout<<"Train End"<<endl;
}

int main(int argc, char ** argv) {
	output_model  = 1;
	logg = fopen("log.txt","w");
	cout<<"Init Begin."<<endl;
	init();
	cout<<bags_train.size()<<' '<<bags_test.size()<<endl;
	cout<<"Init End."<<endl;
	train();
	fclose(logg);
	return 0;
}
