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



vector<float> train(int *sentence, int *trainPositionE1, int *trainPositionE2, int len, vector<int> &tip) {
	vector<float> r;
	r.resize(dimensionC);
	for (int i = 0; i < dimensionC; i++) {
		r[i] = 0;
		int last = i * dimension * window;
		int lastt = i * dimensionWPE * window;
		float mx = -FLT_MAX;
		for (int i1 = 0; i1 <= len - window; i1++) {
			float res = 0;
			int tot = 0;
			int tot1 = 0;
			for (int j = i1; j < i1 + window; j++)  {
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
	return r;
}

void train_gradient(int *sentence, int *trainPositionE1, int *trainPositionE2, int len, int e1, int e2, int r1, float alpha, vector<float> &r,vector<int> &tip, vector<float> &grad)
{
	for (int i = 0; i < dimensionC; i++) {
		if (fabs(grad[i])<1e-8)
			continue;
		int last = i * dimension * window;
		int tot = 0;
		int lastt = i * dimensionWPE * window;
		int tot1 = 0;
		float g1 = grad[i] * (1 -  r[i] * r[i]);
		for (int j = 0; j < window; j++)  {
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
}

float train_bags(string bags_name)
{
	int bags_size = bags_train[bags_name].size();
	double bags_rate = max(1.0,1.0*bags_size/2);
	vector<vector<float> > rList;
	vector<vector<int> > tipList;
	tipList.resize(bags_size);
	int r1 = -1;
	for (int k=0; k<bags_size; k++)
	{
		tipList[k].resize(dimensionC);
		int i = bags_train[bags_name][k];
		if (r1==-1)
			r1 = relationList[i];
		else
			assert(r1==relationList[i]);
		rList.push_back(train(trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], tipList[k]));
	}
	
	vector<float> f_r;	
	
	vector<int> dropout;
	for (int i = 0; i < dimensionC; i++) 
		//dropout.push_back(1);
		dropout.push_back(rand()%2);
	
	vector<float> weight;
	float weight_sum = 0;
	for (int k=0; k<bags_size; k++)
	{
		float s = 0;
		for (int i = 0; i < dimensionC; i++) 
		{
			float tmp = 0;
			for (int j = 0; j < dimensionC; j++)
				tmp+=rList[k][j]*att_W_Dao[r1][j][i];
			s += tmp * matrixRelationDao[r1 * dimensionC + i];
		}
		s = exp(s); 
		weight.push_back(s);
		weight_sum += s;
	}
	for (int k=0; k<bags_size; k++)
		weight[k] /=weight_sum;
	
	float sum = 0;
	for (int j = 0; j < relationTotal; j++) {	
		vector<float> r;
		r.resize(dimensionC);
		for (int i = 0; i < dimensionC; i++) 
			for (int k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
	
		float ss = 0;
		for (int i = 0; i < dimensionC; i++) {
			ss += dropout[i] * r[i] * matrixRelationDao[j * dimensionC + i];
		}
		ss += matrixRelationPrDao[j];
		f_r.push_back(exp(ss));
		sum+=f_r[j];
	}
	
	double rt = (log(f_r[r1]) - log(sum));
	
	vector<vector<float> > grad;
	grad.resize(bags_size);
	for (int k=0; k<bags_size; k++)
		grad[k].resize(dimensionC);
	vector<float> g1_tmp;
	g1_tmp.resize(dimensionC);
	for (int r2 = 0; r2<relationTotal; r2++)
	{	
		vector<float> r;
		r.resize(dimensionC);
		for (int i = 0; i < dimensionC; i++) 
			for (int k=0; k<bags_size; k++)
				r[i] += rList[k][i] * weight[k];
		
		float g = f_r[r2]/sum*alpha1;
		if (r2 == r1)
			g -= alpha1;
		for (int i = 0; i < dimensionC; i++) 
		{
			float g1 = 0;
			if (dropout[i]!=0)
			{
				g1 += g * matrixRelationDao[r2 * dimensionC + i];
				matrixRelation[r2 * dimensionC + i] -= g * r[i];
			}
			g1_tmp[i]+=g1;
		}
		matrixRelationPr[r2] -= g;
	}
		for (int i = 0; i < dimensionC; i++) 
		{
			float g1 = g1_tmp[i];
			double tmp_sum = 0; //for rList[k][i]*weight[k]
			for (int k=0; k<bags_size; k++)
			{
				grad[k][i]+=g1*weight[k];
				for (int j = 0; j < dimensionC; j++)
				{
					grad[k][j]+=g1*rList[k][i]*weight[k]*matrixRelationDao[r1 * dimensionC + i]*att_W_Dao[r1][j][i];
					matrixRelation[r1 * dimensionC + i] += g1*rList[k][i]*weight[k]*rList[k][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  att_W[r1][j][i] += g1*rList[k][i]*weight[k]*rList[k][j]*matrixRelationDao[r1 * dimensionC + i];
				}
				tmp_sum += rList[k][i]*weight[k];
			}	
			for (int k1=0; k1<bags_size; k1++)
			{
				for (int j = 0; j < dimensionC; j++)
				{
					grad[k1][j]-=g1*tmp_sum*weight[k1]*matrixRelationDao[r1 * dimensionC + i]*att_W_Dao[r1][j][i];
					matrixRelation[r1 * dimensionC + i] -= g1*tmp_sum*weight[k1]*rList[k1][j]*att_W_Dao[r1][j][i];
					if (i==j)
					  att_W[r1][j][i] -= g1*tmp_sum*weight[k1]*rList[k1][j]*matrixRelationDao[r1 * dimensionC + i];
				}
			}
		}
	for (int k=0; k<bags_size; k++)
	{
		int i = bags_train[bags_name][k];
		train_gradient(trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], alpha1,rList[k], tipList[k], grad[k]);
		
	}
	return rt;
}

int turn;

int test_tmp = 0;

vector<string> b_train;
vector<int> c_train;
double score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;

int tot_batch;
void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		test_tmp = 0;
	//	for (int k1 = batch; k1 > 0; k1--)
		while (true)
		{

			pthread_mutex_lock (&mutex1);
			if (score_tmp>=score_max)
			{
				pthread_mutex_unlock (&mutex1);
				break;
			}
			score_tmp+=1;
		//	cout<<score_tmp<<' '<<score_max<<endl;
			pthread_mutex_unlock (&mutex1);
			int j = getRand(0, c_train.size());
			//cout<<j<<'|';
			j = c_train[j];
			//cout<<j<<'|';
			//test_tmp+=bags_train[b_train[j]].size();
			//cout<<test_tmp<<' ';
			score += train_bags(b_train[j]);
		}
		//cout<<endl;
}

void train() {
	int tmp = 0;
	b_train.clear();
	c_train.clear();
	for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		int max_size = 1;//it->second.size()/2;
		for (int i=0; i<max(1,max_size); i++)
			c_train.push_back(b_train.size());
		b_train.push_back(it->first);
		tmp+=it->second.size();
	}
	cout<<c_train.size()<<endl;
	
	att_W.resize(relationTotal);
	for (int i=0; i<relationTotal; i++)
	{
		att_W[i].resize(dimensionC);
		for (int j=0; j<dimensionC; j++)
		{
			att_W[i][j].resize(dimensionC);
			att_W[i][j][j] = 1.00;//1;
		}
	}
	att_W_Dao = att_W;

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

	//	len = trainLists.size();
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
		//	cout<<k<<endl;
			memcpy(positionVecDaoE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
			memcpy(positionVecDaoE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
			memcpy(matrixW1PositionE1Dao, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(matrixW1PositionE2Dao, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(wordVecDao, wordVec, dimension * wordTotal * sizeof(float));

			memcpy(matrixW1Dao, matrixW1, sizeof(float) * dimensionC * dimension * window);
			memcpy(matrixB1Dao, matrixB1, sizeof(float) * dimensionC);
			memcpy(matrixRelationPrDao, matrixRelationPr, relationTotal * sizeof(float));				//add
			memcpy(matrixRelationDao, matrixRelation, dimensionC*relationTotal * sizeof(float));
			att_W_Dao = att_W;
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
	output_model = 1;
	logg = fopen("log.txt","w");
	cout<<"Init Begin."<<endl;
	init();
	//for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	//	cout<<it->first<<endl;
	cout<<bags_train.size()<<' '<<bags_test.size()<<endl;
	cout<<"Init End."<<endl;
	train();
	fclose(logg);
}
