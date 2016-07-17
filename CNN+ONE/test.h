#ifndef TEST_H
#define TEST_H
#include "init.h"
#include <algorithm>
#include <map>

int tipp = 0;
float ress = 0;

vector<double> test(int *sentence, int *testPositionE1, int *testPositionE2, int len, float *r) {
	int tip[dimensionC];
		
	for (int i = 0; i < dimensionC; i++) {
		int last = i * dimension * window;
		int lastt = i * dimensionWPE * window;
		float mx = -FLT_MAX;
		for (int i1 = -window+1; i1 < len; i1++) {
			float res = 0;
			int tot = 0;
			int tot1 = 0;
			for (int j = i1; j < i1 + window; j++)  
			if (j>=0&&j<len){
				int last1 = sentence[j] * dimension;
			 	for (int k = 0; k < dimension; k++) {
			 		res += matrixW1[last + tot] * wordVec[last1+k];
			 		tot++;
			 	}
			 	int last2 = testPositionE1[j] * dimensionWPE;
			 	int last3 = testPositionE2[j] * dimensionWPE;
			 	for (int k = 0; k < dimensionWPE; k++) {
			 		res += matrixW1PositionE1[lastt + tot1] * positionVecE1[last2+k];
			 		res += matrixW1PositionE2[lastt + tot1] * positionVecE2[last3+k];
			 		tot1++;
			 	}
			}
			else
			{
				tot+=dimension;
				tot1+=dimensionWPE;
			}
			if (res > mx) mx = res;
		}
		r[i] = mx + matrixB1[i];
	}

	for (int i = 0; i < dimensionC; i++)
		r[i] = CalcTanh(r[i]);

	vector<double> res;
	double tmp = 0;
	for (int j = 0; j < relationTotal; j++) {
		float s = 0;
		for (int i = 0; i < dimensionC; i++)
			s +=  0.5 * matrixRelation[j * dimensionC + i] * r[i];
		s += matrixRelationPr[j];
		s = exp(s);
		tmp+=s;
		res.push_back(s);
	}
	for (int j = 0; j < relationTotal; j++) 
		res[j]/=tmp;
	return res;
}


bool cmp(pair<string, pair<int,double> > a,pair<string, pair<int,double> >b)
{
    return a.second.second>b.second.second;
}

vector<string> b;
double tot;

vector<pair<string, pair<int,double> > >aa;

pthread_mutex_t mutex;
vector<int> ll_test;

void* testMode(void *id ) 
{
	int ll = ll_test[(long long)id];
	int rr;
	if ((long long)id==num_threads-1)
		rr = b.size();
	else
		rr = ll_test[(long long)id+1];
	//cout<<ll<<' '<<rr<<' '<<((long long)id)<<endl;
	float *r = (float *)calloc(dimensionC, sizeof(float));
	double eps = 0.1;
	int ttt = -1;
	for (int ii = ll; ii < rr; ii++)
	{
		vector<double> sum, sum_neg;
		vector<vector<double> > scoreList;
		scoreList.resize(relationTotal);
		for (int j = 0; j < relationTotal; j++)
			sum.push_back(0.0);
		sum_neg = sum;
		map<int,int> ok;
		ok.clear();
		for (int k=0; k<bags_test[b[ii]].size(); k++)
		{
			int i = bags_test[b[ii]][k];
			if (testrelationList[i]>0&&ttt==-1)
				ttt = testrelationList[i];
		}
		for (int k=0; k<bags_test[b[ii]].size(); k++)
		{
			int i = bags_test[b[ii]][k];
			ok[testrelationList[i]]=1;
			vector<double> score = test(testtrainLists[i],  testPositionE1[i], testPositionE2[i], testtrainLength[i], r);
			if (k<=0)
			for (int j = 0; j < relationTotal; j++) 
				sum[j] = max(sum[j], score[j]);
		}
		pthread_mutex_lock (&mutex);
		for (int j = 1; j < relationTotal; j++) 
		{
			int i = bags_test[b[ii]][0];
			aa.push_back(make_pair(wordList[testheadList[i]]+' '+wordList[testtailList[i]]+' '+nam[j],make_pair(ok.count(j),sum[j])));
		}
		pthread_mutex_unlock(&mutex);
	}

	free(r);
}

double max_pre = 0;

void test() {
	aa.clear();
	b.clear();
	tot = 0;
	ll_test.clear();
	vector<int> b_sum;
	b_sum.clear();
	for (map<string,vector<int> >:: iterator it = bags_test.begin(); it!=bags_test.end(); it++)
	{
		
		map<int,int> ok;
		ok.clear();
		for (int k=0; k<it->second.size(); k++)
		{
			int i = it->second[k];
			if (testrelationList[i]>0)
				ok[testrelationList[i]]=1;		}
		tot+=ok.size();
		{
			b.push_back(it->first);
			b_sum.push_back(it->second.size());
		}
	}
	for (int i=1; i<b_sum.size(); i++)
		b_sum[i] += b_sum[i-1];
	int now = 0;
	ll_test.resize(num_threads+1);
	for (int i=0; i<b_sum.size(); i++)
		if (b_sum[i]>=b_sum[b_sum.size()-1]/num_threads*now)
		{
			ll_test[now] = i;
			now+=1;
		}
	cout<<"tot:\t"<<tot<<endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (int a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, testMode,  (void *)a);
	for (int a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	cout<<"begin sort"<<' '<<aa.size()<<endl;
	sort(aa.begin(),aa.end(),cmp);
	double correct=0;
	float correct1 = 0;
	for (int i=0; i<min(2000,int(aa.size())); i++)
	{
		if (aa[i].second.first!=0)
			correct1++;	
		float precision = correct1/(i+1);
		float recall = correct1/tot;
		if (i%100==0)
			cout<<"precision:\t"<<correct1/(i+1)<<'\t'<<"recall:\t"<<correct1/tot<<endl;
		if (recall>0.1&&precision>max_pre)
		{
			max_pre = precision;
		}
	}
	//assert(version!="");
	//if (output_flag)
	{
		FILE* f = fopen(("out/pr"+version+".txt").c_str(), "w");
		for (int i=0; i<2000; i++)
		{
			if (aa[i].second.first!=0)
				correct++;	
			fprintf(f,"%lf\t%lf\t%lf\t%s\n",correct/(i+1), correct/tot,aa[i].second.second, aa[i].first.c_str());
		}
		fclose(f);
		if (!output_model)
			return ;
		FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\t%d\n", dimensionC, dimension, window, dimensionWPE);
		for (int i = 0; i < dimensionC; i++) {
			for (int j = 0; j < dimension * window; j++)
				fprintf(fout, "%f\t",matrixW1[i* dimension*window+j]);
			for (int j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE1[i* dimensionWPE*window+j]);
			for (int j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE2[i* dimensionWPE*window+j]);
			fprintf(fout, "%f\n", matrixB1[i]);
		}
		fclose(fout);

		fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relationTotal, dimensionC);
		for (int i = 0; i < relationTotal; i++) {
			for (int j = 0; j < dimensionC; j++)
				fprintf(fout, "%f\t", matrixRelation[i * dimensionC + j]);
			fprintf(fout, "\n");
		}
		for (int i = 0; i < relationTotal; i++) 
			fprintf(fout, "%f\t",matrixRelationPr[i]);
		fprintf(fout, "\n");
		fclose(fout);

		fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\n", PositionTotalE1, PositionTotalE2, dimensionWPE);
		for (int i = 0; i < PositionTotalE1; i++) {
			for (int j = 0; j < dimensionWPE; j++)
				fprintf(fout, "%f\t", positionVecE1[i * dimensionWPE + j]);
			fprintf(fout, "\n");
		}
		for (int i = 0; i < PositionTotalE2; i++) {
			for (int j = 0; j < dimensionWPE; j++)
				fprintf(fout, "%f\t", positionVecE2[i * dimensionWPE + j]);
			fprintf(fout, "\n");
		}
		fclose(fout);
	
		fout = fopen(("./out/word2vec.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n",wordTotal,dimension);
		for (int i = 0; i < wordTotal; i++)
		{
			for (int j=0; j<dimension; j++)
				fprintf(fout,"%f\t",wordVec[i*dimension+j]);
			fprintf(fout,"\n");
		}
		fclose(fout);
	}
}

#endif
