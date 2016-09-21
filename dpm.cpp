#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <iostream>

#include "DebugUtils.h"
#include "DataSet.h"
#include "util.h"
#include "Table.h"
#include "Customer.h"

using namespace std;


int MAX_SWEEP=100;
int BURNIN=10;
char* result_dir = "./";

// Variables
int d,n;
double m,eta;

class DPMTable : public Table
{
public:
	DPMTable(int d):Table(d)
	{}
	void operator=(const DPMTable& d)
	{
		Table::operator=(d);
	}
	void calculateDist()
	{
		double s1=Global::kappa + npoints;
		dist.eta = m + 1  + this->npoints - d; 
		dist.mu = (sampleMean*(npoints/s1) + mu0 * (kappa/s1) ) ;
		Vector& diff = mu0 - sampleMean;
		dist.cholsigma = 
			((Psi + sampleScatter +
			((diff)>>(diff))
			*(npoints*kappa/(s1)))
			*((s1+1)/(s1*dist.eta))).chol();
		dist.calculateNormalizer();
	}

	void addPoint(Vector& v)
	{
	npoints++;  // npoint is current number of points in the table
	Vector& diff = (v - sampleMean); // Get abstract of the output buffer
	sampleScatter  =  ( sampleScatter + (diff>>diff)*((npoints-1.0)/(npoints))) ; 
	sampleMean = ((sampleMean * (npoints-1) ) + v ) / (npoints); 
	calculateDist();
	}

	void removePoint(Vector& v)
	{
	if (npoints<=0)
		npoints = 0;


	if (npoints > 1)
	{
		Vector& diff = (v - sampleMean); // Get abstract of the output buffer
		sampleScatter  =  ( sampleScatter - (diff>>diff)*(npoints/(npoints-1.0))) ; 
		sampleMean = ((sampleMean * (npoints/(npoints - 1.0))) - v *(1.0/(npoints - 1.0)))  ; 
		npoints--;
		calculateDist();
	}
	else
	{
		sampleScatter.zero();
		sampleMean.zero();
		npoints--;
	}
	}

	list<DPMTable>::iterator copy;

	friend ostream& operator<<(ostream& os, const DPMTable& t)
	{
	// os should be binary file
	os.write((char*) &t.tableid,sizeof(int)); 
	os.write((char*) &t.npoints,sizeof(int)); 
	// os.write((char*) &t.logprob,sizeof(double)); 
	os.write((char*) &t.loglikelihood,sizeof(double)); 
	os << t.sampleScatter;
	os << t.sampleMean;
	os << t.dist;
	return os;
	}
	
};








class DPMCustomer : public Customer
{
public:
	DPMCustomer(Vector& data,double likelihood) : Customer(data,likelihood)
	{}
	list<DPMTable>::iterator table;
	friend ostream& operator<<(ostream& os, const DPMCustomer& c)
	{
	os.write((char*) &c.loglik0,sizeof(double));
	os.write((char*) &c.table->tableid,sizeof(int));
	os << c.data;
	return os;
	}
};








int SAMPLE=(MAX_SWEEP-BURNIN)/10; 

PILL_DEBUG
int main(int argc,char** argv)
{
	debugMode(1);
	char* datafile = argv[1];
	char* priorfile = argv[2];
	char* configfile = argv[3];

	generator.seed(time(NULL));
	srand(time(NULL));
	// Default values , 1000 , 100  , out
	if (argc>4)
		MAX_SWEEP = atoi(argv[4]);
	if (argc>5)
		BURNIN = atoi(argv[5]);
	if (argc > 6)
		result_dir = argv[6];
	SAMPLE = (MAX_SWEEP-BURNIN)/10; 
	if (argc > 7)
		SAMPLE = atoi(argv[7]); // Keep sample with SAMPLE iteration apart
	
	step();
					 // Computing buffer

	string ss(result_dir);
	ofstream nsampleslog(ss.append("nsamples.log"));

	printf("Reading...\n");
	DataSet ds(datafile,priorfile,configfile);
	d = ds.d;
	n = ds.n;
	eta = ds.m - d + 1;

	precomputeGammaLn(2*(n+d)+1);  // With 0.5 increments
	
	Vector priormean(d); 
	
	Matrix priorvariance(d,d);
	
	Global::Psi = ds.prior;
	

	init_buffer(1,d);	// Only one calculation buffer

	Global::Psi.r = d; // Last row is shadowed
	Global::Psi.n = d*d;
	Global::mu0 =  ds.prior(ds.d).copy();
	

	

	Global::eta = eta;
	priorvariance = Global::Psi*((Global::kappa+1)/((Global::kappa)*eta));
	priormean = Global::mu0;
	
	Stut stt(priormean,priorvariance,eta); 
	Vector loglik0 = stt.likelihood(ds.data);


	// INITIALIZATION
	list<DPMTable> tables,besttables;
	vector<DPMCustomer> customers,bestcust;
	Matrix     sampledLabels((MAX_SWEEP-BURNIN)/SAMPLE + 1,ds.n);

	customers.reserve(n);
	tables.emplace_back(d); // First table

	int i,j,k,nsweep,bestiter;


	for (i=0;i<n;i++)
	{
		customers.emplace_back(ds.data(i),loglik0[i]);
		customers[i].table = tables.begin(); // Assing table pointers to first table initially
		tables.begin()->addInitPoint(ds.data(i));
	}
	
	tables.front().calculateCov();
	tables.front().calculateDist();
	
	DPMTable copy(d);
	double newclust,max,sum,randvar,likelihood,best;
	list<DPMTable>::iterator tit,titc;
	best = -INFINITY;
	for (nsweep=0;nsweep <= MAX_SWEEP;nsweep++)
	{
		likelihood = 0;
		for (i=0;i<n;i++)
		{
			Vector& x = customers[i].data;
			list<DPMTable>::iterator oldt = customers[i].table;
			copy = *oldt;
			sum = 0;


			oldt->removePoint(x);

			if (oldt->npoints == 0)
				tables.erase(customers[i].table);

			newclust = customers[i].loglik0 + log(Global::alpha);
			max = newclust;

			for (DPMTable& table : tables)
			{
				table.loglikelihood = table.dist.likelihood(x);
				table.logprob = table.loglikelihood + log(table.npoints);
				if (table.logprob>max)
					max =  table.logprob;
			}

			for (DPMTable& table : tables)
			{
				table.logprob = exp(table.logprob - max);
				sum += table.logprob;
			}
			
			sum += exp(newclust - max);

			randvar = urand()*sum;

			for(tit=tables.begin();tit!=tables.end();tit++)
			{
				if (tit->logprob >= randvar)
				{
					// Find it
					break;
				}
				randvar -= tit->logprob;
			}

			if (tit==tables.end()) // Not in current tables add new one
			{
				tables.emplace_front(d); // New empty table
				tit = tables.begin();
				tit->loglikelihood = customers[i].loglik0;
			}

			customers[i].table = tit;
			likelihood += tit->loglikelihood;


			if (tit==oldt)
				*tit = copy;
			else
				tit->addPoint(x); // Add point to selected one


		}

		for (DPMTable& table : tables)
		{
			nsampleslog << table.npoints << " ";
		}
		nsampleslog << endl;

		

		if (likelihood > best)
		{
			best = likelihood;
			bestiter = nsweep;
			besttables = tables;
			for(tit=tables.begin(),titc=besttables.begin();tit!=tables.end();tit++,titc++)
				tit->copy = titc;
			bestcust = customers;
			for (DPMCustomer& cust : bestcust)
				cust.table = cust.table->copy;
		}

		if  (((nsweep-BURNIN)%SAMPLE)==0 && nsweep >= BURNIN)
		{
			for (tit=tables.begin(),i=0;tit!=tables.end();tit++,i++)
				tit->tableid = i;
			for(i=0;i<ds.n;i++)
				sampledLabels((nsweep-BURNIN)/SAMPLE)[i] = customers[i].table->tableid;
		}

		printf("Iter %d Likelihood %f nTables %d\n",nsweep,likelihood,tables.size());
		flush(cout);
	}


	
	string s(result_dir);
	ofstream tablefile(s.append("tables.bin"),ios::out | ios::binary); // Be careful result_dir should include '\'

	i=0;
	int ntables = besttables.size();
	tablefile.write((char*)& ntables,sizeof(int));
	for (DPMTable& table : besttables)
	{
		i++;
		table.tableid = i;
		tablefile << table;
	}
	tablefile.close();



	s.assign(result_dir);
	ofstream customerfile( s.append("customers.bin"),ios::out | ios::binary);
	int ncust = customers.size();
	customerfile.write((char*)& ncust,sizeof(int));
	for (DPMCustomer& cust : bestcust)
	{
		customerfile << cust;
	}
	customerfile.close();


	s.assign(result_dir);
	ofstream labelsout( s.append("Labels.matrix"),ios::out | ios::binary);
	labelsout << sampledLabels;
	labelsout.close();

	nsampleslog.close();

	printf("Best number of tables %d gives likelihood %f in iter %d\n",besttables.size(),best,bestiter);
}