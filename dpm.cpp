#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <iostream>
#include "FastMat.h"
#include "DataSet.h"
#include "Table.h"
#include "Customer.h"

using namespace std;


int MAX_SWEEP = 1500;
int BURNIN = 1000;
int SAMPLE = 20;
int STEP = (MAX_SWEEP - BURNIN) / SAMPLE;

char* result_dir = "./";

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
		double s1=kappa + npoints;
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


class TotalLikelihood : public Task
{
public:
	atomic<int> taskid;
	int ntable;
	int nchunks;
	vector<DPMCustomer>& customers;
	atomic<double> totalsum;
	TotalLikelihood(vector<DPMCustomer>& customers, int nchunks) : nchunks(nchunks), customers(customers) {
	}
	void reset() {
		totalsum = 0;
		taskid = 0;
	}
	void run(int id) {
		// Use thread specific buffer
		SETUP_ID()
			int taskid = this->taskid++; // Oth thread is the main process
		auto range = trange(n, nchunks, taskid); // 2xNumber of Threads chunks			
		double logsum = 0;
		for (auto i = range[0]; i< range[1]; i++) // Operates on its own chunk
		{
			logsum += customers[i].table->dist.likelihood(customers[i].data);
		}
		totalsum = totalsum + logsum;
	}
};



void updateTables(list<DPMTable> & tables)
{
	for (auto tit = tables.begin(); tit != tables.end(); tit++)
	{
		tit->calculateDist();
	}
}


bool compare_clusters(DPMTable& c1, DPMTable& c2)
{
	return c1.npoints > c2.npoints;
}


void reid(list<DPMTable>& tables)
{
	tables.sort(compare_clusters);
	int i = 0;
	for (auto& acul : tables)
		acul.tableid = i++;
}



double eta;
PILL_DEBUG
int main(int argc, char** argv)
{
	debugMode(1);
	generator.seed(time(NULL));
	srand(time(NULL));

	printf("Reading...\n");
	DataSet ds(argc, argv);
	// Default values , 1000 , 100  , out
	if (argc > 4)
		MAX_SWEEP = atoi(argv[4]);
	if (argc > 5)
		BURNIN = atoi(argv[5]);
	if (argc > 6)
		result_dir = argv[6];
	else
	{
		string str(argv[1]);
		result_dir = (char*)str.substr(0, str.find_last_of("/\\")).c_str();
	}

	if (argc > 7)
	{
		SAMPLE = atoi(argv[7]);
	}
	STEP = (MAX_SWEEP - BURNIN) / SAMPLE;
	if (BURNIN >= MAX_SWEEP | STEP == 0) // Housekeeping
	{
		BURNIN = MAX_SWEEP - 2;
		SAMPLE = 1; STEP = 1;
	}

	step();
	// Computing buffer

	string ss(result_dir);
	ofstream nsampleslog(ss.append("nsamples.log"));


	eta = m - d + 1;

	precomputegamLn(2 * (n + d) + 1);  // With 0.5 increments

	Vector priormean(d);

	Matrix priorvariance(d, d);

	Psi = ds.prior;


	init_buffer(1, d);	// Only one calculation buffer

	Psi.r = d; // Last row is shadowed
	Psi.n = d*d;
	mu0 = ds.prior(d).copy();




	eta = eta;
	priorvariance = Psi*((kappa + 1) / ((kappa)*eta));
	priormean = mu0;

	Stut stt(priormean, priorvariance, eta);
	Vector loglik0 = stt.likelihood(ds.data);


	// INITIALIZATION
	list<DPMTable> tables, besttables;
	vector<DPMCustomer> customers, bestcust;
	Matrix     sampledLabels(SAMPLE, n);

	customers.reserve(n);
	tables.emplace_back(d); // First table

	int i, j, k, nsweep, bestiter;


	for (i = 0; i < n; i++)
	{
		customers.emplace_back(ds.data(i), loglik0[i]);
		customers[i].table = tables.begin(); // Assing table pointers to first table initially
		tables.begin()->addInitPoint(ds.data(i));
	}

	tables.front().calculateCov();
	tables.front().calculateDist();

	DPMTable copy(d);
	double newclust, max, sum, randvar, likelihood, best;
	list<DPMTable>::iterator tit, titc;
	best = -INFINITY;
	int nthd = thread::hardware_concurrency();
	TotalLikelihood tl(customers, nthd * 2);
	ThreadPool tpool(nthd);
	init_buffer(nthd, d);
	for (nsweep = 0; nsweep <= MAX_SWEEP; nsweep++)
	{


		Vector logprob = zeros(50);
		int idx;
		int k = 0;
		for (kappa = 0.01; kappa < 5; kappa += 0.1)
		{
			updateTables(tables);
			tl.reset();
			for (i = 0; i < tl.nchunks; i++)
				tpool.submit(tl);
			tpool.waitAll(); // Wait for finishing all jobs
			logprob[k] = tl.totalsum / n;

			k++;
			//cout << logprob << endl;
		}
		idx = sampleFromLog(logprob);
		kappa = idx*0.1 + 0.01;
		//cout << kappa << endl;
		updateTables(tables);

		k = 0;
		for (m = d + 2; m < (d + 2 + 50 * d); m += d)
		{
			Psi = eye(d)*(m - d - 1);
			updateTables(tables);
			tl.reset();
			for (i = 0; i < tl.nchunks; i++)
				tpool.submit(tl);
			tpool.waitAll(); // Wait for finishing all jobs
			logprob[k] = tl.totalsum / n;
			k++;
			//cout << logprob << endl;
		}
		idx = sampleFromLog(logprob);
		m = d + 2 + (idx*d);
		//cout << m << endl;
		Psi = eye(d)*(m - d - 1);
		updateTables(tables);

		likelihood = 0;
		for (i = 0; i < n; i++)
		{
			Vector& x = customers[i].data;
			list<DPMTable>::iterator oldt = customers[i].table;
			copy = *oldt;
			sum = 0;


			oldt->removePoint(x);

			if (oldt->npoints == 0)
				tables.erase(customers[i].table);

			newclust = customers[i].loglik0 + log(alpha);
			max = newclust;

			for (DPMTable& table : tables)
			{
				table.loglikelihood = table.dist.likelihood(x);
				table.logprob = table.loglikelihood + log(table.npoints);
				if (table.logprob > max)
					max = table.logprob;
			}

			for (DPMTable& table : tables)
			{
				table.logprob = exp(table.logprob - max);
				sum += table.logprob;
			}

			sum += exp(newclust - max);

			randvar = urand()*sum;

			for (tit = tables.begin(); tit != tables.end(); tit++)
			{
				if (tit->logprob >= randvar)
				{
					// Find it
					break;
				}
				randvar -= tit->logprob;
			}

			if (tit == tables.end()) // Not in current tables add new one
			{
				tables.emplace_front(d); // New empty table
				tit = tables.begin();
				tit->loglikelihood = customers[i].loglik0;
			}

			customers[i].table = tit;
			likelihood += tit->loglikelihood;


			if (tit == oldt)
				*tit = copy;
			else
				tit->addPoint(x); // Add point to selected one


		}
		reid(tables);
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
			for (tit = tables.begin(), titc = besttables.begin(); tit != tables.end(); tit++, titc++)
				tit->copy = titc;
			bestcust = customers;
			for (DPMCustomer& cust : bestcust)
				cust.table = cust.table->copy;
		}

		if (((MAX_SWEEP - nsweep - 1) % STEP) == 0 && nsweep >= BURNIN)
		{
			int sampleno = (MAX_SWEEP - nsweep - 1) / STEP;
			if (sampleno < SAMPLE)
			{
				for (tit = tables.begin(), i = 0; tit != tables.end(); tit++, i++)
					tit->tableid = i;
				for (i = 0; i < n; i++)
					sampledLabels(sampleno)[i] = customers[i].table->tableid;
			}
		}


		printf("Iter %d Likelihood %f nTables %d\n", nsweep, likelihood, tables.size());
		flush(cout);
	}


	try {
		string s(result_dir);
		ofstream tablefile(s.append("tables.bin"), ios::out | ios::binary); // Be careful result_dir should include '\'

		i = 0;
		int ntables = besttables.size();
		tablefile.write((char*)& ntables, sizeof(int));
		for (DPMTable& table : besttables)
		{
			i++;
			table.tableid = i;
			tablefile << table;
		}
		tablefile.close();



		s.assign(result_dir);
		ofstream customerfile(s.append("customers.bin"), ios::out | ios::binary);
		int ncust = customers.size();
		customerfile.write((char*)& ncust, sizeof(int));
		for (DPMCustomer& cust : bestcust)
		{
			customerfile << cust;
		}
		customerfile.close();


		s.assign(result_dir);
		ofstream labelsout(s.append("Labels.matrix"), ios::out | ios::binary);
		labelsout << sampledLabels;
		labelsout.close();

		nsampleslog.close();

		printf("Best number of tables %d gives likelihood %f in iter %d\n", besttables.size(), best, bestiter);
	}
	catch (exception e)
	{
		cout << "Error in writing files" << endl;
	}
}