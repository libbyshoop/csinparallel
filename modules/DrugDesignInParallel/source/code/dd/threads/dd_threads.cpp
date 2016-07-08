/** drug design example with C++11 threads and tbb containers */
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_queue.h>
#include <tbb/parallel_sort.h>
#include <thread>


#define DEFAULT_max_ligand 7
#define DEFAULT_nligands 120
#define DEFAULT_nthreads 4
#define DEFAULT_protein "the cat in the hat wore the hat to the cat hat party"


using namespace std;


// key-value pairs, used for both Map() out/Reduce() in and for Reduce() out
struct Pair {
  int key;
  string val;
  Pair(int k, const string &v) {key = k;  val = v;}
};


// MR class provides map-reduce structural pattern
class MR {
private:
  int max_ligand;
  int nligands;
  int nthreads;
  string protein;
  string SENTINEL;  // indicates end of task queue


  tbb::concurrent_bounded_queue<string> tasks;
  tbb::concurrent_vector<Pair> pairs;
  vector<Pair> results;


  void Generate_tasks(tbb::concurrent_bounded_queue<string> &q);
  void do_Maps(void);
  void Map(const string &str, tbb::concurrent_vector<Pair> &pairs);
  void do_sort(tbb::concurrent_vector<Pair> &vec);
  int Reduce(int key, const tbb::concurrent_vector<Pair> &pairs, int index, 
             string &values);
public:
  MR() { SENTINEL=""; }
  const vector<Pair> &run(int ml, int nl, int nt, const string& p);
};


// Auxiliary routines
class Help {
public:
  static string get_ligand(int max_ligand);
  static int score(const char*, const char*);
};




// Main program
int main(int argc, char **argv) {
  int max_ligand = DEFAULT_max_ligand;
  int nligands = DEFAULT_nligands;
  int nthreads = DEFAULT_nthreads;
  string protein = DEFAULT_protein;
  
  if (argc > 1)
    max_ligand = strtol(argv[1], NULL, 10);
  if (argc > 2)
    nligands = strtol(argv[2], NULL, 10);
  if (argc > 3)
    nthreads = strtol(argv[3], NULL, 10);
  if (argc > 4)
    protein = argv[4];
  // command-line args parsed

  cout << "max_ligand=" << max_ligand 
       << "  nligands=" << nligands
       << "  nthreads=" << nthreads << endl;

  MR map_reduce;
  vector<Pair> results = 
    map_reduce.run(max_ligand, nligands, nthreads, protein);


  cout << "maximal score is " << results[0].key 
       << ", achieved by ligands " << endl 
       << results[0].val << endl;


  return 0;
}


/*  class MR methods */


const vector<Pair> &MR::run(int ml, int nl, int nt, const string& p) {
  max_ligand = ml;  nligands = nl;  nthreads = nt;  protein = p;


  Generate_tasks(tasks);
  // assert -- tasks is non-empty


  thread *pool = new thread[nthreads];
  for (int i = 0;  i < nthreads;  i++)
    pool[i] = thread(&MR::do_Maps, this);


  for (int i = 0;  i < nthreads;  i++)
    pool[i].join();


  do_sort(pairs);


  int next = 0;  // index of first unprocessed pair in pairs[]
  while (next < pairs.size()) {
    string values;
    values = "";
    int key = pairs[next].key;
    next = Reduce(key, pairs, next, values);
    Pair p(key, values);
    results.push_back(p);
  }


  return results;
}


void MR::Generate_tasks(tbb::concurrent_bounded_queue<string> &q) {
  for (int i = 0;  i < nligands;  i++) {
    q.push(Help::get_ligand(max_ligand));
  }
  q.push(SENTINEL);
}


void MR::do_Maps(void) {
  string lig;
  tasks.pop(lig);
  while (lig != SENTINEL) {
    Map(lig, pairs);
    tasks.pop(lig);
  }
  tasks.push(SENTINEL);  // restore end marker for another thread
}


void MR::Map(const string &ligand, tbb::concurrent_vector<Pair> &pairs) {
  Pair p(Help::score(ligand.c_str(), protein.c_str()), ligand);
  pairs.push_back(p);
}


bool compare(const Pair &p1, const Pair &p2) {
  return p1.key > p2.key;
}


void MR::do_sort(tbb::concurrent_vector<Pair> &vec) {
  tbb::parallel_sort(vec.begin(), vec.end(), compare);
}


int MR::Reduce(int key, const tbb::concurrent_vector<Pair> &pairs, int index, 
           string &values) {
  while (index < pairs.size() && pairs[index].key == key) 
    values += pairs[index++].val + " ";
  return index;
}




/*  class Help methods */


// returns arbitrary string of lower-case letters of length at most max_ligand
string Help::get_ligand(int max_ligand) {
  int len = 1 + rand()%max_ligand;
  string ret(len, '?');
  for (int i = 0;  i < len;  i++)
    ret[i] = 'a' + rand() % 26;  
  return ret;
}


int Help::score(const char *str1, const char *str2) {
  if (*str1 == '\0' || *str2 == '\0')
    return 0;
  // both argument strings non-empty
  if (*str1 == *str2)
    return 1 + score(str1 + 1, str2 + 1);
  else // first characters do not match
    return max(score(str1, str2 + 1), score(str1 + 1, str2));
}
