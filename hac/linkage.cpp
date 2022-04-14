#include <string>
#include <vector>

// This code is part of the project "ParChain: A Framework for Parallel Hierarchical Agglomerative
// Clustering using Nearest-Neighbor Chain"
// Copyright (c) 2022 Shangdi Yu, Yiqiu Wang, Yan Gu, Laxman Dhulipala, Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "../IO/pointIO.h"
#include "../IO/parseCommandLine.h"
#include "utils/point.h"
#include "clusterer.h"
#include "dist.h"

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"



// g++ -O3 -std=c++20 -mcx16  -ldl -pthread -I../external/parlaylib/include linkage.cpp -o linkage

using namespace std;
using namespace HACTree;
using parlay::internal::timer;

template<int dim>
void run(commandLine& params){

    char* filename = params.getArgument(0);
    string output = params.getOptionValue("-o", "");
    double eps = params.getOptionDoubleValue("-eps", 0);
    // cout << "eps = " << std::setprecision(25) <<  eps << endl;
    string method = params.getOptionValue("-method", "invalid");
    int cache_size = params.getOptionIntValue("-cachesize", 32);
    if(cache_size == 1) cache_size =0;
    cout << "cache_size = " <<  cache_size << endl;

    bool no_cache = cache_size==0;
    timer t;t.start();
    auto P0 = pointIO::readPointsFromFile<point<dim>>(filename);
    parlay::sequence<iPoint<dim>> P = makeIPoint<dim>(P0);
    t.next("load points");
    vector<dendroLine> dendro;
    if(method ==  "complete"){
        dendro = runCompleteHAC<dim>(P, no_cache, cache_size, eps);
    }else if(method ==  "ward"){
       dendro = runWARDHAC<dim>(P, no_cache, cache_size, eps);
    }else if(method ==  "avg"){
        dendro = runAVGHAC<dim>(P, no_cache, cache_size, eps);
    }else if(method ==  "avgsq"){
        dendro = runAVGSQHAC<dim>(P, no_cache, cache_size, eps);
    }else{
        cout << "invalid method" << endl;
        exit(1);
    }
    t.next("clustering");
    double checksum = getCheckSum(dendro);
    cout << "Cost: " << std::setprecision(10) << checksum << endl;

    int n = dendro.size()+1;
    if(output != ""){
        ofstream file_obj;
        file_obj.open(output.c_str()); 
        for(size_t i=0;i<n-1;i++){
            dendro[i].print(file_obj);
        }
        file_obj.close();
    }

    // return dendro;
}

int main(int argc, char *argv[]) {
    commandLine P(argc,argv,"[-o <outFile>] [-d <dim>] [-method <method>] [-cachesize <cachesize>] <inFile>");
    int dim = P.getOptionIntValue("-d",2);
    cout << "num workers: " << parlay::num_workers() << endl;

    if(dim ==2){
        run<2>(P);
    }else if(dim == 3){
         run<3>(P);
    }else if(dim == 4){
         run<4>(P);
    }else if(dim == 5){
         run<5>(P);
    }else if(dim == 6){
         run<6>(P);
    }else if(dim == 7){
         run<7>(P);
    }else if(dim == 8){
         run<8>(P);
    }else if(dim == 9){
         run<9>(P);
    }else if(dim == 10){
         run<10>(P);
    }else{
        cerr << "dim not supported" << endl;
    }
}
