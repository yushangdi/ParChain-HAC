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

#include "linkage.h"
#include "linkage_types.h"
#include "../IO/pointIO.h"
#include "../IO/parseCommandLine.h"
#include "../hac/utils/point.h"

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"


// g++ -O3 -std=c++20 -mcx16  -ldl -pthread -I../external/parlaylib/include linkage.cpp -o linkage

using namespace std;
using namespace HACMatrix;
using parlay::internal::timer;

void run(commandLine& params){
    using T = double;
    char* filename = params.getArgument(0);
    string output = params.getOptionValue("-o", "");
    string method = params.getOptionValue("-method", "invalid");
    int dim = params.getOptionIntValue("-d",2);

    timer t;t.start();
    auto P = pointIO::readNumbersFromFile(filename);
    t.next("load points");
    vector<dendroLine> dendro;
    if(method ==  "complete"){
        SymMatrix<double> *W = getDistanceMatrix(P, dim); 
        using distT = distComplete<T>;
        dendro = chain_linkage_matrix<T, distT>(W);
    }else if(method ==  "ward"){
        SymMatrix<double> *W = getDistanceMatrix(P, dim); 
        using distT = distWard<T>;
        dendro = chain_linkage_matrix<T, distT>(W);
    }else if(method ==  "avg"){
         SymMatrix<double> *W = getDistanceMatrix(P, dim); 
        using distT = distAverage<T>;
        dendro = chain_linkage_matrix<T, distT>(W);
    }else if(method ==  "avgsq"){
         SymMatrix<double> *W = getDistanceMatrix(P, dim, &distancesq); 
        using distT = distAverage<T>;
        dendro = chain_linkage_matrix<T, distT>(W);
    }else{
        cout << "invalid method" << endl;
        exit(1);
    }
    t.next("clustering");
    double checksum = parlay::reduce(parlay::delayed_seq<double>(dendro.size(), [&](size_t i){return dendro[i].height;}));
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
    commandLine P(argc,argv,"[-o <outFile>] [-d <dim>] [-method <method>] <inFile>");
    
    cout << "num workers: " << parlay::num_workers() << endl;

    run(P);
}
