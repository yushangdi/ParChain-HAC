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

#pragma once


#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace HACMatrix {

// symmetric matrix, diagonal is all diag_val = 0 by default
template<class T> 
struct SymMatrix{

    std::size_t n = 0;
    T* distmat = nullptr;
    std::size_t distmat_size = 0;
    T diag_val = 0;

    SymMatrix(){}
    SymMatrix(std::size_t _n, T _diag_val = 0){
        n = _n;
        diag_val = _diag_val;
        distmat_size = n*(n-1)/2;
        distmat = (T *)malloc(distmat_size * sizeof(T));
    }

    void init(std::size_t _n){
        n = _n;
        distmat_size = n*(n-1)/2;
        distmat = (T *)malloc(distmat_size * sizeof(T));
    }

    inline long getInd(std::size_t i, std::size_t j){
        if(i == j) return distmat_size;
        long r_ = static_cast<long>(i);
        long c_ = static_cast<long>(j);
        return (((2*n-r_-3) * r_) >> 1 )+ (c_)-1;
    }
    
    inline void setDiag(T v){
        diag_val = v;
    }

    inline T get(std::size_t r_, std::size_t c_){
        if(r_ == c_) return diag_val;
        if(r_ > c_) swap(r_,c_);
        return( distmat[getInd(r_, c_)] );
    }

    inline void update(std::size_t r_, std::size_t c_, T dist){
        if(r_ == c_) return;
        if(r_ > c_) swap(r_,c_);
        distmat[getInd(r_, c_)] = dist;
    }

    ~SymMatrix(){
        free(distmat);
    }
    void printMatrix(){
    // for (std::size_t i=0; i<distmat_size; i++){
    //     cout << distmat[i] << endl;
    // }
    for (std::size_t i=0; i<n; i++){
        for (std::size_t j=0; j<n; j++) {
            std::cout << get(i,j) << " ";
        }
        std::cout << std::endl;
    } 
    cout << "===" << endl;
    for (std::size_t i=0; i<n; i++){
        for (std::size_t j=i+1; j<n; j++) {
            std::cout << i << " " << j << " " <<getInd(i,j) << " " << get(i,j) << std::endl;
        }
    } 
    }
};

inline double distancesq(parlay::slice<double*, double*> a, parlay::slice<double*, double*> b) {
    double sum=0;
    for (std::size_t k=0; k<a.size(); k++) {
        float tmp=a[k]-b[k];
        sum+=(tmp*tmp);
    }
    return sum;
}

inline double distance(parlay::slice<double*, double*> a, parlay::slice<double*, double*> b) {
    return sqrt(distancesq(a,b));
}


template <class F>
SymMatrix<double>* getDistanceMatrix(parlay::sequence<double>& datapoints, int dim, F f) {
    int n = datapoints.size()/dim;
    SymMatrix<double> *matrix = new SymMatrix<double>(n);
    parlay::parallel_for(0, n, [&](size_t i){
        parlay::parallel_for(i+1, n,[&] (size_t j){
	        matrix->update(i, j, f(datapoints.cut(i*dim, (i+1)*dim), datapoints.cut(j*dim, (j+1)*dim)));
        });
    });
    matrix->setDiag(0);
    return matrix;
}

//return a symmetric matrix representation of the euclidean distance
// between all pairs of datapoints.
// assume all points have the same dimension and are in dense format
SymMatrix<double>* getDistanceMatrix(parlay::sequence<double>& datapoints, int dim) {
    return getDistanceMatrix(datapoints, dim, &distance);
}


}  //namespace HACMatrix