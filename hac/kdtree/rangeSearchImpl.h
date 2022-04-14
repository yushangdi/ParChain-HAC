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

#include "parlay/parallel.h"
#include "parlay/utilities.h"
#include "kdtree.h"

using namespace std;

namespace HACTree {

    template <int dim, typename objT, typename nodeT, typename F>
    void rangeTraverse(nodeT *Q, objT center, double r, F* f) {
    int relation = Q->boxBallCompare(center, r, Q->pMin, Q->pMax);
    if(relation == Q->boxExclude) return;
    if (f->isComplete(Q)) return;

    if (Q->isLeaf()) {
        for(int i=0; i<Q->size(); ++i) {
            if (Q->itemInBall(center, r, Q->items[i])) {
                f->checkComplete(Q->items[i]);
            }
        }
    } else {
        if(f->Par(Q)){
            parlay::par_do([&](){
                rangeTraverse<dim, objT, nodeT, F>(Q->left, center, r, f);
            },[&](){    
                rangeTraverse<dim, objT, nodeT, F>(Q->right, center, r, f);
            });
        }else{
            rangeTraverse<dim, objT, nodeT, F>(Q->left, center, r, f);
            rangeTraverse<dim, objT, nodeT, F>(Q->right, center, r, f);
        }
    }
  }

}