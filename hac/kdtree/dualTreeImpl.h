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

#include <limits>
#include <fstream>
#include <string>
#include <vector>
#include "kdtree.h"

using namespace std;

#define dualtree_spawn_macro(_condA, _condB, _Q1, _R1, _Q2, _R2){ \
    if(_condA && _condB){ \
       parlay::par_do([&](){dualtree<nodeT, F>(_Q1, _R1, f, false);}, \
          [&](){dualtree<nodeT, F>(_Q2, _R2, f, false);});  \
    }else if(_condA){ \
        dualtree<nodeT, F>(_Q1, _R1, f, false); \
    }else{ \
        dualtree<nodeT, F>(_Q2, _R2, f, false); \
    } }

namespace HACTree {
        
    template<class nodeT, class F>
    void dualtree_serial(nodeT *Q, nodeT *R, F *f, bool check = true){

        if(f->Score(Q,R,check)) return;

        if(f->isLeaf(Q) && f->isLeaf(R)){
            f->BasePost(Q,R);
        }else if (f->isLeaf(Q)){
           double dLeft = f->NodeDistForOrder(Q, R->left);
	       double dRight = f->NodeDistForOrder(Q, R->right);
           if(f->SpawnOrder(dLeft , dRight)){
              if(!f->Score(dRight, Q, R->right)) dualtree_serial<nodeT, F>(Q, R->right, f, false);
              if(!f->Score(dLeft, Q, R->left)) dualtree_serial<nodeT, F>(Q, R->left, f, false);
           }else{
              if(!f->Score(dLeft, Q, R->left)) dualtree_serial<nodeT, F>(Q, R->left, f, false);
              if(!f->Score(dRight, Q, R->right)) dualtree_serial<nodeT, F>(Q, R->right, f, false);
           }
           f->QLPost(Q,R);
        }else if(f->isLeaf(R)){
           double dLeft = f->NodeDistForOrder(Q->left, R);
	       double dRight = f->NodeDistForOrder(Q->right, R);
           if(f->SpawnOrder(dLeft , dRight)){
              if(!f->Score(dRight, Q->right, R)) dualtree_serial<nodeT, F>(Q->right, R, f, false);
              if(!f->Score(dLeft, Q->left, R)) dualtree_serial<nodeT, F>(Q->left, R, f, false);
           }else{
              if(!f->Score(dLeft, Q->left, R)) dualtree_serial<nodeT, F>(Q->left, R, f, false);
              if(!f->Score(dRight, Q->right, R)) dualtree_serial<nodeT, F>(Q->right, R, f, false);
           }
            f->RLPost(Q,R);
        }else{
            pair<double, pair<nodeT *, nodeT *>> callOrder[4];
            callOrder[0] = make_pair(f->NodeDistForOrder(Q->left, R->left), make_pair(Q->left, R->left));
            callOrder[1] = make_pair(f->NodeDistForOrder(Q->right, R->left), make_pair(Q->right, R->left));
            callOrder[2] = make_pair(f->NodeDistForOrder(Q->left, R->right), make_pair(Q->left, R->right));
            callOrder[3] = make_pair(f->NodeDistForOrder(Q->right, R->right), make_pair(Q->right, R->right));
            sort(callOrder, callOrder + 4);
            for (int cc = 0; cc < 4; ++ cc) {
                int c = f->SpawnOrder(cc);
                nodeT *QQ = callOrder[c].second.first;
                nodeT *RR = callOrder[c].second.second;
                if(!f->Score(callOrder[c].first, QQ, RR)) dualtree_serial<nodeT, F>(QQ, RR, f, false);
            }
            f->Post(Q,R);
        }
    }

    template<class nodeT, class F>
    void dualtree(nodeT *Q, nodeT *R, F *f, bool check = true){
        if(Q->size() + R->size() < 4000){
            dualtree_serial<nodeT, F>(Q, R, f, check);
            return;
        }

        if(f->Score(Q,R,check)) return;

        if(f->isLeaf(Q) && f->isLeaf(R)){ // not parallel bc leaf size should be small
            f->BasePost(Q,R);
        }else if (f->isLeaf(Q)){
           double dLeft = f->NodeDistForOrder(Q, R->left);
	       double dRight = f->NodeDistForOrder(Q, R->right);
            bool condA = !f->Score(dRight, Q, R->right);
            bool condB = !f->Score(dLeft, Q, R->left);
           if(f->SpawnOrder(dLeft , dRight)){
               dualtree_spawn_macro(condA, condB, Q, R->right, Q, R->left);
           }else{
               dualtree_spawn_macro(condB, condA, Q, R->left, Q, R->right);
           }
           f->QLPost(Q,R);
        }else if(f->isLeaf(R)){
           double dLeft = f->NodeDistForOrder(Q->left, R);
	       double dRight = f->NodeDistForOrder(Q->right, R);
            bool condA = !f->Score(dRight,Q->right, R);
            bool condB = !f->Score(dLeft, Q->left, R);
           if(f->SpawnOrder(dLeft , dRight)){
               dualtree_spawn_macro(condA, condB, Q->right, R, Q->left, R);
           }else{
              dualtree_spawn_macro(condB, condA, Q->left, R, Q->right, R);
           }
            f->RLPost(Q,R);
        }else{
            pair<double, pair<nodeT *, nodeT *>> callOrder[4];
            callOrder[0] = make_pair(f->NodeDistForOrder(Q->left, R->left), make_pair(Q->left, R->left));
            callOrder[1] = make_pair(f->NodeDistForOrder(Q->right, R->left), make_pair(Q->right, R->left));
            callOrder[2] = make_pair(f->NodeDistForOrder(Q->left, R->right), make_pair(Q->left, R->right));
            callOrder[3] = make_pair(f->NodeDistForOrder(Q->right, R->right), make_pair(Q->right, R->right));
            sort(callOrder, callOrder + 4);
            parlay::parallel_for(0,4,[&](int cc) {
                int c = f->SpawnOrder(cc);
                nodeT *QQ = callOrder[c].second.first;
                nodeT *RR = callOrder[c].second.second;
                if(!f->Score(callOrder[c].first, QQ, RR)) dualtree<nodeT, F>(QQ, RR, f, false);
            },1);
            f->Post(Q,R);
        }
    }

}