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

#include <iostream>
#include <fstream> 
#include <math.h>
#include <atomic>
#include <tuple>
#include <limits>
#include "point.h"
#include "parlay/utilities.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "utils.h"

#define SIZE_T_MAX std::numeric_limits<size_t>::max()
#define NO_NEIGH -1


using namespace std;


namespace HACTree{
  
struct EDGE{//nodes unordered
    volatile int first;
    volatile int second;
    volatile double w;//weight

    EDGE(int t_u, int t_v, double t_w):first(t_u), second(t_v), w(t_w){}
    EDGE():first(NO_NEIGH), second(NO_NEIGH), w(std::numeric_limits<double>::max()){}

    inline void print(){
        std::cout << "(" << first << ", " <<  second << "):" << w << std::endl;
    }

    inline double getW() const {return w;}
    inline std::pair<int, int> getE() const {return std::make_pair(first,second);}
    inline void update(int t_u, int t_v, double t_w){
      first = t_u;
      second = t_v;
      w = t_w;
    }
};


struct edgeComparator2{
    double eps = 1e-20;
    edgeComparator2(){}
    edgeComparator2(double _eps):eps(_eps){}
    bool operator () (EDGE i, EDGE j) {
      // if(abs(i.getW() - j.getW()) < i.getW()*std::numeric_limits<double>::epsilon()) return i.second < j.second;
      if(abs(i.getW() - j.getW()) <= eps) return i.second < j.second;
      return i.getW() < j.getW();
      }
};




struct TreeChainInfo{
  parlay::sequence<int> terminal_nodes;// must be cluster id
  parlay::sequence<int> chain;//chain[i] is cluster i's nn, NO_NEIGH for unknown or invalid// -1 for unknown, -2 for invalid
  parlay::sequence<bool> is_terminal;
  parlay::sequence<bool> flag;
  int chainNum;

  parlay::sequence<pair<int, double>> revTerminals; // indexed by ith temrinal nodes
  static constexpr pair<double, long> invalid_rev = make_pair(numeric_limits<double>::max(), (long)-1);
  pair<double, long> *chainRev; // reverse chain, indexed by cid TODO: change to just double?


  TreeChainInfo(int n, double _eps = 1e-20){
    terminal_nodes = parlay::sequence<int>(n);
    parlay::parallel_for(0,n,[&](int i){terminal_nodes[i] = i;});
    chain = parlay::sequence<int>(n, NO_NEIGH);
    is_terminal = parlay::sequence<bool>(n, true);
    flag = parlay::sequence<bool>(n);
    chainNum = n;

    revTerminals = parlay::sequence<pair<int, double>>(n);

    chainRev = (pair<double, long> *)malloc(sizeof(pair<double, long>) * n);//can't use util's writemin in parlay sequence data
    parlay::parallel_for(0,n,[&](int i){chainRev[i] = invalid_rev;});
    // EC2 = LDS::PairComparator21<pair<long, double>>(_eps);
  }

  ~TreeChainInfo(){
    free(chainRev);
  }

  inline void updateChain(int cid, int nn, double w){
    chain[cid] = nn;
    utils::writeMin(&chainRev[nn], make_pair(w,(long)cid), std::less<pair<double, long> >()); 
  }
  // then in checking, only check for -1
  inline void invalidate(int cid, int code){
    chain[cid] = code;
    chainRev[cid] = invalid_rev;
  }

  // inline void invalidateRev(int cid){
  //   chainRev[cid] = invalid_rev;
  // }

  //get the rev of ith terminal nodes 
  inline pair<int, double> getChainPrev(int i){
    return revTerminals[i];
  }

  inline int getNN(int cid){return chain[cid];}
  inline bool isTerminal(int cid){return is_terminal[cid];}

  // update findNN, terminal_nodes and chainNum
  template<class F>
  inline void next(F *finder, int round){
    parlay::parallel_for(0, chainNum, [&](int i){
      is_terminal[terminal_nodes[i]] = false;
    });
    int C = finder->C;
    parlay::parallel_for(0, C, [&](int i){
      int cid = finder->activeClusters[i];
      int nn = getNN(cid);
      //getNN(getNN(cid)) == NO_NEIGH
      flag[i] = nn == NO_NEIGH || finder->justMerge(nn, round) ;// only merged clusters have negative neighbor in chain ok because -2 won't be in active clusters
      is_terminal[cid] = flag[i];
    });
    chainNum = parlay::pack_into(make_slice(finder->activeClusters).cut(0,C), flag, terminal_nodes);

    parlay::parallel_for(0, chainNum, [&](int i){
      revTerminals[i] =  make_pair((int)chainRev[terminal_nodes[i]].second, chainRev[terminal_nodes[i]].first);
    });
  }

};

}
