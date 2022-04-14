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

#include <atomic>

#include "utils/point.h"
#include "utils/node.h"
#include "kdtree/kdtree.h"
#include "treeUtilities.h"

#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "parlay/utilities.h"


using namespace std;

namespace HACTree{
template<class pointT>
inline double pointDist(pointT* p, pointT* q){
  return p->pointDist(q);
}

template<class pointT>
inline double pointDist(pointT p, pointT q){
  return p.pointDist(q);
}

template<class pointT>
inline double getPointId(pointT* p){
  return p->idx();
}

template<class pointT>
inline double getPointId(pointT p){
  return p.idx();
}

#define LINKAGE_AVE_BRUTE_THRESH 100000
    //find the average　distance between P1 and P2, return random pair of points
    template<class pointT, class ARR>
    inline pair<pair<int, int>, double> bruteForceAverage(ARR P1, ARR P2, int n1, int n2, bool div = true){
        pair<int, int> result = make_pair(getPointId(P1[0]), getPointId(P2[0]));
        std::atomic<double> total_d; total_d.store(0);
        long total_n = (long)n1 * (long)n2;
        // intT eltsPerCacheLine = 128 /sizeof(double);
        if(total_n < LINKAGE_AVE_BRUTE_THRESH){ 
            double local_d = 0;
            for(int i=0; i< n1; ++i){
                for(int j=0;j<n2; ++j){
                    local_d +=  pointDist<pointT>(P1[i], P2[j]);
                }
            }
            total_d.store(local_d);
        }else{
            if(n1 < n2){
                swap(n1, n2);
                swap(P1, P2);
            }
            int nBlocks1 =  parlay::num_workers() * 8;

            int BLOCKSIZE1 = max(1,n1/nBlocks1);

            if(n1 > nBlocks1*BLOCKSIZE1) BLOCKSIZE1 += 1;

            parlay::parallel_for(0, nBlocks1, [&](int ii){
                double thread_sum = 0;
                for (int i=ii*BLOCKSIZE1; i< min((ii+1)*BLOCKSIZE1, n1); ++i){
                    for (int j=0; j< n2; ++j){
                        thread_sum += pointDist<pointT>(P1[i], P2[j]);
                    }
                }
                parlay::write_add(&total_d, thread_sum);
            },1);

        }

        if(div) return make_pair(result, total_d.load()/total_n);
        return make_pair(result, total_d.load());
    }

    template<class pointT, class nodeT>
    inline pair<pair<int, int>, double> bruteForceAverage(nodeT *inode,  nodeT *jnode, pointT *clusteredPts){
        pointT *P1 = clusteredPts + inode->getOffset();
        pointT *P2 = clusteredPts + jnode->getOffset();
        int n1 = inode->n;
        int n2 = jnode->n;
        return bruteForceAverage<pointT, pointT*>(P1, P2, n1, n2, true);
    }

   //find the farthest pair of points in t1 and t2
    template<class _objT>
    inline pair<pair<int, int>, double> bruteForceFarthest(parlay::slice<_objT **, _objT **> P1, parlay::slice<_objT **, _objT **> P2){
        pair<int, int> result = make_pair(-1,-1);
        double maxd = 0;
        for(size_t i=0; i< P1.size(); ++i){
            for(size_t j=0;j< P2.size(); ++j){
            double d = P1[i]->pointDist(P2[j]);
            if(d > maxd){
                result = make_pair(P1[i]->idx(),P2[j]->idx());
                maxd = d;
            }
            // else if(d==maxd && P2[j]->idx()<result.second){
            //     result = make_pair(P1[i]->idx(),P2[j]->idx());
            //     maxd = d;
            // }
            }
        }
        return make_pair(result, maxd);
    }

    template<int dim, class nodeT, class _objT>
    inline pair<pair<int, int>, double> bruteForceFarthest(nodeT *t1, nodeT *t2){
        parlay::slice<_objT **, _objT **> P1 = t1->getItems();
        parlay::slice<_objT **, _objT **> P2 = t2->getItems();
        return bruteForceFarthest(P1, P2);
    }

template<int dim>
struct distComplete {
  typedef node<dim, iPoint<dim>, nodeInfo> kdnodeT;
  typedef tree<dim, iPoint<dim>, nodeInfo> kdtreeT;
  using M = MarkClusterId<kdnodeT>; //mark the all points kdtree
  typedef tuple<int, int, long> countCacheT;//(round, count, cid), use long to pack to 2^i bytes

  static const Method method = COMP;
  bool squared = false;
  M marker;
  int round = 0;
  int n;
  kdtreeT **kdtrees;
  countCacheT *countTbs; // cluster to count
  parlay::sequence<int> clusterOffsets; // used to distribute flags chunks
  parlay::sequence<bool> flags;

  distComplete(UnionFind::ParUF<int> *t_uf, iPoint<dim>* PP){
      marker = M(t_uf);
      n = t_uf->size();
      kdtrees = (kdtreeT **) malloc(n*sizeof(kdtreeT *));
      parlay::parallel_for(0,n,[&](int i){
        kdtrees[i] = new kdtreeT(PP+i, i);
      });

      int PNum =  parlay::num_workers();
      countTbs = (countCacheT *)malloc(static_cast<size_t>(sizeof(countCacheT)) * PNum * n);
      parlay::parallel_for(0, static_cast<size_t>(n) * PNum,[&](size_t i){
        countTbs[i] = make_tuple(1, 0, (long)-1);
      });
      clusterOffsets = parlay::sequence<int>(n);
      parlay::parallel_for(0, n,[&](size_t i){
        clusterOffsets[i] = i+1; // result of scan([1,1,1,1...])
      });
      flags = parlay::sequence<bool>(n);
  }

  ~distComplete(){
    free(kdtrees);
    free(countTbs);   
  }

  inline static void printName(){
    cout << "Complete" << endl;
  }

  countCacheT *initClusterTb(int pid, int C){return countTbs+(n*pid);}

  inline static bool doRebuild(){return false;}
  inline double updateDistO(double d1, double d2, double nql, double nqr, double nr, double dij){
    return max(d1,d2);
  }

  inline double updateDistN(double d1, double d2, double d3, double d4, 
                       double nql, double nqr, double nrl, double nrr,
                       double dij, double dklr){
    return max(max(max(d1,d2), d3), d4);
  }

  double getDistNaive(int cid1, int cid2, double lb = -1, double ub = numeric_limits<double>::max(), bool par = true){ 
    double result;
    if(kdtrees[cid1]->size() + kdtrees[cid2]->size() < 200){
      pair<pair<int, int>, double> result = bruteForceFarthest<dim, kdnodeT, iPoint<dim>>(kdtrees[cid1], kdtrees[cid2]);
      return result.second;
    }

    EDGE e = EDGE(-1,-1,lb);;
    if(lb == -1){
        lb = kdtrees[cid1]->at(0)->pointDist(kdtrees[cid2]->at(0));
        e = EDGE(kdtrees[cid1]->at(0)->idx(),kdtrees[cid2]->at(0)->idx(),lb);
        if(lb > ub) return LARGER_THAN_UB;
    }

    BCFP<kdnodeT> fComp = BCFP<kdnodeT>(e, ub);//FComp(LDS::EDGE(cid1,cid2, result));
    if(par){
        dualtree<kdnodeT, BCFP<kdnodeT>>(kdtrees[cid1], kdtrees[cid2], &fComp); 
    }else{
        dualtree_serial<kdnodeT, BCFP<kdnodeT>>(kdtrees[cid1], kdtrees[cid2], &fComp); 
    }    
    result = fComp.getResultW();
    return result;
  }

   double getDistNaive(Node<dim> *inode,  Node<dim> *jnode, double lb = -1, double ub = numeric_limits<double>::max(), bool par = true){ 
    return getDistNaive(inode->cId, jnode->cId, lb, ub, par);
  }

    // range query radius
    inline double getBall(Node<dim>* query, double beta){
        return beta;
    }

    template<class F>
    void update(int _round, F *finder){
      if(finder->C==1) return;
      round = _round;

      int  C = finder->C;
      parlay::parallel_for(0, C, [&](int i){
        clusterOffsets[i] = finder->getNode(finder->activeClusters[i])->size();
      });
      parlay::scan_inclusive_inplace(clusterOffsets.cut(0,C));

      parlay::parallel_for(0,C,[&](int i){
        int cid  = finder->activeClusters[i];
        Node<dim> *clusterNode = finder->getNode(cid);
        if(clusterNode->round == round){//merged this round, build new tree
          int cid1 = clusterNode->left->cId;
          int cid2 = clusterNode->right->cId;
          int start = 0;
          if(i>0) start = clusterOffsets[i-1];
          kdtreeT *new_tree = new kdtreeT(kdtrees[cid1], kdtrees[cid2], flags.cut(start, clusterOffsets[i]));
          delete kdtrees[cid1];
          delete kdtrees[cid2];
          kdtrees[cid] = new_tree;
        }
        
      });

      if(marker.doMark(C, round)){ 
          HACTree::singletree<kdnodeT, M, int>(finder->kdtree, &marker, marker.initVal);
      }
      round++;// when using it, we want to use the next round
    }
};

template<int dim>
struct distWard {
  Method method = WARD;
  bool squared = false;
  int min_n = 1;

  //TODO: mark minN on each tree node for performance
  //using MCenter = HACTree::MarkMinN<pointT, nodeInfo>; 
  typedef node<dim, iPoint<dim>, nodeInfo> kdnodeT;

  inline static void printName(){
    cout << "WARD" << endl;
  }

  inline static bool doRebuild(){return true;}
  inline double updateDistO(double dik, double djk, double ni, double nj, double nk, double dij){
    double ntotal = ni + nj + nk;
    double d = sqrt( ( ((ni + nk)  * dik * dik) + ((nj + nk) * djk * djk) - (nk * dij * dij) )/ ntotal );
    return d;
  }

  inline double updateDistN(double dikl, double dikr, double djkl, double djkr, 
                       double ni, double nj, double nkl, double nkr,
                       double dij, double dklr ){
    double dik = updateDistO(dikl, dikr, nkl, nkr, ni, dklr);
    double djk = updateDistO(djkl, djkr, nkl, nkr, nj, dklr);
    return updateDistO(dik, djk, ni, nj, (nkl + nkr), dij);
  }

  inline double getDistNaive(Node<dim> *inode,  Node<dim> *jnode, 
                          double lb = -1, double ub = numeric_limits<double>::max(), 
                          bool par = true){ 
    double ni = inode->n; 
    double nj = jnode->n;
    if(ni + nj > 2) return sqrt(2*(ni*nj)*inode->dist(jnode)/(ni + nj));
    return sqrt(inode->dist(jnode));
  }

    // range query radius
    inline double getBall(Node<dim>* query, double beta){
        double n = query->size();
        // return beta*sqrt((n+1)/2.0/n);
        return beta*sqrt((n+min_n)/2.0/min_n/n);
    }

    template<class F>
    inline void update(int round, F *finder){
        min_n = *(parlay::min_element(parlay::delayed_seq<double>(finder->C, 
            [&](size_t i){
                int cid = finder->activeClusters[i];
                return finder->getNode(cid)->size();
            })));
        // if(marker.doMark(finder->C, round)){ 
        //     HACTree::singletree<kdnodeT, M, typename M::infoT>(finder->kdtree, &marker, marker.initVal);
        // }
    }
};

template<int dim>
struct distAverage {
  typedef iPoint<dim> pointT;
  typedef Node<dim> nodeT;

  Method method = AVG;
  bool squared = false;

  pointT *clusteredPts1;
  pointT *clusteredPts2;//clusteredPts point to one of the two
  parlay::sequence<int> clusterOffsets;  //same order as activeClusters in finder
  pointT *clusteredPts;

  distAverage(iPoint<dim> *PP, int n){
    clusteredPts1 = (pointT *) malloc(n* sizeof(pointT));
    clusteredPts2 = (pointT *) malloc(n* sizeof(pointT));
    clusteredPts = clusteredPts1;
    clusterOffsets = parlay::sequence<int>(n);
    parlay::parallel_for(0,n,[&](int i){
      clusteredPts1[i]=iPoint<dim>(PP[i], i);
      clusterOffsets[i] = i+1; // the result of scan([1,1,1,1...])
    });
  }

  ~distAverage(){
      free(clusteredPts1);
      free(clusteredPts2);
  }

  inline static void printName(){
    cout << "AVG" << endl;
  }
  inline static bool doRebuild(){return true;}
  inline double updateDistO(double d1, double d2, double nql, double nqr, double nr, double dij){
    double n1 = (double)nql * (double)nr;
    double n2 = (double)nqr * (double)nr;
    double alln = n1 + n2 ;
    d1 = n1 / alln * d1;
    d2 = n2 / alln * d2;
    return d1 + d2;
  }

  inline double updateDistN(double d1, double d2, double d3, double d4, 
                       double nql, double nqr, double nrl, double nrr,
                       double dij, double dklr){
    double n1 = (double)nql * (double)nrl;
    double n2 = (double)nql * (double)nrr;
    double n3 = (double)nqr * (double)nrl;
    double n4 = (double)nqr * (double)nrr;
    double alln = n1 + n2 + n3 + n4;
    d1 = n1 / alln * d1;
    d2 = n2 / alln * d2;
    d3 = n3 / alln * d3;
    d4 = n4 / alln * d4;
    return d1  + d2  + d3 + d4;
  }

    double getDistNaive(Node<dim> *inode,  Node<dim> *jnode, 
                          double lb = -1, double ub = numeric_limits<double>::max(), 
                          bool par = true){ 
    pair<pair<int, int>, double> result = bruteForceAverage(inode, jnode, clusteredPts);
    return result.second;
    }


    // range query radius
    inline double getBall(Node<dim>* query, double beta){
        return beta;
    }

    // copy points from oldArray[oldOffset:oldOffset+n] to newArray[newOffset:newOffset+n]
    inline void copyPoints(pointT *oldArray, pointT *newArray, int copyn, int oldOffset, int newOffset){
        parlay::parallel_for(0, copyn,[&](int j){
          newArray[newOffset + j] = oldArray[oldOffset+j];
      });
    }

    template<class F>
    inline void update(int round, F *finder){
    int *activeClusters = finder->activeClusters.data();
    int  C = finder->C;
    //  put points into clusters
    parlay::parallel_for(0, C, [&](int i){
      clusterOffsets[i] = finder->getNode(activeClusters[i])->size();
    });
    parlay::scan_inclusive_inplace(clusterOffsets.cut(0,C));
    // sequence::prefixSum(clusterOffsets, 0, C);
    pointT *oldArray = clusteredPts;
    pointT *newArray = clusteredPts1;
    if(clusteredPts == clusteredPts1){
      newArray  = clusteredPts2;
    }

    parlay::parallel_for(0,C,[&](int i){
      int cid  = activeClusters[i];
      nodeT *clusterNode = finder->getNode(cid);
      int oldOffset = clusterNode->getOffset(); // must save this before setting new
      int newOffset = 0;
      if(i>0) newOffset = clusterOffsets[i-1];
      clusterNode->setOffset(newOffset);
      if(clusterNode->round == round){//merged this round, copy left and right
        nodeT *clusterNodeL = clusterNode->left;
        oldOffset = clusterNodeL->getOffset();
        copyPoints(oldArray, newArray, clusterNodeL->n, oldOffset, newOffset);
        newOffset += clusterNodeL->n;

        nodeT *clusterNodeR = clusterNode->right;
        oldOffset = clusterNodeR->getOffset();
        copyPoints(oldArray, newArray, clusterNodeR->n, oldOffset, newOffset);

      }else{//not merged this round, just copy
        copyPoints(oldArray, newArray, clusterNode->n, oldOffset, newOffset);
      } 
    });
    clusteredPts = newArray;
    }
};


template<int dim>
struct distAverageSq {
  Method method = AVGSQ;
  bool squared = true;

  inline static void printName(){cout << "AVGSQ" << endl;}

  distAverageSq(){};

  inline static bool doRebuild(){return true;}

  inline double updateDistO(double d1, double d2, double nql, double nqr, double nr, double dij){
    double n1 = (double)nql * (double)nr;
    double n2 = (double)nqr * (double)nr;
    double alln = n1 + n2 ;
    d1 = n1 / alln * d1;
    d2 = n2 / alln * d2;
    return d1 + d2;
  }

  inline double updateDistN(double d1, double d2, double d3, double d4, 
                       double nql, double nqr, double nrl, double nrr,
                       double dij, double dklr){
    double n1 = (double)nql * (double)nrl;
    double n2 = (double)nql * (double)nrr;
    double n3 = (double)nqr * (double)nrl;
    double n4 = (double)nqr * (double)nrr;
    double alln = n1 + n2 + n3 + n4;
    d1 = n1 / alln * d1;
    d2 = n2 / alln * d2;
    d3 = n3 / alln * d3;
    d4 = n4 / alln * d4;
    return d1  + d2  + d3 + d4;
  }

    inline double getDistNaive(Node<dim> *inode,  Node<dim> *jnode, 
                            double lb = -1, double ub = numeric_limits<double>::max(), 
                            bool par = true){ 
    return inode->center.pointDistSq(jnode->center) + inode->var + jnode->var;
    }

    // range query radius
    inline double getBall(Node<dim>* query, double beta){
        return sqrt(beta);
    }

    template<class F>
    inline void update(int round, F *finder){}
};
}