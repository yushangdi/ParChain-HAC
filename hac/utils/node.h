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

#include "point.h"
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "utils.h"

using namespace std;


namespace HACTree {

#define LARGER_THAN_UB numeric_limits<double>::max()

enum Method { WARD, COMP, AVG, AVGSQ };

// Node data structure, used for dendrogram tree, each node represents a intermediate cluster
// TODO: changed size_t to int, but need to check overflow for multication computation
template<int dim, bool keep_var = true>
struct Node{
    typedef point<dim> pointT;
    int cId; // cluster id corresponds to the union find id
    int round = 0; // merged at which round. if 0, a sigleton cluster
    int idx; // node idx, unique for each node
    double n = 1; // use double because need to do floating point computation using n
    double height = 0;
    Node<dim> *left = nullptr;
    Node<dim> *right = nullptr;
    point<dim> pMin, pMax;
    int offset = -1;  // the offset of points in this cluster in a contiguous array, used in average linkage
    point<dim> center;
    double var = 0; // variance

    //get the node from merged left and right nodes
    // if height=0, assume the left and right are identical
    Node(int t_cid, int t_round, int t_idx, Node<dim> *t_left, Node<dim> *t_right, double _height):
        cId(t_cid),
        round(t_round),
        idx(t_idx),
        height(_height),
        left(t_left),
        right(t_right){

            double left_n = left->size();
            double right_n = right->size();
            pointT left_mu = left->center;
            pointT right_mu = right->center;
            
            n = left_n + right_n;
            //compute bounding box and center
            pMin = left->pMin;
            pMax = left->pMax;
            pMin.minCoords(right->pMin);
            pMax.maxCoords(right->pMax);
            if(height ==0){
                center = t_left->center; //important to avoid numerical error
            }else{
                for (int i=0; i<dim; ++i) {
                    center.updateX(i, 
                                    (t_left->n    / n * left_mu[i] ) + 
                                    (t_right->n   / n * right_mu[i]) );
                }
            }
            if(keep_var){
                // update variance
                double new_var = left_n * left_mu.pointDistSq(center) + \
                            right_n * right_mu.pointDistSq(center) + \
                            left_n * left->var + right_n * right->var; 
                var = new_var/(left_n+right_n);
            }

        }
    
    // initialize a singleton node for a point
    Node(std::size_t  t_cid, point<dim> p):
        cId(t_cid),
        idx(t_cid),
        offset(t_cid){
                pMin = p;
                pMax = p;
                center = p;
            }
    Node(){}

    inline bool isLeaf() const {return n == 1;}
    inline double getHeight() const {return height;}
    inline std::size_t getRound() const {return round;}
    inline std::size_t getIdx() const {return idx;}
    inline std::size_t size() const {return n;}
    inline int getOffset(){return offset;}
    inline void setOffset(int i){offset = i;}
    inline double dist(Node<dim> *node){return center.pointDistSq(node->center);}
};

// used for dendrogram sorting 
template<int dim>
struct nodeComparator{
  double eps;
  nodeComparator(double _eps):eps(_eps) {
  }

  bool operator() (const Node<dim> i, const Node<dim> j) const {
      if(abs(i.getHeight() - j.getHeight()) <= eps) return i.getRound() < j.getRound();
     return i.getHeight() < j.getHeight();
  }
};

// node info used for kd-tree
struct nodeInfo{
    typedef std::tuple<int, int> infoT; // <cid, min_n>
    int cId = -1;
    // std::atomic<double> ub; // need to write an = assign method if use atomic
    double ub = numeric_limits<double>::max();
    // int round = 0;
    // int idx = -1; // node idx
    int min_n = 1; //used for ward's linkage

    nodeInfo(){
        // cId = -1;
        // ub.store(numeric_limits<double>::max()); // used for allptsnn
        ub = numeric_limits<double>::max();
    }

    inline double getUB(){return ub;}//{return ub.load();}
    inline void updateUB(double tmp){
        // parlay::write_min(&ub, tmp, std::less<double>());
        utils::writeMin(&ub, tmp);
    }
    inline void resetUB(){
        // ub.store(numeric_limits<double>::max());
        ub = numeric_limits<double>::max();
    }
    inline int getCId(){return cId;}
    inline void setCId(int id){cId = id;}
    // inline int getRound(){return round;}
    // inline void setRound(int r){round = r;}
    // inline int getIdx(){return idx;}
    // inline void setIdx(int r){idx = r;}
    inline infoT initInfoVal() {return make_tuple(-1, numeric_limits<int>::max());}
    inline void setInfo(infoT info){
      tie(cId, min_n) = info;
    }
    inline double getMinN(){return (double)min_n;} //return double because it's used in floating point computation
    inline void setMinN(int nn){min_n = nn;}

};

}