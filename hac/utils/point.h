#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <limits>

#include "parlay/slice.h"

using namespace std;

namespace HACTree{

// from pbbs
template <int _dim> class point;

template <int _dim> class point {
public:
  typedef double floatT;
  // typedef vect<_dim> vectT;
  typedef point pointT;
  floatT x[_dim];
  static const int dim = _dim;
  static constexpr double empty = numeric_limits<double>::max();
  int dimension() {return _dim;}
  bool isEmpty() {return x[0]==empty;}
  point() { for (int i=0; i<_dim; ++i) x[i]=empty; }
  // point(vectT v) { for (int i=0; i<_dim; ++i) x[i]=v[i]; }
  point(floatT* p) { for (int i=0; i<_dim; ++i) x[i]=p[i]; }
  point(pointT* p) { for (int i=0; i<_dim; ++i) x[i]=p->x[i]; }
  
  template<class _tIn>
  point(parlay::slice<_tIn*,_tIn*> p) {
      for(int i=0; i<_dim; ++i) x[i] = (floatT)p[i];}
  floatT& operator[](int i) {return x[i];}
  void updateX(int i, floatT val) {x[i]=val;}
  floatT at(int i) {return x[i];}
  void minCoords(pointT b) {
    for (int i=0; i<_dim; ++i) x[i] = min(x[i], b.x[i]);}
  void minCoords(floatT* b) {
    for (int i=0; i<_dim; ++i) x[i] = min(x[i], b[i]);}
  void maxCoords(pointT b) {
    for (int i=0; i<_dim; ++i) x[i] = max(x[i], b.x[i]);}
  void maxCoords(floatT* b) {
    for (int i=0; i<_dim; ++i) x[i] = max(x[i], b[i]);}

   pointT mult(floatT c) {
      pointT r;
      for(int i=0; i<dim; ++i) r[i] = x[i]*c;
      return r;}

   pointT operator*(floatT c) {
      pointT r;
      for(int i=0; i<dim; ++i) r[i] = x[i]*c;
      return r;}

  floatT* coords() {return x;}  
  bool outOfBox(pointT center, floatT hsize) {
    for (int i=0; i<_dim; ++i) {
      if (x[i]-hsize > center.x[i] || x[i]+hsize < center.x[i])
        return true;
    }
    return false;
  }
  inline floatT pointDist(pointT p) {
    floatT xx=0;
    for (int i=0; i<_dim; ++i) xx += (x[i]-p.x[i])*(x[i]-p.x[i]);
    return sqrt(xx);
  }

  floatT dist(pointT p) {
    return pointDist(p);
  }
  
  floatT pointDistSq(point p) {
    floatT xx=0;
    for (int i=0; i<_dim; ++i) xx += (x[i]-p.x[i])*(x[i]-p.x[i]);
    return xx;
  }
};

template<int dim>
struct iPoint : public point<dim> {
  typedef double floatT;
  typedef point<dim> pointT;
  int i;
  iPoint(pointT pp, int ii): point<dim>(pp), i(ii) {}
  iPoint(pointT pp): point<dim>(pp), i(-1) {}
  iPoint(): i(-1) {}
  bool isEmpty() {return i<0;}
  int idx() {return i;}
  void idx(int ii) {i=ii;}
};

template<int dim>
static std::ostream& operator<<(std::ostream& os, const point<dim> v) {
  for (int i=0; i<v.dim; ++i)
    os << std::setprecision(2) << v.x[i] << ", ";
  return os;
}

template<int dim>
static std::ostream& operator<<(std::ostream& os, const iPoint<dim> v) {
  for (int i=0; i<v.dim; ++i)
    os << std::setprecision(2) << v.x[i] << ", ";
  return os;
}

template<int dim>
parlay::sequence<iPoint<dim>> makeIPoint(parlay::slice<point<dim> *, point<dim> *> P){
  auto PP = parlay::sequence<iPoint<dim>>(P.size());
  parlay::parallel_for(0,P.size(),[&](int i){PP[i]=iPoint<dim>(P[i], i);});
  return std::move(PP);
}

template<int dim>
parlay::sequence<iPoint<dim>> makeIPoint(point<dim> *P, int n){
  auto PP = parlay::sequence<iPoint<dim>>(n);
  parlay::parallel_for(0,n,[&](int i){PP[i]=iPoint<dim>(P[i], i);});
  return std::move(PP);
}

template<int dim>
parlay::sequence<iPoint<dim>> makeIPoint(parlay::sequence<point<dim>>& P){
  return makeIPoint(P.cut(0,P.size()));
}

}
