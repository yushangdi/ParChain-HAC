// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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
#include "parlay/primitives.h"
#include "IO.h"

  namespace pointIO {

  std::string pbbsHeader(int dim) {
    if (dim < 2 || dim > 9) {
      throw std::runtime_error("Error, unsupported dimension");
    }
    return "pbbs_sequencePoint" + std::to_string(dim) + "d";
  }

  bool isGenericHeader(std::string line) {
    for (auto c: line) {
      if (!IO::is_number(c) && !IO::is_delim(c)) return true;
    }
    return false;
  }

  int countEntry(std::string line) {
    while (IO::is_delim(line.back()) ||
	   IO::is_space(line.back()) ||
	   IO::is_newline(line.back())) {
      line.pop_back();
    }

    int count = 0;
    for (auto c: line) {
      if (IO::is_delim(c)) count ++;
    }
    return count + 1;
  }

  // returns dim
  int readHeader(const char* fileName) {
    std::ifstream file (fileName);
    if (!file.is_open())
      throw std::runtime_error("Unable to open file");

    std::string line1; std::getline(file, line1);
    if (isGenericHeader(line1)) {
      std::string line2; std::getline(file, line2);
      return countEntry(line2);
    } else {
      return countEntry(line1);
    }
  }

  // todo deprecate
  int readDimensionFromFile(char* const fileName) {
    std::cout << "warning: using deprecated function readDimensionFromFile\n";
    return readHeader(fileName);
  }

  template <class pointT>
  int writePointsToFile(parlay::sequence<pointT> const &P, char const *fname) {
    int r = IO::writeSeqToFile("", P, fname);
    return r;
  }

  template <class pointT>
 int writePointsToFilePbbs(parlay::sequence<pointT> const &P, char const *fname) {
    std::string Header = pbbsHeader(pointT::dim);
    int r = IO::writeSeqToFile(Header, P, fname);
    return r;
  }

  template <class pointT, class Seq>
  parlay::sequence<pointT> parsePoints(Seq W) {
    using coord = double;
    int d = pointT::dim;
    size_t n = W.size()/d;
    auto a = parlay::tabulate(d * n, [&] (size_t i) -> coord {
				       return atof(W[i]);});
    parlay::sequence<pointT> points = parlay::tabulate(n, [&] (size_t i) -> pointT {
					return pointT(a.cut(d*i,d*(i + 1)));});
    return points;
  }

  template <class pointT>
  parlay::sequence<pointT> readPointsFromFile(char const *fname) {
    parlay::sequence<char> S = IO::readStringFromFile(fname);
    parlay::sequence<char*> W = IO::stringToWords(S);
    int d = pointT::dim;
    if (W.size() == 0)
      throw std::runtime_error("readPointsFromFile empty file");

    if (isGenericHeader(W[0]))
      return parsePoints<pointT>(W.cut(1,W.size()));
    else
      return parsePoints<pointT>(W.cut(0,W.size()));
  }

} // End namespace pointIO
