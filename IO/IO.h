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

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
// #include "pargeo/point.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/io.h"

namespace IO {

  auto is_newline = [] (char c) {
    switch (c)  {
    case '\r': 
    case '\n': return true;
    default : return false;
    }
  };

  auto is_delim = [] (char c) {
    switch (c)  {
    case '\t':
    case ';':
    case ',':
    case ' ' : return true;
    default : return false;
    }
  };

  auto is_space = [] (char c) {
    return is_newline(c) || is_delim(c) || c==0;
  };

  auto is_number = [] (char c) {
    switch (c)  {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '.':
    case '+':
    case '-':
    case 'e' : return true;
    default : return false;
    }
  };

  // parallel code for converting a string to word pointers
  // side effects string by setting to null after each word
  template <class Seq>
    parlay::sequence<char*> stringToWords(Seq &Str) {
    size_t n = Str.size();
    
    parlay::parallel_for(0, n, [&] (long i) {
	if (is_space(Str[i])) Str[i] = 0;}); 

    // mark start of words
    auto FL = parlay::tabulate(n, [&] (long i) -> bool {
	return (i==0) ? Str[0] : Str[i] && !Str[i-1];});
    
    // offset for each start of word
    auto Offsets = parlay::pack_index<long>(FL);

    // pointer to each start of word
    auto SA = parlay::tabulate(Offsets.size(), [&] (long j) -> char* {
	return Str.begin() + Offsets[j];});
    
    return SA;
  }

  inline int xToStringLen(parlay::sequence<char> const &a) { return a.size();}
  inline void xToString(char* s, parlay::sequence<char> const &a) {
    for (int i=0; i < a.size(); i++) s[i] = a[i];}

  inline int xToStringLen(long a) { return 21;}
  inline void xToString(char* s, long a) { sprintf(s,"%ld",a);}

  inline int xToStringLen(unsigned long a) { return 21;}
  inline void xToString(char* s, unsigned long a) { sprintf(s,"%lu",a);}

  inline uint xToStringLen(uint a) { return 12;}
  inline void xToString(char* s, uint a) { sprintf(s,"%u",a);}

  inline int xToStringLen(int a) { return 12;}
  inline void xToString(char* s, int a) { sprintf(s,"%d",a);}

  inline int xToStringLen(double a) { return 18;}
  inline void xToString(char* s, double a) { sprintf(s,"%.11le", a);}

  inline int xToStringLen(char* a) { return strlen(a)+1;}
  inline void xToString(char* s, char* a) { sprintf(s,"%s",a);}

  template <class A, class B>
  inline int xToStringLen(std::pair<A,B> a) { 
    return xToStringLen(a.first) + xToStringLen(a.second) + 1;
  }

  template <class A, class B>
  inline void xToString(char* s, std::pair<A,B> a) {
    int l = xToStringLen(a.first);
    xToString(s, a.first);
    s[l] = ' ';
    xToString(s+l+1, a.second);
  }

//   template<int dim>
//   inline int xToStringLen(point<dim> a) {
//     int s = 0;
//     for (int i=0; i<dim; ++i) s += xToStringLen(a[i]);
//     return s+dim-1;
//   }

//   template<int dim>
//   inline void xToString(char* s, point<dim> a, bool comma=false) {
//     char* ss = s;
//     for (int i=0; i<dim; ++i) {
//       int li = xToStringLen(a[i]);
//       xToString(ss, a[i]);
//       if (i != dim-1) {
// 	if(comma) ss[li] = ',';
// 	else ss[li] = ' ';
// 	ss += li+1;
//       }
//     }
//   }

  template <class Seq>
  parlay::sequence<char> seqToString(Seq const &A) {
    size_t n = A.size();
    auto L = parlay::tabulate(n, [&] (size_t i) -> long {
	typename Seq::value_type x = A[i];
	return xToStringLen(x)+1;});
    size_t m;
    std::tie(L,m) = parlay::scan(std::move(L));

    parlay::sequence<char> B(m+1, (char) 0);
    char* Bs = B.begin();

    parlay::parallel_for(0, n-1, [&] (long i) {
      xToString(Bs + L[i], A[i]);
      Bs[L[i+1] - 1] = '\n';
      });
    xToString(Bs + L[n-1], A[n-1]);
    Bs[m] = Bs[m-1] = '\n';

    parlay::sequence<char> C = parlay::filter(B, [&] (char c) {return c != 0;}); 
    C[C.size()-1] = 0;
    return C;
  }

  template <class T>
  void writeSeqToStream(std::ofstream& os, parlay::sequence<T> const &A) {
    size_t bsize = 10000000;
    size_t offset = 0;
    size_t n = A.size();
    while (offset < n) {
      // Generates a string for a sequence of size at most bsize
      // and then wrties it to the output stream
      parlay::sequence<char> S = seqToString(A.cut(offset, std::min(offset + bsize, n)));
      os.write(S.begin(), S.size()-1);
      offset += bsize;
    }
  }

  template <class T>
  int writeSeqToFile(std::string header,
		     parlay::sequence<T> const &A,
		     char const *fileName) {
    auto a = A[0];

    std::ofstream file (fileName, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open file");
    }
    if (header.size() > 0) file << header << std::endl;
    writeSeqToStream(file, A);
    file.close();
    return 0;
  }

  template <class T1, class T2>
  int write2SeqToFile(std::string header,
		      parlay::sequence<T1> const &A,
		      parlay::sequence<T2> const &B,
		      char const *fileName) {
    std::ofstream file (fileName, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open file");
    }
    file << header << std::endl;
    writeSeqToStream(file, A);
    writeSeqToStream(file, B);
    file.close();
    return 0;
  }

  parlay::sequence<char> readStringFromFile(char const *fileName) {
    std::ifstream file (fileName, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open file");
    }
    long end = file.tellg();
    file.seekg (0, std::ios::beg);
    long n = end - file.tellg();
    parlay::sequence<char> bytes(n, (char) 0);
    file.read (bytes.begin(), n);
    file.close();
    return bytes;
  }

  std::string intHeaderIO = "sequenceInt";

  template <class T>
  int writeIntSeqToFile(parlay::sequence<T> const &A, char const *fileName) {
    return writeSeqToFile(intHeaderIO, A, fileName);
  }

  parlay::sequence<parlay::sequence<char>> get_tokens(char const *fileName) {
    // parlay::internal::timer t("get_tokens");
    // auto S = parlay::chars_from_file(fileName);
    auto S = parlay::file_map(fileName);
    // t.next("file map");
    auto r =  parlay::tokens(S, is_space);
    // t.next("tokens");
    return r;
  }

  template <class T>
  parlay::sequence<T> readIntSeqFromFile(char const *fileName) {
    auto W = get_tokens(fileName);
    std::string header(W[0].begin(),W[0].end());
    if (header != intHeaderIO) {
      throw std::runtime_error("readIntSeqFromFile: bad input");
    }
    long n = W.size()-1;
    auto A = parlay::tabulate(n, [&] (long i) -> T {
	return parlay::chars_to_long(W[i+1]);});
    return A;
  }
} // End namespace IO