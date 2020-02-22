// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Michael Rabinovich
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "slim.h"

#include "boundary_loop.h"
#include "cotmatrix.h"
#include "edge_lengths.h"
#include "grad.h"
#include "local_basis.h"
#include "repdiag.h"
#include "vector_area_matrix.h"
#include "arap.h"
#include "cat.h"
#include "doublearea.h"
#include "grad.h"
#include "local_basis.h"
#include "per_face_normals.h"
#include "slice_into.h"
#include "volume.h"
#include "polar_svd.h"
#include "flip_avoiding_line_search.h"
#include "mapping_energy_with_jacobians.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include "Timer.h"
#include "sparse_cached.h"
#include "AtA_cached.h"

#ifdef CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#undef SLIM_CACHED

#define USE_MKL_SOLVER
#define USE_NUMERIC

// #ifdef SLIM_CACHED
// #undef USE_NUMERIC
// #endif

#include "inline_expansion/utils.hpp"

#include "ie_helper.hpp"

namespace igl
{
namespace slim
{
// Definitions of internal functions
IGL_INLINE void buildRhs(igl::SLIMData &s, const Eigen::SparseMatrix<double> &A);
IGL_INLINE void add_soft_constraints(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
IGL_INLINE void add_soft_constraints_numeric_together(igl::SLIMData &s);
IGL_INLINE double compute_energy(igl::SLIMData &s, Eigen::MatrixXd &V_new);
IGL_INLINE double compute_soft_const_energy(igl::SLIMData &s,
                                            const Eigen::MatrixXd &V,
                                            const Eigen::MatrixXi &F,
                                            Eigen::MatrixXd &V_o);

IGL_INLINE void solve_weighted_arap(igl::SLIMData &s,
                                    const Eigen::MatrixXd &V,
                                    const Eigen::MatrixXi &F,
                                    Eigen::MatrixXd &uv,
                                    Eigen::VectorXi &soft_b_p,
                                    Eigen::MatrixXd &soft_bc_p);
IGL_INLINE void update_weights_and_closest_rotations(igl::SLIMData &s,
                                                     Eigen::MatrixXd &uv);
IGL_INLINE void compute_jacobians(igl::SLIMData &s, const Eigen::MatrixXd &uv);
IGL_INLINE void build_linear_system(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
IGL_INLINE void build_linear_system_eigen(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
IGL_INLINE void build_linear_system_cached(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
IGL_INLINE void build_linear_system_numeric_together(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
IGL_INLINE void pre_calc(igl::SLIMData &s);

// Implementation

IGL_INLINE void compute_jacobians(igl::SLIMData &s, const Eigen::MatrixXd &uv)
{
  if (s.F.cols() == 3)
  {
    // Ji=[D1*u,D2*u,D1*v,D2*v];
    s.Ji.col(0) = s.Dx * uv.col(0);
    s.Ji.col(1) = s.Dy * uv.col(0);
    s.Ji.col(2) = s.Dx * uv.col(1);
    s.Ji.col(3) = s.Dy * uv.col(1);
  }
  else /*tet mesh*/
  {
    // Ji=[D1*u,D2*u,D3*u, D1*v,D2*v, D3*v, D1*w,D2*w,D3*w];
    s.Ji.col(0) = s.Dx * uv.col(0);
    s.Ji.col(1) = s.Dy * uv.col(0);
    s.Ji.col(2) = s.Dz * uv.col(0);
    s.Ji.col(3) = s.Dx * uv.col(1);
    s.Ji.col(4) = s.Dy * uv.col(1);
    s.Ji.col(5) = s.Dz * uv.col(1);
    s.Ji.col(6) = s.Dx * uv.col(2);
    s.Ji.col(7) = s.Dy * uv.col(2);
    s.Ji.col(8) = s.Dz * uv.col(2);
  }
}

IGL_INLINE void update_weights_and_closest_rotations(igl::SLIMData &s, Eigen::MatrixXd &uv)
{
  compute_jacobians(s, uv);
  slim_update_weights_and_closest_rotations_with_jacobians(s.Ji, s.slim_energy, s.exp_factor, s.W, s.Ri);
}

IGL_INLINE void solve_weighted_arap(igl::SLIMData &s,
                                    const Eigen::MatrixXd &V,
                                    const Eigen::MatrixXi &F,
                                    Eigen::MatrixXd &uv,
                                    Eigen::VectorXi &soft_b_p,
                                    Eigen::MatrixXd &soft_bc_p)
{
  using namespace Eigen;

  igl::Timer t;
  Eigen::SparseMatrix<double> L;
  t.start();
  build_linear_system_numeric_together(s, L);
  std::vector<double> tmp = s.result_vector;
  // build_linear_system_numeric_together(s, L);
  // for (int i=0; i<tmp.size(); i++){
  //   std::cout<<tmp[i]-s.result_vector[i]<<"\n";
  // }
  t.stop();
  std::cout << "Total time building linear system took " << t.getElapsedTimeInMicroSec() << " microseconds";

  //t.start();
  // solve
  Eigen::VectorXd Uc;
#ifndef USE_NUMERIC
  std::vector<double> x(L.rows());
#else
  std::vector<double> x(s.n_rows);
#endif
#ifndef CHOLMOD
  if (s.dim == 2)
  {
    igl::Timer t;
#ifndef USE_MKL_SOLVER
    SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    t.start();
    Uc = solver.compute(L).solve(s.rhs);
    t.stop();
    std::cout << "\n"
              << "Eigen solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
#else
#ifndef USE_NUMERIC
    Eigen::SparseMatrix<double, Eigen::ColMajor> L_upper = L.triangularView<Lower>();
#else
#endif
    if (s.first_called)
    {
      t.start();
      pardiso_init(s.pardiso_data);
#ifndef USE_NUMERIC
      pardiso_support_matrix(s.pardiso_data, L_upper);
#else
      pardiso_support_matrix(s.pardiso_data, s.L_outer.data(), s.L_inner.data(), s.result_vector.data(), s.n_rows);
#endif
      t.stop();
      std::cout << "\n"
                << "Init everything for the mkl solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
      t.start();
      pardiso_symbolic_factor(s.pardiso_data);
      t.stop();
      std::cout << "\n"
                << "Symbolic factorization took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
      s.first_called = false;
    }
    else
    {
#ifndef USE_NUMERIC
      pardiso_support_value(s.pardiso_data, L_upper.valuePtr());
#else
      pardiso_support_value(s.pardiso_data, s.result_vector.data());
#endif
    }
    t.start();
    pardiso_numeric_factor(s.pardiso_data);
    t.stop();
    std::cout << "\n"
              << "Numeric factorization took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
    std::vector<double> b(s.rhs.data(), s.rhs.data() + s.rhs.rows() * s.rhs.cols());
    t.start();
    pardiso_solve(s.pardiso_data, x.data(), b.data());
    t.stop();
    std::cout << "MKL Pardiso solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n\n";
#endif
  }
  else
  {
    igl::Timer t;
#ifndef USE_MKL_SOLVER
    // seems like CG performs much worse for 2D and way better for 3D
    Eigen::VectorXd guess(uv.rows() * s.dim);
    for (int i = 0; i < s.v_num; i++)
      for (int j = 0; j < s.dim; j++)
        guess(uv.rows() * j + i) = uv(i, j); // flatten vector

    ConjugateGradient<Eigen::SparseMatrix<double>, Lower | Upper> cg;
    t.start();
    cg.setTolerance(1e-8);
    cg.compute(L);
    Uc = cg.solveWithGuess(s.rhs, guess);
    t.stop();
    std::cout << "\n"
              << "Eigen ConjugateGradient solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
#else
    t.start();
    Eigen::SparseMatrix<double, Eigen::ColMajor> L_upper = L.triangularView<Lower>();
    t.stop();
    std::cout << "\n"
              << "Getting upper triangular took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
    if (s.first_called)
    {
      t.start();
      pardiso_init(s.pardiso_data);
      pardiso_support_matrix(s.pardiso_data, L_upper);
      t.stop();
      std::cout << "\n"
                << "Init everything for the mkl solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
      t.start();
      pardiso_symbolic_factor(s.pardiso_data);
      t.stop();
      std::cout << "\n"
                << "Symbolic factorization took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
      s.first_called = false;
    }
    else
    {
      pardiso_support_value(s.pardiso_data, L_upper.valuePtr());
    }
    t.start();
    pardiso_numeric_factor(s.pardiso_data);
    t.stop();
    std::cout << "\n"
              << "Numeric factorization took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
    std::vector<double> b(s.rhs.data(), s.rhs.data() + s.rhs.rows() * s.rhs.cols());
    t.start();
    pardiso_solve(s.pardiso_data, x.data(), b.data());
    t.stop();
    std::cout << "MKL Pardiso solver took " << t.getElapsedTimeInMicroSec() << " microseconds\n";
#endif
  }
#else
  CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  Uc = solver.compute(L).solve(s.rhs);
#endif
  for (int i = 0; i < s.dim; i++)
#ifndef USE_MKL_SOLVER
    uv.col(i) = Uc.block(i * s.v_n, 0, s.v_n, 1);
#else
    uv.col(i) = Eigen::Map<Eigen::VectorXd>(x.data() + i * s.v_n, s.v_n);
#endif

  // t.stop();
  // std::cerr << "solve: " << t.getElapsedTime() << std::endl;
}

IGL_INLINE void pre_calc(igl::SLIMData &s)
{
  if (!s.has_pre_calc)
  {
    s.v_n = s.v_num;
    s.f_n = s.f_num;

    if (s.F.cols() == 3)
    {
      s.dim = 2;
      Eigen::MatrixXd F1, F2, F3;
      igl::local_basis(s.V, s.F, F1, F2, F3);
      Eigen::SparseMatrix<double> G;
      igl::grad(s.V, s.F, G);
      Eigen::SparseMatrix<double> Face_Proj;

      auto face_proj = [](Eigen::MatrixXd &F) {
        std::vector<Eigen::Triplet<double>> IJV;
        int f_num = F.rows();
        for (int i = 0; i < F.rows(); i++)
        {
          IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
          IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
          IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
      };

      s.Dx = face_proj(F1) * G;
      s.Dy = face_proj(F2) * G;
    }
    else
    {
      s.dim = 3;
      Eigen::SparseMatrix<double> G;
      igl::grad(s.V, s.F, G,
                s.mesh_improvement_3d /*use normal gradient, or one from a "regular" tet*/);
      s.Dx = G.block(0, 0, s.F.rows(), s.V.rows());
      s.Dy = G.block(s.F.rows(), 0, s.F.rows(), s.V.rows());
      s.Dz = G.block(2 * s.F.rows(), 0, s.F.rows(), s.V.rows());
    }

    s.W.resize(s.f_n, s.dim * s.dim);
    s.Dx.makeCompressed();
    s.Dy.makeCompressed();
    s.Dz.makeCompressed();
    s.Ri.resize(s.f_n, s.dim * s.dim);
    s.Ji.resize(s.f_n, s.dim * s.dim);
    s.rhs.resize(s.dim * s.v_num);

    // flattened weight matrix
    s.WGL_M.resize(s.dim * s.dim * s.f_n);
    for (int i = 0; i < s.dim * s.dim; i++)
      for (int j = 0; j < s.f_n; j++)
        s.WGL_M(i * s.f_n + j) = s.M(j);

    s.first_solve = true;
    s.has_pre_calc = true;
  }
}

IGL_INLINE void build_linear_system_eigen(igl::SLIMData &s, Eigen::SparseMatrix<double> &L)
{
  std::cout << "BUILDING LINEAR SYSTEM USING EIGEN\n";
  std::vector<Eigen::Triplet<double>> IJV;
  igl::Timer t;
  t.start();
  slim_buildA(s.Dx, s.Dy, s.Dz, s.W, IJV);
  t.stop();
  std::cout << "SLIM_BUILDA TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  Eigen::SparseMatrix<double> A(s.dim * s.dim * s.f_n, s.dim * s.v_n);
  A.setFromTriplets(IJV.begin(), IJV.end());
  A.makeCompressed();
  t.stop();
  std::cout << "CONSTRUCTING MATRIX A TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  Eigen::SparseMatrix<double> At = A.transpose();
  At.makeCompressed();
  t.stop();
  std::cout << "TAKING A TRANSPOSE TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  Eigen::SparseMatrix<double> id_m(A.cols(), A.cols());
  id_m.setIdentity();
  t.stop();
  std::cout << "SETTING AN IDENTITY MATRIX TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  L = At * s.WGL_M.asDiagonal() * A + s.proximal_p * id_m; //add also a proximal term
  L.makeCompressed();
  t.stop();
  std::cout << "EIGEN COMPUTING L WITHOUT SOFT CONSTRAINTS " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  buildRhs(s, A);
  t.stop();
  std::cout << "BUILDING RHS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  add_soft_constraints(s, L);
  t.stop();
  std::cout << "ADDING SOFT CONSTRAINTS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
}

IGL_INLINE void build_linear_system_cached(igl::SLIMData &s, Eigen::SparseMatrix<double> &L)
{
  std::cout << "BUILDING LINEAR SYSTEM USING DANIELE'S METHOD\n";
  std::vector<Eigen::Triplet<double>> IJV;
  igl::Timer t;
  t.start();
  slim_buildA(s.Dx, s.Dy, s.Dz, s.W, IJV);
  t.stop();
  std::cout << "SLIM_BUILDA TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  if (s.A.rows() == 0)
  {
    s.A = Eigen::SparseMatrix<double>(s.dim * s.dim * s.f_n, s.dim * s.v_n);
    igl::sparse_cached_precompute(IJV, s.A_data, s.A);
    export_mtx(s.A, "small_face.mtx");
  }
  else
    igl::sparse_cached(IJV, s.A_data, s.A);
  t.stop();
  std::cout << "CONSTRUCTING MATRIX A TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  Eigen::SparseMatrix<double> id_m(s.A.cols(), s.A.cols());
  id_m.setIdentity();
  t.stop();
  std::cout << "SETTING AN IDENTITY MATRIX TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  s.AtA_data.W = s.WGL_M;
  if (s.AtA.rows() == 0)
  {
    igl::AtA_cached_precompute(s.A, s.AtA_data, s.AtA);
  }
  else
  {
    igl::AtA_cached(s.A, s.AtA_data, s.AtA);
  }
  t.stop();
  std::cout << "COMPUTING ATDA TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  L = s.AtA + s.proximal_p * id_m; //add also a proximal
  L.makeCompressed();
  t.stop();
  std::cout << "ADDING PROXIMAL TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  buildRhs(s, s.A);
  t.stop();
  std::cout << "BUILDING RHS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  add_soft_constraints(s, L);
  t.stop();
  std::cout << "ADDING SOFT CONSTRAINTS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
}

IGL_INLINE void build_linear_system_numeric_together(igl::SLIMData &s, Eigen::SparseMatrix<double> &L)
{
  std::cout << "BUILDING LINEAR SYSTEM USING NUMERIC DOING ALL COMPUTATIONS TOGETHER\n";
  std::vector<Eigen::Triplet<double>> IJV;
  igl::Timer t;
  t.start();
  slim_buildA(s.Dx, s.Dy, s.Dz, s.W, IJV);
  t.stop();
  std::cout << "SLIM_BUILDA TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  if (s.A.rows() == 0)
  {
    s.A = Eigen::SparseMatrix<double>(s.dim * s.dim * s.f_n, s.dim * s.v_n);
    igl::sparse_cached_precompute(IJV, s.A_data, s.A);
    // export_mtx(s.A, "small_face.mtx");
  }
  else
    igl::sparse_cached(IJV, s.A_data, s.A);
  t.stop();
  std::cout << "CONSTRUCTING MATRIX A TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  buildRhs(s, s.A);
  t.stop();
  std::cout << "BUILDING RHS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  add_soft_constraints_numeric_together(s);
  t.stop();
  std::cout << "GETTING SOFT CONSTRAINTS VALUE " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  if (s.first_called)
  {
    s.n_rows = s.A.cols();
    // build A in numeric type
    Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor> A_numeric = ie::to_sparse_numeric<double, Eigen::ColMajor>(s.A, 0);
    s.datas[0].resize(s.A.nonZeros());
    s.datas[2].resize(s.WGL_M.rows() * s.WGL_M.cols());

    // build the soft constraints
    Eigen::SparseMatrix<double, Eigen::ColMajor> soft_constraints(s.A.cols(), s.A.cols());
    soft_constraints.setFromTriplets(s.soft_constraints_triplet.begin(), s.soft_constraints_triplet.end());
    Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor> soft_constraints_numeric = ie::to_sparse_numeric<double, Eigen::ColMajor>(soft_constraints, 1);

    // build the two diagonal matrices
    std::vector<Eigen::Triplet<ie::NumericType>> diagonals;
    diagonals.reserve(s.A.cols());
    ie::NumericType proximal = ie::NumericType(s.proximal_p);
    Eigen::SparseMatrix<double, Eigen::ColMajor> wgl_m(s.A.rows(), s.A.rows());
    std::vector<Eigen::Triplet<double>> wgl_m_trip;
    wgl_m_trip.reserve(s.A.rows());
    for (int i = 0; i < s.A.cols(); i++)
    {
      diagonals.push_back(Eigen::Triplet<ie::NumericType>(i, i, proximal));
    }
    for (int i = 0; i < s.A.rows(); i++)
    {
      wgl_m_trip.push_back(Eigen::Triplet<double>(i, i, 2.0));
    }
    Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor> diagonal_numeric(s.A.cols(), s.A.cols());
    diagonal_numeric.setFromTriplets(diagonals.begin(), diagonals.end());
    wgl_m.setFromTriplets(wgl_m_trip.begin(), wgl_m_trip.end());
    Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor> wgl_m_numeric = ie::to_sparse_numeric<double, Eigen::ColMajor>(wgl_m, 2);

    // the actual structure initialization
    Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor> result_numeric = Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor>(A_numeric.transpose()) * wgl_m_numeric * A_numeric + diagonal_numeric + soft_constraints_numeric;

    result_numeric = result_numeric.triangularView<Eigen::Lower>();
    s.ex = ie::NumericExecutor(result_numeric, 0);
    s.result_vector.resize(result_numeric.nonZeros(), 0);

    // copy the outer index pointer and inner index pointer to s
    s.L_outer.resize(result_numeric.rows() + 1);
    s.L_inner.resize(result_numeric.nonZeros());
    s.L_outer.assign(result_numeric.outerIndexPtr(), result_numeric.outerIndexPtr() + result_numeric.rows() + 1);
    s.L_inner.assign(result_numeric.innerIndexPtr(), result_numeric.innerIndexPtr() + result_numeric.nonZeros());
    s.soft_constraints_triplet.resize(0);
    result_numeric.resize(0, 0);
  }
  t.start();
  s.datas[0].assign(s.A.valuePtr(), s.A.valuePtr() + s.A.nonZeros());
  s.datas[2].assign(s.WGL_M.data(), s.WGL_M.data() + s.WGL_M.rows() * s.WGL_M.cols());
  t.stop();
  std::cout << "ASSIGNING DATAS TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
  t.start();
  s.ex.ExecuteMulti(s.datas, s.result_vector);
  t.stop();
  std::cout << "NUMERIC COMPUTING EVERYTHING TOGETHER TOOK " << t.getElapsedTimeInMicroSec() << " microseconds\n";
}

IGL_INLINE void add_soft_constraints(igl::SLIMData &s, Eigen::SparseMatrix<double> &L)
{
  int v_n = s.v_num;
  for (int d = 0; d < s.dim; d++)
  {
    for (int i = 0; i < s.b.rows(); i++)
    {
      int v_idx = s.b(i);
      s.rhs(d * v_n + v_idx) += s.soft_const_p * s.bc(i, d);          // rhs
      L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.soft_const_p; // diagonal of matrix
    }
  }
}

IGL_INLINE void add_soft_constraints_numeric_together(igl::SLIMData &s)
{
  int v_n = s.v_num;
  int pos = 0;
  if (s.first_called)
  {
    s.datas.resize(3);
    std::vector<std::pair<int, int>> positions;
    positions.reserve(s.dim * s.b.rows());
    s.soft_constraints_triplet.reserve(s.dim * s.b.rows());
    s.datas[1].resize(s.dim * s.b.rows());

    for (int d = 0; d < s.dim; d++)
    {
      for (int i = 0; i < s.b.rows(); i++)
      {
        int v_idx = s.b(i);
        positions.push_back({d * v_n + v_idx, pos});
        s.soft_constraints_triplet.push_back(Eigen::Triplet<double>(d * v_n + v_idx, d * v_n + v_idx, 2.0));
        pos++;
      }
    }
    std::sort(positions.begin(), positions.end()); // sort the positions, because we want it to be the value array in csr format
    s.soft_constraints_pos.resize(positions.size());
    for (int i = 0; i < positions.size(); i++)
    {
      s.soft_constraints_pos[positions[i].second] = i;
    }
  }
  // now rearrange the values
  pos = 0;
  for (int d = 0; d < s.dim; d++)
  {
    for (int i = 0; i < s.b.rows(); i++)
    {
      s.datas[1][s.soft_constraints_pos[pos]] = s.soft_const_p;
      int v_idx = s.b(i);
      s.rhs(d * v_n + v_idx) += s.soft_const_p * s.bc(i, d); // rhs
      pos++;
    }
  }
}

IGL_INLINE double compute_energy(igl::SLIMData &s, Eigen::MatrixXd &V_new)
{
  compute_jacobians(s, V_new);
  return mapping_energy_with_jacobians(s.Ji, s.M, s.slim_energy, s.exp_factor) +
         compute_soft_const_energy(s, s.V, s.F, V_new);
}

IGL_INLINE double compute_soft_const_energy(igl::SLIMData &s,
                                            const Eigen::MatrixXd &V,
                                            const Eigen::MatrixXi &F,
                                            Eigen::MatrixXd &V_o)
{
  double e = 0;
  for (int i = 0; i < s.b.rows(); i++)
  {
    e += s.soft_const_p * (s.bc.row(i) - V_o.row(s.b(i))).squaredNorm();
  }
  return e;
}

IGL_INLINE void buildRhs(igl::SLIMData &s, const Eigen::SparseMatrix<double> &A)
{
  Eigen::VectorXd f_rhs(s.dim * s.dim * s.f_n);
  f_rhs.setZero();
  if (s.dim == 2)
  {
    /*b = [W11*R11 + W12*R21; (formula (36))
             W11*R12 + W12*R22;
             W21*R11 + W22*R21;
             W21*R12 + W22*R22];*/
    for (int i = 0; i < s.f_n; i++)
    {
      f_rhs(i + 0 * s.f_n) = s.W(i, 0) * s.Ri(i, 0) + s.W(i, 1) * s.Ri(i, 1);
      f_rhs(i + 1 * s.f_n) = s.W(i, 0) * s.Ri(i, 2) + s.W(i, 1) * s.Ri(i, 3);
      f_rhs(i + 2 * s.f_n) = s.W(i, 2) * s.Ri(i, 0) + s.W(i, 3) * s.Ri(i, 1);
      f_rhs(i + 3 * s.f_n) = s.W(i, 2) * s.Ri(i, 2) + s.W(i, 3) * s.Ri(i, 3);
    }
  }
  else
  {
    /*b = [W11*R11 + W12*R21 + W13*R31;
             W11*R12 + W12*R22 + W13*R32;
             W11*R13 + W12*R23 + W13*R33;
             W21*R11 + W22*R21 + W23*R31;
             W21*R12 + W22*R22 + W23*R32;
             W21*R13 + W22*R23 + W23*R33;
             W31*R11 + W32*R21 + W33*R31;
             W31*R12 + W32*R22 + W33*R32;
             W31*R13 + W32*R23 + W33*R33;];*/
    for (int i = 0; i < s.f_n; i++)
    {
      f_rhs(i + 0 * s.f_n) = s.W(i, 0) * s.Ri(i, 0) + s.W(i, 1) * s.Ri(i, 1) + s.W(i, 2) * s.Ri(i, 2);
      f_rhs(i + 1 * s.f_n) = s.W(i, 0) * s.Ri(i, 3) + s.W(i, 1) * s.Ri(i, 4) + s.W(i, 2) * s.Ri(i, 5);
      f_rhs(i + 2 * s.f_n) = s.W(i, 0) * s.Ri(i, 6) + s.W(i, 1) * s.Ri(i, 7) + s.W(i, 2) * s.Ri(i, 8);
      f_rhs(i + 3 * s.f_n) = s.W(i, 3) * s.Ri(i, 0) + s.W(i, 4) * s.Ri(i, 1) + s.W(i, 5) * s.Ri(i, 2);
      f_rhs(i + 4 * s.f_n) = s.W(i, 3) * s.Ri(i, 3) + s.W(i, 4) * s.Ri(i, 4) + s.W(i, 5) * s.Ri(i, 5);
      f_rhs(i + 5 * s.f_n) = s.W(i, 3) * s.Ri(i, 6) + s.W(i, 4) * s.Ri(i, 7) + s.W(i, 5) * s.Ri(i, 8);
      f_rhs(i + 6 * s.f_n) = s.W(i, 6) * s.Ri(i, 0) + s.W(i, 7) * s.Ri(i, 1) + s.W(i, 8) * s.Ri(i, 2);
      f_rhs(i + 7 * s.f_n) = s.W(i, 6) * s.Ri(i, 3) + s.W(i, 7) * s.Ri(i, 4) + s.W(i, 8) * s.Ri(i, 5);
      f_rhs(i + 8 * s.f_n) = s.W(i, 6) * s.Ri(i, 6) + s.W(i, 7) * s.Ri(i, 7) + s.W(i, 8) * s.Ri(i, 8);
    }
  }
  Eigen::VectorXd uv_flat(s.dim * s.v_n);
  for (int i = 0; i < s.dim; i++)
    for (int j = 0; j < s.v_n; j++)
      uv_flat(s.v_n * i + j) = s.V_o(j, i);

  s.rhs = (f_rhs.transpose() * s.WGL_M.asDiagonal() * A).transpose() + s.proximal_p * uv_flat;
}

} // namespace slim
} // namespace igl

IGL_INLINE void igl::slim_update_weights_and_closest_rotations_with_jacobians(const Eigen::MatrixXd &Ji,
                                                                              igl::MappingEnergyType slim_energy,
                                                                              double exp_factor,
                                                                              Eigen::MatrixXd &W,
                                                                              Eigen::MatrixXd &Ri)
{
  const double eps = 1e-8;
  double exp_f = exp_factor;
  const int dim = (Ji.cols() == 4 ? 2 : 3);

  if (dim == 2)
  {
    for (int i = 0; i < Ji.rows(); ++i)
    {
      typedef Eigen::Matrix2d Mat2;
      typedef Eigen::Matrix<double, 2, 2, Eigen::RowMajor> RMat2;
      typedef Eigen::Vector2d Vec2;
      Mat2 ji, ri, ti, ui, vi;
      Vec2 sing;
      Vec2 closest_sing_vec;
      RMat2 mat_W;
      Vec2 m_sing_new;
      double s1, s2;

      ji(0, 0) = Ji(i, 0);
      ji(0, 1) = Ji(i, 1);
      ji(1, 0) = Ji(i, 2);
      ji(1, 1) = Ji(i, 3);

      igl::polar_svd(ji, ri, ti, ui, sing, vi);

      s1 = sing(0);
      s2 = sing(1);

      // Update Weights according to energy
      switch (slim_energy)
      {
      case igl::MappingEnergyType::ARAP:
      {
        m_sing_new << 1, 1;
        break;
      }
      case igl::MappingEnergyType::SYMMETRIC_DIRICHLET:
      {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
        break;
      }
      case igl::MappingEnergyType::LOG_ARAP:
      {
        double s1_g = 2 * (log(s1) / s1);
        double s2_g = 2 * (log(s2) / s2);
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
        break;
      }
      case igl::MappingEnergyType::CONFORMAL:
      {
        double s1_g = 1 / (2 * s2) - s2 / (2 * pow(s1, 2));
        double s2_g = 1 / (2 * s1) - s1 / (2 * pow(s2, 2));

        double geo_avg = sqrt(s1 * s2);
        double s1_min = geo_avg;
        double s2_min = geo_avg;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
        break;
      }
      case igl::MappingEnergyType::EXP_CONFORMAL:
      {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));

        double geo_avg = sqrt(s1 * s2);
        double s1_min = geo_avg;
        double s2_min = geo_avg;

        double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
        break;
      }
      case igl::MappingEnergyType::EXP_SYMMETRIC_DIRICHLET:
      {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));

        double in_exp = exp_f * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
        break;
      }
      }

      if (std::abs(s1 - 1) < eps)
        m_sing_new(0) = 1;
      if (std::abs(s2 - 1) < eps)
        m_sing_new(1) = 1;
      mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

      W.row(i) = Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>>(mat_W.data());
      // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
      Ri.row(i) = Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>>(ri.data());
    }
  }
  else
  {
    typedef Eigen::Matrix<double, 3, 1> Vec3;
    typedef Eigen::Matrix<double, 3, 3, Eigen::ColMajor> Mat3;
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> RMat3;
    Mat3 ji;
    Vec3 m_sing_new;
    Vec3 closest_sing_vec;
    const double sqrt_2 = sqrt(2);
    for (int i = 0; i < Ji.rows(); ++i)
    {
      ji << Ji(i, 0), Ji(i, 1), Ji(i, 2),
          Ji(i, 3), Ji(i, 4), Ji(i, 5),
          Ji(i, 6), Ji(i, 7), Ji(i, 8);

      Mat3 ri, ti, ui, vi;
      Vec3 sing;
      igl::polar_svd(ji, ri, ti, ui, sing, vi);

      double s1 = sing(0);
      double s2 = sing(1);
      double s3 = sing(2);

      // 1) Update Weights
      switch (slim_energy)
      {
      case igl::MappingEnergyType::ARAP:
      {
        m_sing_new << 1, 1, 1;
        break;
      }
      case igl::MappingEnergyType::LOG_ARAP:
      {
        double s1_g = 2 * (log(s1) / s1);
        double s2_g = 2 * (log(s2) / s2);
        double s3_g = 2 * (log(s3) / s3);
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
        break;
      }
      case igl::MappingEnergyType::SYMMETRIC_DIRICHLET:
      {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        double s3_g = 2 * (s3 - pow(s3, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
        break;
      }
      case igl::MappingEnergyType::EXP_SYMMETRIC_DIRICHLET:
      {
        double s1_g = 2 * (s1 - pow(s1, -3));
        double s2_g = 2 * (s2 - pow(s2, -3));
        double s3_g = 2 * (s3 - pow(s3, -3));
        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

        double in_exp = exp_f * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;
        s3_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

        break;
      }
      case igl::MappingEnergyType::CONFORMAL:
      {
        double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

        double s1_g = (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2))) / common_div;
        double s2_g = (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2))) / common_div;
        double s3_g = (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2))) / common_div;

        double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
        double s1_min = closest_s;
        double s2_min = closest_s;
        double s3_min = closest_s;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min))), sqrt(s3_g / (2 * (s3 - s3_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min, s3_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
        break;
      }
      case igl::MappingEnergyType::EXP_CONFORMAL:
      {
        // E_conf = (s1^2 + s2^2 + s3^2)/(3*(s1*s2*s3)^(2/3) )
        // dE_conf/ds1 = (-2*(s2*s3)*(s2^2+s3^2 -2*s1^2) ) / (9*(s1*s2*s3)^(5/3))
        // Argmin E_conf(s1): s1 = sqrt(s1^2+s2^2)/sqrt(2)
        double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

        double s1_g = (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2))) / common_div;
        double s2_g = (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2))) / common_div;
        double s3_g = (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2))) / common_div;

        double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow((s1 * s2 * s3), 2. / 3)));
        ;
        double exp_thing = exp(in_exp);

        double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
        double s1_min = closest_s;
        double s2_min = closest_s;
        double s3_min = closest_s;

        s1_g *= exp_thing * exp_f;
        s2_g *= exp_thing * exp_f;
        s3_g *= exp_thing * exp_f;

        m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min))), sqrt(s3_g / (2 * (s3 - s3_min)));

        // change local step
        closest_sing_vec << s1_min, s2_min, s3_min;
        ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
      }
      }
      if (std::abs(s1 - 1) < eps)
        m_sing_new(0) = 1;
      if (std::abs(s2 - 1) < eps)
        m_sing_new(1) = 1;
      if (std::abs(s3 - 1) < eps)
        m_sing_new(2) = 1;
      RMat3 mat_W;
      mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

      W.row(i) = Eigen::Map<Eigen::Matrix<double, 1, 9, Eigen::RowMajor>>(mat_W.data());
      // 2) Update closest rotations (not rotations in case of conformal energy)
      Ri.row(i) = Eigen::Map<Eigen::Matrix<double, 1, 9, Eigen::RowMajor>>(ri.data());
    } // for loop end

  } // if dim end
}

IGL_INLINE void igl::slim_buildA(const Eigen::SparseMatrix<double> &Dx,
                                 const Eigen::SparseMatrix<double> &Dy,
                                 const Eigen::SparseMatrix<double> &Dz,
                                 const Eigen::MatrixXd &W,
                                 std::vector<Eigen::Triplet<double>> &IJV)
{
  const int dim = (W.cols() == 4) ? 2 : 3;
  const int f_n = W.rows();
  const int v_n = Dx.cols();

  // formula (35) in paper
  if (dim == 2)
  {
    IJV.reserve(4 * (Dx.outerSize() + Dy.outerSize()));

    /*A = [W11*Dx, W12*Dx;
          W11*Dy, W12*Dy;
          W21*Dx, W22*Dx;
          W21*Dy, W22*Dy];*/
    for (int k = 0; k < Dx.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it)
      {
        int dx_r = it.row();
        int dx_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, val * W(dx_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(dx_r, v_n + dx_c, val * W(dx_r, 1)));

        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dx_r, dx_c, val * W(dx_r, 2)));
        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dx_r, v_n + dx_c, val * W(dx_r, 3)));
      }
    }

    for (int k = 0; k < Dy.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it)
      {
        int dy_r = it.row();
        int dy_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, dy_c, val * W(dy_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, v_n + dy_c, val * W(dy_r, 1)));

        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dy_r, dy_c, val * W(dy_r, 2)));
        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dy_r, v_n + dy_c, val * W(dy_r, 3)));
      }
    }
  }
  else
  {

    /*A = [W11*Dx, W12*Dx, W13*Dx;
            W11*Dy, W12*Dy, W13*Dy;
            W11*Dz, W12*Dz, W13*Dz;
            W21*Dx, W22*Dx, W23*Dx;
            W21*Dy, W22*Dy, W23*Dy;
            W21*Dz, W22*Dz, W23*Dz;
            W31*Dx, W32*Dx, W33*Dx;
            W31*Dy, W32*Dy, W33*Dy;
            W31*Dz, W32*Dz, W33*Dz;];*/
    IJV.reserve(9 * (Dx.outerSize() + Dy.outerSize() + Dz.outerSize()));
    for (int k = 0; k < Dx.outerSize(); k++)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it)
      {
        int dx_r = it.row();
        int dx_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, val * W(dx_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(dx_r, v_n + dx_c, val * W(dx_r, 1)));
        IJV.push_back(Eigen::Triplet<double>(dx_r, 2 * v_n + dx_c, val * W(dx_r, 2)));

        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r, dx_c, val * W(dx_r, 3)));
        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r, v_n + dx_c, val * W(dx_r, 4)));
        IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r, 2 * v_n + dx_c, val * W(dx_r, 5)));

        IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r, dx_c, val * W(dx_r, 6)));
        IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r, v_n + dx_c, val * W(dx_r, 7)));
        IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r, 2 * v_n + dx_c, val * W(dx_r, 8)));
      }
    }

    for (int k = 0; k < Dy.outerSize(); k++)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it)
      {
        int dy_r = it.row();
        int dy_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, dy_c, val * W(dy_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, v_n + dy_c, val * W(dy_r, 1)));
        IJV.push_back(Eigen::Triplet<double>(f_n + dy_r, 2 * v_n + dy_c, val * W(dy_r, 2)));

        IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r, dy_c, val * W(dy_r, 3)));
        IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r, v_n + dy_c, val * W(dy_r, 4)));
        IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r, 2 * v_n + dy_c, val * W(dy_r, 5)));

        IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r, dy_c, val * W(dy_r, 6)));
        IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r, v_n + dy_c, val * W(dy_r, 7)));
        IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r, 2 * v_n + dy_c, val * W(dy_r, 8)));
      }
    }

    for (int k = 0; k < Dz.outerSize(); k++)
    {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Dz, k); it; ++it)
      {
        int dz_r = it.row();
        int dz_c = it.col();
        double val = it.value();

        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r, dz_c, val * W(dz_r, 0)));
        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r, v_n + dz_c, val * W(dz_r, 1)));
        IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r, 2 * v_n + dz_c, val * W(dz_r, 2)));

        IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r, dz_c, val * W(dz_r, 3)));
        IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r, v_n + dz_c, val * W(dz_r, 4)));
        IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r, 2 * v_n + dz_c, val * W(dz_r, 5)));

        IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r, dz_c, val * W(dz_r, 6)));
        IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r, v_n + dz_c, val * W(dz_r, 7)));
        IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r, 2 * v_n + dz_c, val * W(dz_r, 8)));
      }
    }
  }
}
/// Slim Implementation

IGL_INLINE void igl::slim_precompute(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_init,
    igl::SLIMData &data,
    igl::MappingEnergyType slim_energy,
    const Eigen::VectorXi &b,
    const Eigen::MatrixXd &bc,
    double soft_p)
{

  data.V = V;
  data.F = F;
  data.V_o = V_init;

  data.v_num = V.rows();
  data.f_num = F.rows();

  data.slim_energy = slim_energy;

  data.b = b;
  data.bc = bc;
  data.soft_const_p = soft_p;

  data.proximal_p = 0.0001;

  igl::doublearea(V, F, data.M);
  data.M /= 2.;
  data.mesh_area = data.M.sum();
  data.mesh_improvement_3d = false; // whether to use a jacobian derived from a real mesh or an abstract regular mesh (used for mesh improvement)
  data.exp_factor = 1.0;            // param used only for exponential energies (e.g exponential symmetric dirichlet)

  assert(F.cols() == 3 || F.cols() == 4);

  igl::slim::pre_calc(data);
  data.energy = igl::slim::compute_energy(data, data.V_o) / data.mesh_area;
}

IGL_INLINE Eigen::MatrixXd igl::slim_solve(igl::SLIMData &data, int iter_num)
{
  for (int i = 0; i < iter_num; i++)
  {
    Eigen::MatrixXd dest_res;
    dest_res = data.V_o;

    // Solve Weighted Proxy
    igl::slim::update_weights_and_closest_rotations(data, dest_res);
    igl::slim::solve_weighted_arap(data, data.V, data.F, dest_res, data.b, data.bc);

    double old_energy = data.energy;

    std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
                                                                  Eigen::MatrixXd &aaa) { return igl::slim::compute_energy(data, aaa); };

    data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
                                                 data.energy * data.mesh_area) /
                  data.mesh_area;
  }
  return data.V_o;
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
#endif
