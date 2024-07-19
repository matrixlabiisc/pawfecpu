// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author  Nikhil
//
#include <dft.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::applyMultipoleDirichletBC(
    const dealii::DoFHandler<3> &            _dofHandler,
    const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
    dealii::AffineConstraints<double> &      constraintMatrix)

  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);

    const unsigned int vertices_per_cell =
      dealii::GeometryInfo<3>::vertices_per_cell;
    const unsigned int dofs_per_cell  = _dofHandler.get_fe().dofs_per_cell;
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const unsigned int dofs_per_face  = _dofHandler.get_fe().dofs_per_face;

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

    std::vector<bool> dofs_touched(_dofHandler.n_dofs(), false);
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  _dofHandler.begin_active(),
                                                endc = _dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              const unsigned int boundaryId = cell->face(iFace)->boundary_id();
              if (boundaryId == 0)
                {
                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      if (!onlyHangingNodeConstraints.is_constrained(nodeId))
                        {
                          // FIXME: check whether this is the best way to get
                          // the support point
                          dealii::Point<3> p =
                            d_supportPointsPRefined.find(nodeId)->second;
                          double r          = p.norm();
                          double dipole     = 0;
                          double quadrupole = 0;
                          for (unsigned int iDim = 0; iDim < 3; ++iDim)
                            {
                              dipole += d_dipole[iDim] * p[iDim];
                              for (unsigned int jDim = 0; jDim < 3; ++jDim)
                                quadrupole += d_quadrupole[iDim * 3 + jDim] *
                                              p[iDim] * p[jDim];
                            }
                          double constraintValue = 0.0;
                          if (std::abs(d_monopole) > r * 1e-12)
                            constraintValue += d_monopole / r;
                          if (std::abs(dipole) > std::pow(r, 3) * 1e-12)
                            constraintValue += dipole / std::pow(r, 3);
                          if (std::abs(quadrupole) > 2 * std::pow(r, 5) * 1e-12)
                            constraintValue +=
                              quadrupole / 2.0 / std::pow(r, 5);
                          constraintMatrix.add_line(nodeId);
                          constraintMatrix.set_inhomogeneity(nodeId,
                                                             constraintValue);
                        } // non-hanging node check
                    }     // Face dof loop
                }         // non-periodic boundary id
            }             // Face loop
        }                 // cell locally owned
  }
#include "dft.inst.cc"
} // namespace dftfe
