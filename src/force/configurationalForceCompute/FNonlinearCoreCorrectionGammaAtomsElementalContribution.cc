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
// @author Phani Motamarri, Sambit Das (2020)
//
#include <force.h>
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>
#include <dft.h>

namespace dftfe
{
  //(locally used function) compute FNonlinearCoreCorrection contibution due to
  // Gamma(Rj) for given set of cells
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    FNonlinearCoreCorrectionGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>> &vxcQuads,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (unsigned int idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);
    const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
    const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const unsigned int totalNumberAtoms =
      numberGlobalAtoms + numberImageCharges;
    const unsigned int numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const unsigned int numQuadPoints = forceEval.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);

        unsigned int atomId = iAtom;
        if (iAtom >= numberGlobalAtoms)
          {
            const int imageId = iAtom - numberGlobalAtoms;
            atomId            = dftPtr->d_imageIdsTrunc[imageId];
          }

        bool isLocalDomainOutsideCoreRhoTail = false;
        if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
          isLocalDomainOutsideCoreRhoTail = true;

        if (isLocalDomainOutsideCoreRhoTail)
          continue;

        bool isCellOutsideCoreRhoTail = true;
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            // get grad rho for iAtom
            if (!isLocalDomainOutsideCoreRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
                if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    isCellOutsideCoreRhoTail        = false;
                    const std::vector<double> &temp = it->second;
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * 3];
                        gradRhoCoreAtomsQuads[q][1][iSubCell] = temp[q * 3 + 1];
                        gradRhoCoreAtomsQuads[q][2][iSubCell] = temp[q * 3 + 2];
                      }
                  }
              }
          } // subCell loop

        if (isCellOutsideCoreRhoTail)
          continue;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            forceEval.submit_value(-eshelbyTensor::getFNonlinearCoreCorrection(
                                     vxcQuads[q], gradRhoCoreAtomsQuads[q]),
                                   q);
          }
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
          forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
            forceEval.integrate_value();

        if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
            forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
          forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
            std::vector<double>(3, 0.0);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (unsigned int idim = 0; idim < 3; idim++)
            {
              forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId]
                                                                 [idim] +=
                forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                  [idim][iSubCell];
            }
      } // iAtom loop
  }


  //(locally used function) compute FNonlinearCoreCorrection contibution due to
  // Gamma(Rj) for given set of cells
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    FNonlinearCoreCorrectionGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &derExcGradRho,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &hessianRhoCoreAtoms)
  {
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          {
            zeroTensor1[idim][jdim] = dealii::make_vectorized_array(0.0);
          }
      }

    const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
    const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const unsigned int totalNumberAtoms =
      numberGlobalAtoms + numberImageCharges;
    const unsigned int numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const unsigned int numQuadPoints = forceEval.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);

        unsigned int atomId = iAtom;
        if (iAtom >= numberGlobalAtoms)
          {
            const int imageId = iAtom - numberGlobalAtoms;
            atomId            = dftPtr->d_imageIdsTrunc[imageId];
          }

        bool isLocalDomainOutsideCoreRhoTail = false;
        if (hessianRhoCoreAtoms.find(iAtom) == hessianRhoCoreAtoms.end())
          isLocalDomainOutsideCoreRhoTail = true;

        if (isLocalDomainOutsideCoreRhoTail)
          continue;

        bool isCellOutsideCoreRhoTail = true;

        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            // get grad rho for iAtom
            if (!isLocalDomainOutsideCoreRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

                if (it != hessianRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    isCellOutsideCoreRhoTail        = false;
                    const std::vector<double> &temp = it->second;
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          for (unsigned int jDim = 0; jDim < 3; ++jDim)
                            hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                              temp[q * 3 * 3 + 3 * iDim + jDim];
                      }
                  }
              }
          } // subCell loop

        if (isCellOutsideCoreRhoTail)
          continue;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            forceEval.submit_value(-eshelbyTensor::getFNonlinearCoreCorrection(
                                     derExcGradRho[q],
                                     hessianRhoCoreAtomsQuads[q]),
                                   q);
          }
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
          forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
            forceEval.integrate_value();

        if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
            forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
          forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
            std::vector<double>(3, 0.0);

        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (unsigned int idim = 0; idim < 3; idim++)
            {
              forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId]
                                                                 [idim] +=
                forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                  [idim][iSubCell];
            }
      } // iAtom loop
  }

  //(locally used function) compute FNonlinearCoreCorrection contibution due to
  // Gamma(Rj) for given set of cells
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
      std::map<unsigned int, std::vector<double>>
        &forceContributionFNonlinearCoreCorrectionGammaAtoms,
      dealii::FEEvaluation<
        3,
        1,
        C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
        3> &                               forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin0,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin1,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin0,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin1,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &        hessianRhoCoreAtoms,
      const bool isXCGGA)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (unsigned int idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (unsigned int idim = 0; idim < 3; idim++)
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);

    const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
    const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const unsigned int totalNumberAtoms =
      numberGlobalAtoms + numberImageCharges;
    const unsigned int numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const unsigned int numQuadPoints = forceEval.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

        unsigned int atomId = iAtom;
        if (iAtom >= numberGlobalAtoms)
          {
            const int imageId = iAtom - numberGlobalAtoms;
            atomId            = dftPtr->d_imageIdsTrunc[imageId];
          }

        bool isLocalDomainOutsideCoreRhoTail = false;
        if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
          isLocalDomainOutsideCoreRhoTail = true;

        if (isLocalDomainOutsideCoreRhoTail)
          continue;

        bool isCellOutsideCoreRhoTail = true;
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            // get grad rho for iAtom
            if (!isLocalDomainOutsideCoreRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
                if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    isCellOutsideCoreRhoTail        = false;
                    const std::vector<double> &temp = it->second;
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        gradRhoCoreAtomsQuads[q][0][iSubCell] =
                          temp[q * 3] / 2.0;
                        gradRhoCoreAtomsQuads[q][1][iSubCell] =
                          temp[q * 3 + 1] / 2.0;
                        gradRhoCoreAtomsQuads[q][2][iSubCell] =
                          temp[q * 3 + 2] / 2.0;
                      }
                  }

                if (isXCGGA && !isCellOutsideCoreRhoTail)
                  {
                    std::map<dealii::CellId,
                             std::vector<double>>::const_iterator it2 =
                      hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

                    if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                      {
                        const std::vector<double> &temp2 = it2->second;
                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          {
                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              for (unsigned int jDim = 0; jDim < 3; ++jDim)
                                hessianRhoCoreAtomsQuads
                                  [q][iDim][jDim][iSubCell] =
                                    temp2[q * 3 * 3 + 3 * iDim + jDim] / 2.0;
                          }
                      }
                  }
              }
          } // subCell loop

        if (isCellOutsideCoreRhoTail)
          continue;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          forceEval.submit_value(-eshelbyTensorSP::getFNonlinearCoreCorrection(
                                   vxcQuadsSpin0[q],
                                   vxcQuadsSpin1[q],
                                   derExcGradRhoSpin0[q],
                                   derExcGradRhoSpin1[q],
                                   gradRhoCoreAtomsQuads[q],
                                   hessianRhoCoreAtomsQuads[q],
                                   isXCGGA),
                                 q);

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
          forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
            forceEval.integrate_value();

        if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
            forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
          forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
            std::vector<double>(3, 0.0);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (unsigned int idim = 0; idim < 3; idim++)
            {
              forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId]
                                                                 [idim] +=
                forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                  [idim][iSubCell];
            }
      } // iAtom loop
  }
#include "../force.inst.cc"
} // namespace dftfe
