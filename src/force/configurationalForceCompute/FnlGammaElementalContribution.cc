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
// @author Sambit Das
//
#include <force.h>
#include <dft.h>

namespace dftfe
{
  //(locally used function) compute Fnl contibution due to Gamma(Rj) for given
  // set
  // of cells
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    FnlGammaAtomsElementalContribution(
      std::map<unsigned int, std::vector<double>>
        &                                  forceContributionFnlGammaAtoms,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      dealii::FEEvaluation<3,
                           1,
                           C_num1DQuadNLPSP<FEOrder>() *
                             C_numCopies1DQuadNLPSP(),
                           3> &            forceEvalNLP,
      const unsigned int                   cell,
      const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
#ifdef USE_COMPLEX
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
      const std::vector<dataTypes::number> &zetaDeltaVQuadsFlattened,
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
  {
    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
    const unsigned int numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const unsigned int numQuadPoints = forceEvalNLP.n_q_points;

    const unsigned int numNonLocalAtomsCurrentProcess =
      (dftPtr->d_oncvClassPtr->getTotalNumberOfAtomsInCurrentProcessor());
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        zeroTensor3[idim] = dealii::make_vectorized_array(0.0);
      }

    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      FVectQuads(numQuadPoints, zeroTensor3);

    for (int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
      {
        //
        // get the global charge Id of the current nonlocal atom
        //
        const int nonLocalAtomId =
          dftPtr->d_oncvClassPtr->getAtomIdInCurrentProcessor(iAtom);

        // FIXME should use the appropriate map from oncvClassPtr
        // instead of assuming all atoms are nonlocal atoms
        const int globalChargeIdNonLocalAtom =
          dftPtr->d_atomIdPseudopotentialInterestToGlobalId
            .find(nonLocalAtomId)
            ->second;

        // if map entry corresponding to current nonlocal atom id is empty,
        // initialize it to zero
        if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom) ==
            forceContributionFnlGammaAtoms.end())
          forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom] =
            std::vector<double>(3, 0.0);

        std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

        bool isPseudoWfcsAtomInMacroCell = false;
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            bool               isPseudoWfcsAtomInCell = false;
            const unsigned int elementId =
              cellIdToCellNumberMap.find(subCellPtr->id())->second;
            for (unsigned int i = 0;
                 i < (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                        ->getCellIdToAtomIdsLocalCompactSupportMap())
                       .find(elementId)
                       ->second.size();
                 i++)
              if ((dftPtr->d_oncvClassPtr->getNonLocalOperator()
                     ->getCellIdToAtomIdsLocalCompactSupportMap())
                    .find(elementId)
                    ->second[i] == iAtom)
                {
                  isPseudoWfcsAtomInCell      = true;
                  isPseudoWfcsAtomInMacroCell = true;
                  break;
                }

            if (isPseudoWfcsAtomInCell)
              {
                for (unsigned int kPoint = 0;
                     kPoint < dftPtr->d_kPointWeights.size();
                     ++kPoint)
                  {
                    std::vector<double> kcoord(3, 0.0);
                    kcoord[0] = dftPtr->d_kPointCoordinates[kPoint * 3 + 0];
                    kcoord[1] = dftPtr->d_kPointCoordinates[kPoint * 3 + 1];
                    kcoord[2] = dftPtr->d_kPointCoordinates[kPoint * 3 + 2];

                    const unsigned int startingPseudoWfcIdFlattened =
                      kPoint *
                        dftPtr->d_oncvClassPtr->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPoints +
                      (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                         ->getNonTrivialSphericalFnsCellStartIndex())
                          [elementId] *
                        numQuadPoints +
                      (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                         ->getAtomIdToNonTrivialSphericalFnCellStartIndex())
                          .find(iAtom)
                          ->second[elementId] *
                        numQuadPoints;

                    const unsigned int numberPseudoWaveFunctions =
                      dftPtr->d_oncvClassPtr
                        ->getTotalNumberOfSphericalFunctionsForAtomId(
                          nonLocalAtomId);
                    // std::cout<<startingPseudoWfcIdFlattened <<std::endl;
                    std::vector<dataTypes::number> temp2(3);
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        std::vector<dataTypes::number> F(
                          3, dataTypes::number(0.0));

                        for (unsigned int iPseudoWave = 0;
                             iPseudoWave < numberPseudoWaveFunctions;
                             ++iPseudoWave)
                          {
                            const dataTypes::number temp1 =
                              zetaDeltaVQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            temp2[0] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints + q];
                            temp2[1] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 numQuadPoints + q];
                            temp2[2] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 2 * numQuadPoints + q];
#ifdef USE_COMPLEX
                            const dataTypes::number temp3 =
                              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            F[0] +=
                              2.0 * (temp1 * temp2[0] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[0]));
                            F[1] +=
                              2.0 * (temp1 * temp2[1] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[1]));
                            F[2] +=
                              2.0 * (temp1 * temp2[2] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[2]));
#else
                            F[0] += 2.0 * (temp1 * temp2[0]);
                            F[1] += 2.0 * (temp1 * temp2[1]);
                            F[2] += 2.0 * (temp1 * temp2[2]);
#endif
                          } // pseudowavefunctions loop

                        FVectQuads[q][0][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[0]);
                        FVectQuads[q][1][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[1]);
                        FVectQuads[q][2][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[2]);

                        // std::cout<<F[0] <<std::endl;
                        // std::cout<<F[1] <<std::endl;
                        // std::cout<<F[2] <<std::endl;
                      } // quad-loop
                  }     // kpoint loop
              }         // non-trivial cell check
          }             // subcell loop

        if (isPseudoWfcsAtomInMacroCell)
          {
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              forceEvalNLP.submit_value(FVectQuads[q], q);

            const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
              forceContributionFnlGammaiAtomCells =
                forceEvalNLP.integrate_value();

            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              for (unsigned int idim = 0; idim < 3; idim++)
                forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]
                                              [idim] +=
                  forceContributionFnlGammaiAtomCells[idim][iSubCell];
          }
      } // iAtom loop
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void forceClass<FEOrder, FEOrderElectro, memorySpace>::
    FnlGammaxElementalContribution(
      dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &FVectQuads,
      const dealii::MatrixFree<3, double> &                     matrixFreeData,
      const unsigned int                                        numQuadPoints,
      const unsigned int                                        cell,
      const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
      const std::vector<dataTypes::number> &        zetaDeltaVQuadsFlattened,
      const std::vector<dataTypes::number>
        &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
  {
    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
    const unsigned int numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);

    const unsigned int numNonLocalAtomsCurrentProcess =
      (dftPtr->d_oncvClassPtr->getNonLocalOperator()
         ->getTotalAtomInCurrentProcessor());

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        zeroTensor3[idim] = dealii::make_vectorized_array(0.0);
      }
    std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

    for (int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
      {
        //
        // get the global charge Id of the current nonlocal atom
        //
        // FIX ME with correct call from ONCV
        const int nonLocalAtomId =
          dftPtr->d_oncvClassPtr->getAtomIdInCurrentProcessor(iAtom);
        const int globalChargeIdNonLocalAtom =
          dftPtr->d_atomIdPseudopotentialInterestToGlobalId
            .find(nonLocalAtomId)
            ->second;



        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            bool               isPseudoWfcsAtomInCell = false;
            const unsigned int elementId =
              cellIdToCellNumberMap.find(subCellPtr->id())->second;
            for (unsigned int i = 0;
                 i < (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                        ->getCellIdToAtomIdsLocalCompactSupportMap())
                       .find(elementId)
                       ->second.size();
                 i++)
              if ((dftPtr->d_oncvClassPtr->getNonLocalOperator()
                     ->getCellIdToAtomIdsLocalCompactSupportMap())
                    .find(elementId)
                    ->second[i] == iAtom)
                {
                  isPseudoWfcsAtomInCell = true;
                  break;
                }

            if (isPseudoWfcsAtomInCell)
              {
                for (unsigned int kPoint = 0;
                     kPoint < dftPtr->d_kPointWeights.size();
                     ++kPoint)
                  {
                    const unsigned int startingPseudoWfcIdFlattened =
                      kPoint *
                        (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                           ->getTotalNonTrivialSphericalFnsOverAllCells()) *
                        numQuadPoints +
                      (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                         ->getNonTrivialSphericalFnsCellStartIndex())
                          [elementId] *
                        numQuadPoints +
                      (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                         ->getAtomIdToNonTrivialSphericalFnCellStartIndex())
                          .find(iAtom)
                          ->second[elementId] *
                        numQuadPoints;

                    const unsigned int numberPseudoWaveFunctions =
                      dftPtr->d_oncvClassPtr
                        ->getTotalNumberOfSphericalFunctionsForAtomId(
                          nonLocalAtomId);
                    std::vector<dataTypes::number> temp2(3);
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        std::vector<dataTypes::number> F(
                          3, dataTypes::number(0.0));

                        for (unsigned int iPseudoWave = 0;
                             iPseudoWave < numberPseudoWaveFunctions;
                             ++iPseudoWave)
                          {
                            const dataTypes::number temp1 =
                              zetaDeltaVQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            temp2[0] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints + q];
                            temp2[1] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 numQuadPoints + q];
                            temp2[2] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 2 * numQuadPoints + q];
                            F[0] -= 2.0 * (temp1 * temp2[0]);
                            F[1] -= 2.0 * (temp1 * temp2[1]);
                            F[2] -= 2.0 * (temp1 * temp2[2]);
                          } // pseudowavefunctions loop

                        FVectQuads[q][0][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[0]);
                        FVectQuads[q][1][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[1]);
                        FVectQuads[q][2][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[2]);
                      } // quad-loop
                  }     // kpoint loop
              }         // non-trivial cell check
          }             // subcell loop
      }                 // iAtom loop
  }

  //(locally used function) accumulate and distribute Fnl contibution due to
  // Gamma(Rj)
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    distributeForceContributionFnlGammaAtoms(
      const std::map<unsigned int, std::vector<double>>
        &forceContributionFnlGammaAtoms)
  {
    for (unsigned int iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
      {
        bool doesAtomIdExistOnLocallyOwnedNode = false;
        if (d_atomsForceDofs.find(
              std::pair<unsigned int, unsigned int>(iAtom, 0)) !=
            d_atomsForceDofs.end())
          doesAtomIdExistOnLocallyOwnedNode = true;

        std::vector<double> forceContributionFnlGammaiAtomGlobal(3);
        std::vector<double> forceContributionFnlGammaiAtomLocal(3, 0.0);

        if (forceContributionFnlGammaAtoms.find(iAtom) !=
            forceContributionFnlGammaAtoms.end())
          forceContributionFnlGammaiAtomLocal =
            forceContributionFnlGammaAtoms.find(iAtom)->second;
        // accumulate value
        MPI_Allreduce(&(forceContributionFnlGammaiAtomLocal[0]),
                      &(forceContributionFnlGammaiAtomGlobal[0]),
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        if (doesAtomIdExistOnLocallyOwnedNode)
          {
            std::vector<dealii::types::global_dof_index> forceLocalDofIndices(
              3);
            for (unsigned int idim = 0; idim < 3; idim++)
              forceLocalDofIndices[idim] =
                d_atomsForceDofs[std::pair<unsigned int, unsigned int>(iAtom,
                                                                       idim)];
#ifdef USE_COMPLEX
            d_constraintsNoneForce.distribute_local_to_global(
              forceContributionFnlGammaiAtomGlobal,
              forceLocalDofIndices,
              d_configForceVectorLinFEKPoints);
#else
            d_constraintsNoneForce.distribute_local_to_global(
              forceContributionFnlGammaiAtomGlobal,
              forceLocalDofIndices,
              d_configForceVectorLinFE);
#endif
          }
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
