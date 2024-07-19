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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

// source file for electron density related computations
#include <dft.h>
#include <densityCalculator.h>

namespace dftfe
{
  // calculate electron density
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::compute_rhoOut(
    const bool isConsiderSpectrumSplitting,
    const bool isGroundState)
  {
    if ((d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
         d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ||
         d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND") ||
        d_dftParamsPtr->pawPseudoPotential)
      {
        pcout << "Compute Rho Out Nodal: " << std::endl;
        computeRhoNodalFromPSI(isConsiderSpectrumSplitting);

        // normalize rho
        if (!d_dftParamsPtr->pawPseudoPotential)
          {
            const double charge =
              totalCharge(d_matrixFreeDataPRefined, d_densityOutNodalValues[0]);


            const double scalingFactor = ((double)numElectrons) / charge;

            // scale nodal vector with scalingFactor
            for (unsigned int iComp = 0; iComp < d_densityOutNodalValues.size();
                 ++iComp)
              d_densityOutNodalValues[iComp] *= scalingFactor;
            if (d_dftParamsPtr->verbosity >= 3)
              {
                pcout << "Total Charge before scaling: " << charge << std::endl;
                pcout << "Total Charge using nodal Rho out: "
                      << totalCharge(d_matrixFreeDataPRefined,
                                     d_densityOutNodalValues[0])
                      << std::endl;
              }
          }

        // interpolate nodal rhoOut data to quadrature data
        for (unsigned int iComp = 0; iComp < d_densityOutNodalValues.size();
             ++iComp)
          interpolateDensityNodalDataToQuadratureDataGeneral(
            d_basisOperationsPtrElectroHost,
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_densityOutNodalValues[iComp],
            d_densityOutQuadValues[iComp],
            d_gradDensityOutQuadValues[iComp],
            d_gradDensityOutQuadValues[iComp],
            d_excManagerPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA);
      }
    else
      {
        d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId, false);
        const unsigned int nQuadsPerCell =
          d_basisOperationsPtrHost->nQuadsPerCell();
        const unsigned int nCells = d_basisOperationsPtrHost->nCells();
        d_densityOutQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 :
                                                                           1);
        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          {
            d_gradDensityOutQuadValues.resize(
              d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
          }
        for (unsigned int iComp = 0; iComp < d_densityOutQuadValues.size();
             ++iComp)
          d_densityOutQuadValues[iComp].resize(nQuadsPerCell * nCells);

        for (unsigned int iComp = 0; iComp < d_gradDensityOutQuadValues.size();
             ++iComp)
          d_gradDensityOutQuadValues[iComp].resize(3 * nQuadsPerCell * nCells);

#if defined(DFTFE_WITH_DEVICE)
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                            &d_eigenVectorsRotFracFlattenedDevice,
                            d_numEigenValues,
                            d_numEigenValuesRR,
                            eigenValues,
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown,
                            d_basisOperationsPtrDevice,
                            d_BLASWrapperPtr,
                            d_densityDofHandlerIndex,
                            d_densityQuadratureId,
                            d_kPointWeights,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_excManagerPtr->getDensityBasedFamilyType() ==
                              densityFamilyType::GGA,
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr,
                            isConsiderSpectrumSplitting &&
                              d_numEigenValues != d_numEigenValuesRR,
                            d_dftParamsPtr->pawPseudoPotential ? d_pawClassPtr :
                                                                 NULL);
        else
#endif
          computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                            &d_eigenVectorsRotFracDensityFlattenedHost,
                            d_numEigenValues,
                            d_numEigenValuesRR,
                            eigenValues,
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown,
                            d_basisOperationsPtrHost,
                            d_BLASWrapperPtrHost,
                            d_densityDofHandlerIndex,
                            d_densityQuadratureId,
                            d_kPointWeights,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_excManagerPtr->getDensityBasedFamilyType() ==
                              densityFamilyType::GGA,
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr,
                            isConsiderSpectrumSplitting &&
                              d_numEigenValues != d_numEigenValuesRR,
                            d_dftParamsPtr->pawPseudoPotential ? d_pawClassPtr :
                                                                 NULL);
        // normalizeRhoOutQuadValues();

        if (d_dftParamsPtr->computeEnergyEverySCF || isGroundState)
          {
            computeRhoNodalFromPSI(isConsiderSpectrumSplitting);

            // // normalize rho
            // const double charge =
            //   totalCharge(d_matrixFreeDataPRefined,
            //   d_densityOutNodalValues[0]);


            // const double scalingFactor = ((double)numElectrons) / charge;

            // // scale nodal vector with scalingFactor
            // d_densityOutNodalValues[0] *= scalingFactor;
          }
      }

    if (d_dftParamsPtr->computeEnergyEverySCF || isGroundState)
      {
        d_rhoOutNodalValuesDistributed = d_densityOutNodalValues[0];
        d_rhoOutNodalValuesDistributed.update_ghost_values();
        d_constraintsRhoNodalInfo.distribute(d_rhoOutNodalValuesDistributed);
        interpolateDensityNodalDataToQuadratureDataLpsp(
          d_basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_lpspQuadratureIdElectro,
          d_densityOutNodalValues[0],
          d_densityTotalOutValuesLpspQuad,
          d_gradDensityTotalOutValuesLpspQuad,
          true);

        // std::vector<
        //   dftfe::utils::MemoryStorage<double,
        //   dftfe::utils::MemorySpace::HOST>> TempVector, TempVector2;

        // TempVector.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
        // computeRhoFromPSI(
        //   &d_eigenVectorsFlattenedHost,
        //   &d_eigenVectorsRotFracDensityFlattenedHost,
        //   d_numEigenValues,
        //   d_numEigenValuesRR,
        //   eigenValues,
        //   fermiEnergy,
        //   fermiEnergyUp,
        //   fermiEnergyDown,
        //   d_basisOperationsPtrHost,
        //   d_BLASWrapperPtrHost,
        //   d_densityDofHandlerIndex,
        //   d_lpspQuadratureId,
        //   d_kPointWeights,
        //   TempVector,
        //   TempVector2,
        //   false,
        //   d_mpiCommParent,
        //   interpoolcomm,
        //   interBandGroupComm,
        //   *d_dftParamsPtr,
        //   isConsiderSpectrumSplitting && d_numEigenValues !=
        //   d_numEigenValuesRR,
        //   std::shared_ptr<dftfe::pawClass<dataTypes::number,
        //                                   dftfe::utils::MemorySpace::HOST>>(
        //     nullptr));
        // d_densityTotalOutValuesLpspQuad = TempVector[0];
      }

    if (isGroundState &&
        ((d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
          d_dftParamsPtr->solverMode == "GEOOPT") ||
         (d_dftParamsPtr->extrapolateDensity == 2 &&
          d_dftParamsPtr->solverMode == "MD")) &&
        d_dftParamsPtr->spinPolarized != 1)
      {
        d_rhoOutNodalValuesSplit = d_densityOutNodalValues[0];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                     rhoOutValuesCopy = d_densityOutQuadValues[0];
        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        const unsigned int n_q_points = quadrature_formula.size();

        const double charge =
          totalCharge(d_dofHandlerRhoNodal, d_densityOutQuadValues[0]);
        const double scaling = ((double)numElectrons) / charge;

        // scaling rho
        for (unsigned int i = 0; i < rhoOutValuesCopy.size(); ++i)
          rhoOutValuesCopy[i] *= scaling;
        l2ProjectionQuadDensityMinusAtomicDensity(
          d_basisOperationsPtrElectroHost,
          d_constraintsRhoNodal,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          rhoOutValuesCopy,
          d_rhoOutNodalValuesSplit);
      }
  }


  // rho data reinitilization without remeshing. The rho out of last ground
  // state solve is made the rho in of the new solve
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::noRemeshRhoDataInit()
  {
    // cleanup of existing rho Out and rho In data
    clearRhoData();
    d_densityInQuadValues = d_densityOutQuadValues;
    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        d_gradDensityInQuadValues = d_gradDensityOutQuadValues;
      }

    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ||
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      {
        d_densityInNodalValues = d_densityOutNodalValues;

        // normalize rho
        const double charge =
          totalCharge(d_matrixFreeDataPRefined, d_densityInNodalValues[0]);

        const double scalingFactor = ((double)numElectrons) / charge;

        // scale nodal vector with scalingFactor
        for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size();
             ++iComp)
          d_densityInNodalValues[iComp] *= scalingFactor;

        for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size();
             ++iComp)
          interpolateDensityNodalDataToQuadratureDataGeneral(
            d_basisOperationsPtrElectroHost,
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_densityInNodalValues[iComp],
            d_densityInQuadValues[iComp],
            d_gradDensityInQuadValues[iComp],
            d_gradDensityInQuadValues[iComp],
            d_excManagerPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA);

        d_densityOutQuadValues.resize(d_densityInNodalValues.size());
        for (unsigned int iComp = 0; iComp < d_densityOutQuadValues.size();
             ++iComp)
          d_densityOutQuadValues[iComp].resize(
            d_densityInQuadValues[iComp].size());
        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          {
            d_gradDensityOutQuadValues.resize(d_gradDensityInQuadValues.size());
            for (unsigned int iComp = 0; iComp < d_densityOutQuadValues.size();
                 ++iComp)
              d_gradDensityOutQuadValues[iComp].resize(
                d_gradDensityInQuadValues[iComp].size());
          }
      }

    // scale quadrature values
    // normalizeRhoInQuadValues(); Not sure why we normalise here??
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeRhoNodalFromPSI(
    bool isConsiderSpectrumSplitting)
  {
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityPRefinedNodalData;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityPRefinedNodalData;

    // initialize variables to be used later
    const unsigned int dofs_per_cell =
      d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    const dealii::IndexSet &locallyOwnedDofs =
      d_dofHandlerRhoNodal.locally_owned_dofs();
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_gllQuadratureId);
    const unsigned int numQuadPoints = quadrature_formula.size();

    // get access to quadrature point coordinates and density DoFHandler nodal
    // points
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
    std::vector<unsigned int> renumberingMap(numQuadPoints);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (unsigned int i = 0; i < numQuadPoints; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (unsigned int j = 0; j < numQuadPoints; ++j)
          {
            const dealii::Point<3> &quadCoor = quadraturePointCoor[j];
            double                  dist     = quadCoor.distance(nodalCoor);
            if (dist <= 1e-08)
              {
                renumberingMap[i] = j;
                break;
              }
          }
      }

    // allocate the storage to compute 2p nodal values from wavefunctions
    densityPRefinedNodalData.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);

    // compute rho from wavefunctions at nodal locations of 2p DoFHandler
    // nodes in each cell
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                        &d_eigenVectorsRotFracFlattenedDevice,
                        d_numEigenValues,
                        d_numEigenValuesRR,
                        eigenValues,
                        fermiEnergy,
                        fermiEnergyUp,
                        fermiEnergyDown,
                        d_basisOperationsPtrDevice,
                        d_BLASWrapperPtr,
                        d_densityDofHandlerIndex,
                        d_gllQuadratureId,
                        d_kPointWeights,
                        densityPRefinedNodalData,
                        gradDensityPRefinedNodalData,
                        false,
                        d_mpiCommParent,
                        interpoolcomm,
                        interBandGroupComm,
                        *d_dftParamsPtr,
                        isConsiderSpectrumSplitting &&
                          d_numEigenValues != d_numEigenValuesRR,
                        d_dftParamsPtr->pawPseudoPotential ? d_pawClassPtr :
                                                             NULL);

    else
#endif
      computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                        &d_eigenVectorsRotFracDensityFlattenedHost,
                        d_numEigenValues,
                        d_numEigenValuesRR,
                        eigenValues,
                        fermiEnergy,
                        fermiEnergyUp,
                        fermiEnergyDown,
                        d_basisOperationsPtrHost,
                        d_BLASWrapperPtrHost,
                        d_densityDofHandlerIndex,
                        d_gllQuadratureId,
                        d_kPointWeights,
                        densityPRefinedNodalData,
                        gradDensityPRefinedNodalData,
                        false,
                        d_mpiCommParent,
                        interpoolcomm,
                        interBandGroupComm,
                        *d_dftParamsPtr,
                        isConsiderSpectrumSplitting &&
                          d_numEigenValues != d_numEigenValuesRR,
                        d_dftParamsPtr->pawPseudoPotential ? d_pawClassPtr :
                                                             NULL);

    // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
    dealii::DoFHandler<3>::active_cell_iterator cellP = d_dofHandlerRhoNodal
                                                          .begin_active(),
                                                endcP =
                                                  d_dofHandlerRhoNodal.end();
    unsigned int iCell = 0;
    for (; cellP != endcP; ++cellP)
      {
        if (cellP->is_locally_owned())
          {
            std::vector<dealii::types::global_dof_index> cell_dof_indices(
              dofs_per_cell);
            cellP->get_dof_indices(cell_dof_indices);
            const double *nodalValues =
              densityPRefinedNodalData[0].data() + iCell * dofs_per_cell;

            for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
              {
                const dealii::types::global_dof_index nodeID =
                  cell_dof_indices[iNode];
                if (!d_constraintsRhoNodal.is_constrained(nodeID))
                  {
                    if (locallyOwnedDofs.is_element(nodeID))
                      d_densityOutNodalValues[0](nodeID) =
                        nodalValues[renumberingMap[iNode]];
                  }
              }
            ++iCell;
          }
      }

    cellP = d_dofHandlerRhoNodal.begin_active();
    endcP = d_dofHandlerRhoNodal.end();
    iCell = 0;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        for (; cellP != endcP; ++cellP)
          {
            if (cellP->is_locally_owned())
              {
                std::vector<dealii::types::global_dof_index> cell_dof_indices(
                  dofs_per_cell);
                cellP->get_dof_indices(cell_dof_indices);
                const double *nodalValues =
                  densityPRefinedNodalData[1].data() + iCell * dofs_per_cell;

                for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
                  {
                    const dealii::types::global_dof_index nodeID =
                      cell_dof_indices[iNode];
                    if (!d_constraintsRhoNodal.is_constrained(nodeID))
                      {
                        if (locallyOwnedDofs.is_element(nodeID))
                          {
                            d_densityOutNodalValues[1](nodeID) =
                              nodalValues[renumberingMap[iNode]];
                          }
                      }
                  }
                ++iCell;
              }
          }
      }
  }
#include "dft.inst.cc"

} // namespace dftfe
