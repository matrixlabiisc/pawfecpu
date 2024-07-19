// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri, Sambit Das
//
#include <dft.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    interpolateDensityNodalDataToQuadratureDataGeneral(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                              basisOperationsPtr,
      const unsigned int               dofHandlerId,
      const unsigned int               quadratureId,
      const distributedCPUVec<double> &nodalField,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureGradValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureHessianValueData,
      const bool isEvaluateGradData,
      const bool isEvaluateHessianData)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells        = basisOperationsPtr->nCells();
    quadratureValueData.clear();
    quadratureValueData.resize(nQuadsPerCell * nCells);
    nodalField.update_ghost_values();
    if (isEvaluateGradData)
      {
        quadratureGradValueData.clear();
        quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
      }
    if (isEvaluateHessianData)
      {
        quadratureHessianValueData.clear();
        quadratureHessianValueData.resize(9 * nQuadsPerCell * nCells);
      }


    // dealii::FEEvaluation<
    //   3,
    //   C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
    //   C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
    //   1,
    //   double>
    //   feEvalObj(basisOperationsPtr->matrixFreeData(),
    //             dofHandlerId,
    //             quadratureId);
    dealii::FEEvaluation<3, -1> feEvalObj(basisOperationsPtr->matrixFreeData(),
                                          dofHandlerId,
                                          quadratureId);

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      basisOperationsPtr->matrixFreeData()
          .get_quadrature(quadratureId)
          .size() == nQuadsPerCell,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0;
         cell < basisOperationsPtr->matrixFreeData().n_cell_batches();
         ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true,
                           isEvaluateGradData ? true : false,
                           isEvaluateHessianData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
              cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex = basisOperationsPtr->cellIndex(subCellId);

            double *tempVec =
              quadratureValueData.data() + cellIndex * nQuadsPerCell;

            for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                double *tempVec2 = quadratureGradValueData.data() +
                                   3 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }

            if (isEvaluateHessianData)
              {
                double *tempVec3 = quadratureHessianValueData.data() +
                                   9 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                      &hessianVals = feEvalObj.get_hessian(q_point);
                    for (unsigned int i = 0; i < 3; i++)
                      for (unsigned int j = 0; j < 3; j++)
                        tempVec3[9 * q_point + 3 * i + j] =
                          hessianVals[i][j][iSubCell];
                  }
              }
          }
      }
  }


  //
  // interpolate nodal data to quadrature values using FEEvaluation
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    interpolateElectroNodalDataToQuadratureDataGeneral(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                              basisOperationsPtr,
      const unsigned int               dofHandlerId,
      const unsigned int               quadratureId,
      const distributedCPUVec<double> &nodalField,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureGradValueData,
      const bool isEvaluateGradData)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells        = basisOperationsPtr->nCells();
    quadratureValueData.clear();
    quadratureValueData.resize(nQuadsPerCell * nCells);
    nodalField.update_ghost_values();
    if (isEvaluateGradData)
      {
        quadratureGradValueData.clear();
        quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
      }

    // dealii::FEEvaluation<
    //   3,
    //   FEOrderElectro,
    //   C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
    //   1,
    //   double>
    //   feEvalObj(basisOperationsPtr->matrixFreeData(),
    //             dofHandlerId,
    //             quadratureId);

    dealii::FEEvaluation<3, -1> feEvalObj(basisOperationsPtr->matrixFreeData(),
                                          dofHandlerId,
                                          quadratureId);

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      basisOperationsPtr->matrixFreeData()
          .get_quadrature(quadratureId)
          .size() == nQuadsPerCell,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0;
         cell < basisOperationsPtr->matrixFreeData().n_cell_batches();
         ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values_plain(nodalField);
        feEvalObj.evaluate(true, isEvaluateGradData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
              cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex = basisOperationsPtr->cellIndex(subCellId);

            double *tempVec =
              quadratureValueData.data() + cellIndex * nQuadsPerCell;

            for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }
            if (isEvaluateGradData)
              {
                double *tempVec2 = quadratureGradValueData.data() +
                                   3 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }
          }
      }
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    interpolateDensityNodalDataToQuadratureDataLpsp(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                              basisOperationsPtr,
      const unsigned int               dofHandlerId,
      const unsigned int               quadratureId,
      const distributedCPUVec<double> &nodalField,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureGradValueData,
      const bool isEvaluateGradData)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells        = basisOperationsPtr->nCells();
    quadratureValueData.clear();
    quadratureValueData.resize(nQuadsPerCell * nCells);
    nodalField.update_ghost_values();
    if (isEvaluateGradData)
      {
        quadratureGradValueData.clear();
        quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
      }



    // dealii::FEEvaluation<3,
    //                      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
    //                      C_num1DQuadLPSP<FEOrder>() *
    //                      C_numCopies1DQuadLPSP(), 1, double>
    //   feEvalObj(basisOperationsPtr->matrixFreeData(),
    //             dofHandlerId,
    //             quadratureId);

    dealii::FEEvaluation<3, -1> feEvalObj(basisOperationsPtr->matrixFreeData(),
                                          dofHandlerId,
                                          quadratureId);

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      basisOperationsPtr->matrixFreeData()
          .get_quadrature(quadratureId)
          .size() == nQuadsPerCell,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0;
         cell < basisOperationsPtr->matrixFreeData().n_cell_batches();
         ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true, isEvaluateGradData ? true : false, false);

        for (unsigned int iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
              cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex = basisOperationsPtr->cellIndex(subCellId);

            double *tempVec =
              quadratureValueData.data() + cellIndex * nQuadsPerCell;

            for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                double *tempVec2 = quadratureGradValueData.data() +
                                   3 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }
          }
      }
  }


  //
  // compute field l2 norm
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::fieldGradl2Norm(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double> &    nodalField)

  {
    // dealii::FEEvaluation<
    //   3,
    //   C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
    //   C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
    //   1,
    //   double>
    //                    fe_evalField(matrixFreeDataObject, 0, 0);
    dealii::FEEvaluation<3, -1> fe_evalField(matrixFreeDataObject, 0, 0);
    const unsigned int          numQuadPoints = fe_evalField.n_q_points;
    nodalField.update_ghost_values();

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeDataObject.get_vector_partitioner(0)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeDataObject.get_quadrature(0).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::VectorizedArray<double> valueVectorized =
      dealii::make_vectorized_array(0.0);
    double value = 0.0;
    for (unsigned int cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField);
        fe_evalField.evaluate(false, true);
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            dealii::VectorizedArray<double> temp =
              scalar_product(fe_evalField.get_gradient(q_point),
                             fe_evalField.get_gradient(q_point));
            fe_evalField.submit_value(temp, q_point);
          }

        valueVectorized += fe_evalField.integrate_value();
        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          value += valueVectorized[iSubCell];
      }

    return dealii::Utilities::MPI::sum(value, mpi_communicator);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::l2ProjectionQuadToNodal(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                      basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       dofHandlerId,
    const unsigned int                       quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                        quadratureValueData,
    distributedCPUVec<double> &nodalField)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
          return quadratureValueData[basisOperationsPtr->cellIndex(cell->id()) *
                                       nQuadsPerCell +
                                     q];
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      basisOperationsPtr->matrixFreeData().get_dof_handler(dofHandlerId),
      constraintMatrix,
      basisOperationsPtr->matrixFreeData().get_quadrature(quadratureId),
      funcRho,
      nodalField);
    constraintMatrix.set_zero(nodalField);
    nodalField.update_ghost_values();
  }
  //
  // compute mass Vector
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeRhoNodalMassVector(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &massVec)
  {
    const unsigned int nLocalDoFs =
      d_matrixFreeDataPRefined
        .get_vector_partitioner(d_densityDofHandlerIndexElectro)
        ->locally_owned_size();
    massVec.clear();
    massVec.resize(nLocalDoFs, 0.0);

    distributedCPUVec<double> distributedMassVec;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      distributedMassVec, d_densityDofHandlerIndexElectro);

    dealii::QGaussLobatto<3> quadrature(
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1);
    dealii::FEValues<3> fe_values(d_dofHandlerRhoNodal.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int  dofs_per_cell =
      (d_dofHandlerRhoNodal.get_fe()).dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          d_constraintsRhoNodal.distribute_local_to_global(massVectorLocal,
                                                           local_dof_indices,
                                                           distributedMassVec);
        }

    distributedMassVec.compress(dealii::VectorOperation::add);
    d_constraintsRhoNodal.set_zero(distributedMassVec);
    for (unsigned int iDoF = 0; iDoF < nLocalDoFs; ++iDoF)
      massVec[iDoF] = distributedMassVec.local_element(iDoF);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeRhoNodalInverseMassVector()
  {
    const unsigned int nLocalDoFs =
      d_matrixFreeDataPRefined
        .get_vector_partitioner(d_densityDofHandlerIndexElectro)
        ->locally_owned_size();
    d_inverseRhoNodalMassVector.clear();
    d_inverseRhoNodalMassVector.resize(nLocalDoFs, 0.0);
    distributedCPUVec<double> distributedMassVec;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      distributedMassVec, d_densityDofHandlerIndexElectro);

    dealii::QGaussLobatto<3> quadrature(
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1);
    dealii::FEValues<3> fe_values(d_dofHandlerRhoNodal.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int  dofs_per_cell =
      (d_dofHandlerRhoNodal.get_fe()).dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          d_constraintsRhoNodal.distribute_local_to_global(massVectorLocal,
                                                           local_dof_indices,
                                                           distributedMassVec);
        }

    distributedMassVec.compress(dealii::VectorOperation::add);
    d_constraintsRhoNodal.set_zero(distributedMassVec);
    for (unsigned int iDoF = 0; iDoF < nLocalDoFs; ++iDoF)
      {
        if (!d_constraintsRhoNodal.is_constrained(iDoF))
          {
            if (std::fabs(distributedMassVec.local_element(iDoF)) > 1E-15)
              d_inverseRhoNodalMassVector[iDoF] =
                1.0 / distributedMassVec.local_element(iDoF);
          }
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeTotalDensityNodalVector(
      const std::map<dealii::CellId, std::vector<double>> &bQuadValues,
      const distributedCPUVec<double> &                    electronDensity,
      distributedCPUVec<double> &                          totalChargeDensity)
  {
    distributedCPUVec<double> smearedCharge;
    smearedCharge.reinit(electronDensity);
    smearedCharge = 0;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    dealii::FEEvaluation<3, -1> fe_eval_sc(d_matrixFreeDataPRefined,
                                           d_densityDofHandlerIndexElectro,
                                           d_smearedChargeQuadratureIdElectro);
    const unsigned int          numQuadPointsSmearedb = fe_eval_sc.n_q_points;
    dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
      numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
    for (unsigned int macrocell = 0;
         macrocell < d_matrixFreeDataPRefined.n_cell_batches();
         ++macrocell)
      {
        std::fill(smearedbQuads.begin(),
                  smearedbQuads.end(),
                  dealii::make_vectorized_array(0.0));
        bool               isMacroCellTrivial = true;
        const unsigned int numSubCells =
          d_matrixFreeDataPRefined.n_active_entries_per_cell_batch(macrocell);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefined.get_cell_iterator(
              macrocell, iSubCell, d_densityDofHandlerIndexElectro);
            dealii::CellId subCellId = subCellPtr->id();
            if (bQuadValues.find(subCellId) == bQuadValues.end())
              continue;
            const std::vector<double> tempVec =
              bQuadValues.find(subCellId)->second;
            if (tempVec.size() != numQuadPointsSmearedb)
              std::cout << "Issue mismatch: " << tempVec.size() << " "
                        << this_mpi_process << std::endl;
            for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
              smearedbQuads[q][iSubCell] = tempVec[q];
            isMacroCellTrivial = false;
          }

        if (!isMacroCellTrivial)
          {
            fe_eval_sc.reinit(macrocell);
            for (unsigned int q = 0; q < fe_eval_sc.n_q_points; ++q)
              {
                fe_eval_sc.submit_value(smearedbQuads[q], q);
              }
            fe_eval_sc.integrate(true, false);
            fe_eval_sc.distribute_local_to_global(smearedCharge);
          }
      }
    MPI_Barrier(d_mpiCommParent);
    smearedCharge.compress(dealii::VectorOperation::add);
    // pcout<<"Smeared chage L2norm: "<<smearedCharge.l2_norm()<<std::endl;
    for (int iDoF = 0; iDoF < d_inverseRhoNodalMassVector.size(); iDoF++)
      {
        totalChargeDensity.local_element(iDoF) =
          smearedCharge.local_element(iDoF) * d_inverseRhoNodalMassVector[iDoF];
      }
    const double chargeNodalCompensation =
      totalCharge(d_matrixFreeDataPRefined, totalChargeDensity);
    const double chargeNodalElectronDensity =
      totalCharge(d_matrixFreeDataPRefined, electronDensity);
    totalChargeDensity += electronDensity;
    const double chargeNodalTotal =
      totalCharge(d_matrixFreeDataPRefined, totalChargeDensity);

    pcout << "Integral total charge " << chargeNodalTotal << " "
          << chargeNodalElectronDensity << " " << chargeNodalCompensation
          << std::endl;
    // std::exit(0);
  }

#include "dft.inst.cc"

} // namespace dftfe
