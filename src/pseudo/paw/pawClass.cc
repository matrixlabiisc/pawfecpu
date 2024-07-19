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
// @author Kartick Ramakrishnan
//
#include <pawClass.h>
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeL0()
  {
    d_bl0QuadValuesAllAtoms.clear();
    const unsigned int numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();

    pcout << "Quadrature rule" << std::endl;
    dealii::QIterated<3> quadratureHigh(dealii::QGauss<1>(
                                          d_dftParamsPtr->QuadratureOrderComp),
                                        d_dftParamsPtr->QuadratureCopyComp);

    pcout << "FE rule" << std::endl;
    dealii::FEValues<3> fe_values(
      d_BasisOperatorElectroHostPtr->matrixFreeData()
        .get_dof_handler(d_BasisOperatorElectroHostPtr->d_dofHandlerID)
        .get_fe(),
      quadratureHigh,
      dealii::update_quadrature_points);

    const unsigned int numberQuadraturePoints = quadratureHigh.size();

    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
                                    sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);

    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int  atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int  Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double           dL0 = d_DeltaL0coeff[Znum];
        // pcout << "Delta dL0: " << dL0 << " Znum: " << Znum << std::endl;
        double RmaxAug = d_RmaxAug[Znum];

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                std::vector<double> &quadvalues =
                  d_bl0QuadValuesAllAtoms[cell->id()];
                if (quadvalues.size() == 0)
                  {
                    quadvalues.resize(numberQuadraturePoints, 0.0);
                  }
                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                     ++iImageAtomCount)
                  {
                    dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                    if (iImageAtomCount == 0)
                      {
                        chargePoint = nuclearCoordinates;
                      }
                    else
                      {
                        chargePoint[0] =
                          imageCoordinates[3 * iImageAtomCount + 0];
                        chargePoint[1] =
                          imageCoordinates[3 * iImageAtomCount + 1];
                        chargePoint[2] =
                          imageCoordinates[3 * iImageAtomCount + 2];
                      }
                    double x[3];
                    double sphericalHarmonicVal, radialVal,
                      sphericalFunctionValue;
                    double r, theta, phi, angle;

                    for (unsigned int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        x[0] = fe_values.quadrature_point(iQuadPoint)[0] -
                               chargePoint[0];
                        x[1] = fe_values.quadrature_point(iQuadPoint)[1] -
                               chargePoint[1];
                        x[2] = fe_values.quadrature_point(iQuadPoint)[2] -
                               chargePoint[2];

                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        if (r <= sphFn->getRadialCutOff())
                          {
                            radialVal = sphFn->getRadialValue(r);
                            quadvalues[iQuadPoint] +=
                              dL0 * radialVal / sqrt(4 * M_PI);



                          } // inside r <= Rmax

                      } // quad loop

                  } // image atom loop
              }     // cell locallyOwned

          } // iElemComp

      } // iAtom
    MPI_Barrier(d_mpiCommParent);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationCharge(
    TypeOfField typeOfField)
  {
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    pcout << "Quadrature rule" << std::endl;
    dealii::QIterated<3> quadratureHigh(dealii::QGauss<1>(
                                          d_dftParamsPtr->QuadratureOrderComp),
                                        d_dftParamsPtr->QuadratureCopyComp);
    const unsigned int   numberQuadraturePoints = quadratureHigh.size();



    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const unsigned int one = 1;
    (*d_bQuadValuesAllAtoms).clear();

    MPI_Barrier(d_mpiCommParent);
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            std::vector<double> &ValueL0 = it->second;
            std::vector<double>  Temp;
            (*d_bQuadValuesAllAtoms).find(it->first)->second.clear();
            // for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
            //      q_point++)
            //   {
            //     Temp.push_back(ValueL0[q_point]);
            //   }
            (*d_bQuadValuesAllAtoms)[it->first] = ValueL0;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        unsigned int Znum   = atomicNumber[atomId];
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        unsigned int        npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> Tij   = d_ProductOfQijShapeFnAtQuadPoints[atomId];
        std::vector<double> Dij   = D_ij[typeOfField][atomId];
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (unsigned int iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            std::vector<double> &quadvalues =
              (*d_bQuadValuesAllAtoms)
                .find(d_BasisOperatorElectroHostPtr->cellID(elementIndex))
                ->second;
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                long unsigned int loc =
                  iElem * npjsq * numberQuadraturePoints + q_point * npjsq;
                quadvalues[q_point] +=
                  ddot_(&npjsq, &Tij[loc], &one, &Dij[0], &one);
              } // q_point

          } // iElem
      }     // iAtom loop
    MPI_Barrier(d_mpiCommParent);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeMemoryOpt(
    TypeOfField typeOfField)
  {
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    pcout << "Quadrature rule" << std::endl;
    dealii::QIterated<3> quadratureHigh(dealii::QGauss<1>(
                                          d_dftParamsPtr->QuadratureOrderComp),
                                        d_dftParamsPtr->QuadratureCopyComp);
    const unsigned int   numberQuadraturePoints = quadratureHigh.size();



    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const unsigned int one = 1;
    (*d_bQuadValuesAllAtoms).clear();

    MPI_Barrier(d_mpiCommParent);
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            std::vector<double> &ValueL0 = it->second;
            std::vector<double>  Temp;
            (*d_bQuadValuesAllAtoms).find(it->first)->second.clear();
            // for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
            //      q_point++)
            //   {
            //     Temp.push_back(ValueL0[q_point]);
            //   }
            (*d_bQuadValuesAllAtoms)[it->first] = ValueL0;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();
    MPI_Barrier(d_mpiCommParent);
    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        unsigned int Znum   = atomicNumber[atomId];
        unsigned int numberOfProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        unsigned int        npjsq = numberOfProjectors * numberOfProjectors;
        std::vector<double> gLValuesAtQuadPoints =
          d_shapeFnAtQuadPoints[atomId];
        std::vector<double> Dij = D_ij[typeOfField][atomId];
        std::vector<double> Tij = d_productOfMultipoleClebshGordon[Znum];
        unsigned int        numShapeFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        std::vector<double> deltaL(numShapeFunctions, 0.0);
        const char          transA = 'N', transB = 'N';
        const double        Alpha = 1.0, Beta = 0.0;
        const unsigned int  inc = 1;

        dgemm_(&transA,
               &transB,
               &inc,
               &numShapeFunctions,
               &npjsq,
               &Alpha,
               &Dij[0],
               &inc,
               &Tij[0],
               &npjsq,
               &Beta,
               &deltaL[0],
               &inc);



        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        for (unsigned int iElem = 0; iElem < numberElementsInAtomCompactSupport;
             iElem++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElem];
            std::vector<double> &quadvalues =
              (*d_bQuadValuesAllAtoms)
                .find(d_BasisOperatorElectroHostPtr->cellID(elementIndex))
                ->second;
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                long unsigned int loc =
                  iElem * numShapeFunctions * numberQuadraturePoints +
                  q_point * numShapeFunctions;
                quadvalues[q_point] += ddot_(&numShapeFunctions,
                                             &gLValuesAtQuadPoints[loc],
                                             &one,
                                             &deltaL[0],
                                             &one);

              } // q_point

          } // iElem
      }     // iAtom loop
    MPI_Barrier(d_mpiCommParent);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeCoeffMemoryOpt()
  {
    std::vector<double> IntegralValue(9, 0.0);
    const unsigned int  numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    pcout << "Quadrature rule" << std::endl;
    dealii::QIterated<3> quadratureHigh(dealii::QGauss<1>(
                                          d_dftParamsPtr->QuadratureOrderComp),
                                        d_dftParamsPtr->QuadratureCopyComp);

    pcout << "FE rule" << std::endl;
    dealii::FEValues<3> fe_values(
      d_BasisOperatorElectroHostPtr->matrixFreeData()
        .get_dof_handler(d_BasisOperatorElectroHostPtr->d_dofHandlerID)
        .get_fe(),
      quadratureHigh,
      dealii::update_JxW_values | dealii::update_quadrature_points);
    d_jxwcompensationCharge.clear();
    const unsigned int numberQuadraturePoints = quadratureHigh.size();
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            const unsigned int cellIndex =
              d_BasisOperatorElectroHostPtr->d_cellIdToCellIndexMap[it->first];
            dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(cellIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                std::vector<double> jxw(numberQuadraturePoints, 0.0);
                for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                     iQuad++)
                  {
                    jxw[iQuad] = fe_values.JxW(iQuad);
                  }
                d_jxwcompensationCharge[it->first] = jxw;


              } // cell
          }     // it
      }         // if



    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      projectorFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();

    MPI_Barrier(d_mpiCommParent);

    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int  atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int  Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3>    nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double              RmaxAug   = d_RmaxAug[Znum];
        std::vector<double> multipole = d_multipole[Znum];
        const unsigned int  NumRadialSphericalFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        const unsigned int NumProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const unsigned int NumRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        std::vector<double> tempCoeff(numberElementsInAtomCompactSupport *
                                        numberQuadraturePoints *
                                        NumTotalSphericalFunctions,
                                      0.0);

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                unsigned int        Lindex = 0;
                std::vector<double> gLValuesQuadPoints(
                  numberQuadraturePoints * NumTotalSphericalFunctions, 0.0);


                for (unsigned int alpha = 0;
                     alpha < NumRadialSphericalFunctions;
                     ++alpha)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                      sphericalFunction.find(std::make_pair(Znum, alpha))
                        ->second;
                    int lQuantumNumber = sphFn->getQuantumNumberl();

                    for (int mQuantumNumber = -lQuantumNumber;
                         mQuantumNumber <= lQuantumNumber;
                         mQuantumNumber++)
                      {
                        for (int iImageAtomCount = 0;
                             iImageAtomCount < imageIdsSize;
                             ++iImageAtomCount)
                          {
                            dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                            if (iImageAtomCount == 0)
                              {
                                chargePoint = nuclearCoordinates;
                              }
                            else
                              {
                                chargePoint[0] =
                                  imageCoordinates[3 * iImageAtomCount + 0];
                                chargePoint[1] =
                                  imageCoordinates[3 * iImageAtomCount + 1];
                                chargePoint[2] =
                                  imageCoordinates[3 * iImageAtomCount + 2];
                              }
                            double x[3];
                            double sphericalHarmonicVal, radialVal,
                              sphericalFunctionValue;
                            double r, theta, phi, angle;

                            for (int iQuadPoint = 0;
                                 iQuadPoint < numberQuadraturePoints;
                                 ++iQuadPoint)
                              {
                                x[0] =
                                  fe_values.quadrature_point(iQuadPoint)[0] -
                                  chargePoint[0];
                                x[1] =
                                  fe_values.quadrature_point(iQuadPoint)[1] -
                                  chargePoint[1];
                                x[2] =
                                  fe_values.quadrature_point(iQuadPoint)[2] -
                                  chargePoint[2];
                                sphericalHarmonicUtils::
                                  convertCartesianToSpherical(x, r, theta, phi);
                                sphericalHarmonicUtils::getSphericalHarmonicVal(
                                  theta,
                                  phi,
                                  lQuantumNumber,
                                  mQuantumNumber,
                                  sphericalHarmonicVal);
                                if (r <= sphFn->getRadialCutOff())
                                  {
                                    radialVal = sphFn->getRadialValue(r);
                                    sphericalFunctionValue =
                                      radialVal * sphericalHarmonicVal;
                                    long unsigned int loc =
                                      iElemComp * (numberQuadraturePoints *
                                                   NumTotalSphericalFunctions) +
                                      iQuadPoint * NumTotalSphericalFunctions +
                                      Lindex;
                                    tempCoeff[loc] += sphericalFunctionValue;
                                    gLValuesQuadPoints
                                      [Lindex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      fe_values.JxW(iQuadPoint) *
                                      sphericalFunctionValue;
                                    IntegralValue[Lindex] +=
                                      sphericalFunctionValue *
                                      fe_values.JxW(iQuadPoint) *
                                      pow(r, lQuantumNumber) *
                                      sphericalHarmonicVal;
                                  } // inside r <= Rmax


                              } // quad loop

                          } // image atom loop

                        Lindex++;
                      } // mQuantumNumber
                  }     // alpha
                d_gLValuesQuadPoints[std::make_pair(atomId, elementIndex)] =
                  gLValuesQuadPoints;
              }
          } // iElemComp

        d_shapeFnAtQuadPoints[atomId] = tempCoeff;


      } // iAtom
    for (int L = 0; L < IntegralValue.size(); L++)
      {
        double value =
          dealii::Utilities::MPI::sum(IntegralValue[L], d_mpiCommParent);
        pcout << "PAW: Integral of the shapefn " << L << " is: " << value << " "
              << " with error: " << (value - atomicNumber.size()) << std::endl;
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeCompensationChargeCoeff()
  {
    std::vector<double> IntegralValue(8, 0.0);
    const unsigned int  numberNodesPerElement =
      d_BasisOperatorElectroHostPtr->nDofsPerCell();
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);
    const unsigned int numberAtomsOfInterest =
      d_atomicShapeFnsContainer->getNumAtomCentersSize();
    pcout << "Quadrature rule" << std::endl;
    dealii::QIterated<3> quadratureHigh(dealii::QGauss<1>(
                                          d_dftParamsPtr->QuadratureOrderComp),
                                        d_dftParamsPtr->QuadratureCopyComp);

    pcout << "FE rule" << std::endl;
    dealii::FEValues<3> fe_values(
      d_BasisOperatorElectroHostPtr->matrixFreeData()
        .get_dof_handler(d_BasisOperatorElectroHostPtr->d_dofHandlerID)
        .get_fe(),
      quadratureHigh,
      dealii::update_JxW_values | dealii::update_quadrature_points);
    d_jxwcompensationCharge.clear();
    const unsigned int numberQuadraturePoints = quadratureHigh.size();
    if (d_bl0QuadValuesAllAtoms.size() > 0)
      {
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               d_bl0QuadValuesAllAtoms.begin();
             it != d_bl0QuadValuesAllAtoms.end();
             ++it)
          {
            const unsigned int cellIndex =
              d_BasisOperatorElectroHostPtr->d_cellIdToCellIndexMap[it->first];
            dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(cellIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                std::vector<double> jxw(numberQuadraturePoints, 0.0);
                for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                     iQuad++)
                  {
                    jxw[iQuad] = fe_values.JxW(iQuad);
                  }
                d_jxwcompensationCharge[it->first] = jxw;


              } // cell
          }     // it
      }         // if



    const std::vector<unsigned int> &atomicNumber =
      d_atomicShapeFnsContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomicShapeFnsContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomicShapeFnsContainer->getPeriodicImageCoordinatesList();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction = d_atomicShapeFnsContainer->getSphericalFunctions();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      projectorFunction =
        d_atomicProjectorFnsContainer->getSphericalFunctions();
    const std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomicShapeFnsContainer->getAtomIdsInCurrentProcess();

    MPI_Barrier(d_mpiCommParent);

    for (unsigned int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
         iAtom++)
      {
        const unsigned int  atomId = atomIdsInCurrentProcess[iAtom];
        const unsigned int  Znum   = atomicNumber[atomId];
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int imageIdsSize = imageCoordinates.size() / 3;
        std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
          sphericalFunction.find(std::make_pair(Znum, 0))->second;
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomicShapeFnsContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        dealii::Point<3>    nuclearCoordinates(atomCoordinates[3 * atomId + 0],
                                            atomCoordinates[3 * atomId + 1],
                                            atomCoordinates[3 * atomId + 2]);
        double              RmaxAug   = d_RmaxAug[Znum];
        std::vector<double> multipole = d_multipole[Znum];
        const unsigned int  NumRadialSphericalFunctions =
          d_atomicShapeFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int NumTotalSphericalFunctions =
          d_atomicShapeFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
            Znum);
        const unsigned int NumProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const unsigned int NumRadialProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int  numProjSq = NumProjectors * NumProjectors;
        std::vector<double> tempCoeff(numberElementsInAtomCompactSupport *
                                        numberQuadraturePoints * numProjSq,
                                      0.0);

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             iElemComp++)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              d_BasisOperatorElectroHostPtr->getCellIterator(elementIndex);
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                unsigned int        Lindex = 0;
                std::vector<double> gLValuesQuadPoints(
                  numberQuadraturePoints * NumTotalSphericalFunctions, 0.0);


                for (unsigned int alpha = 0;
                     alpha < NumRadialSphericalFunctions;
                     ++alpha)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                      sphericalFunction.find(std::make_pair(Znum, alpha))
                        ->second;
                    int lQuantumNumber = sphFn->getQuantumNumberl();

                    for (int mQuantumNumber = -lQuantumNumber;
                         mQuantumNumber <= lQuantumNumber;
                         mQuantumNumber++)
                      {
                        for (int iImageAtomCount = 0;
                             iImageAtomCount < imageIdsSize;
                             ++iImageAtomCount)
                          {
                            dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                            if (iImageAtomCount == 0)
                              {
                                chargePoint = nuclearCoordinates;
                              }
                            else
                              {
                                chargePoint[0] =
                                  imageCoordinates[3 * iImageAtomCount + 0];
                                chargePoint[1] =
                                  imageCoordinates[3 * iImageAtomCount + 1];
                                chargePoint[2] =
                                  imageCoordinates[3 * iImageAtomCount + 2];
                              }
                            double x[3];
                            double sphericalHarmonicVal, radialVal,
                              sphericalFunctionValue;
                            double r, theta, phi, angle;

                            for (int iQuadPoint = 0;
                                 iQuadPoint < numberQuadraturePoints;
                                 ++iQuadPoint)
                              {
                                x[0] =
                                  fe_values.quadrature_point(iQuadPoint)[0] -
                                  chargePoint[0];
                                x[1] =
                                  fe_values.quadrature_point(iQuadPoint)[1] -
                                  chargePoint[1];
                                x[2] =
                                  fe_values.quadrature_point(iQuadPoint)[2] -
                                  chargePoint[2];
                                // x[0] =
                                //   quadraturePointsVector
                                //     [elementIndex * numberQuadraturePoints *
                                //     3 +
                                //      3 * iQuadPoint] -
                                //   chargePoint[0];
                                // x[1] =
                                //   quadraturePointsVector
                                //     [elementIndex * numberQuadraturePoints *
                                //     3 +
                                //      3 * iQuadPoint + 1] -
                                //   chargePoint[1];
                                // x[2] =
                                //   quadraturePointsVector
                                //     [elementIndex * numberQuadraturePoints *
                                //     3 +
                                //      3 * iQuadPoint + 2] -
                                //   chargePoint[2];
                                sphericalHarmonicUtils::
                                  convertCartesianToSpherical(x, r, theta, phi);
                                sphericalHarmonicUtils::getSphericalHarmonicVal(
                                  theta,
                                  phi,
                                  lQuantumNumber,
                                  mQuantumNumber,
                                  sphericalHarmonicVal);
                                if (r <= sphFn->getRadialCutOff())
                                  {
                                    radialVal = sphFn->getRadialValue(r);
                                    sphericalFunctionValue =
                                      radialVal * sphericalHarmonicVal;

                                    unsigned int alpha_i = 0;
                                    for (int i = 0; i < NumRadialProjectors;
                                         i++)
                                      {
                                        std::shared_ptr<
                                          AtomCenteredSphericalFunctionBase>
                                          projFnI =
                                            projectorFunction
                                              .find(std::make_pair(Znum, i))
                                              ->second;
                                        int l_i = projFnI->getQuantumNumberl();
                                        for (int m_i = -l_i; m_i <= l_i; m_i++)
                                          {
                                            unsigned int alpha_j = 0;
                                            for (int j = 0;
                                                 j < NumRadialProjectors;
                                                 j++)
                                              {
                                                std::shared_ptr<
                                                  AtomCenteredSphericalFunctionBase>
                                                  projFnJ =
                                                    projectorFunction
                                                      .find(
                                                        std::make_pair(Znum, j))
                                                      ->second;
                                                int l_j =
                                                  projFnJ->getQuantumNumberl();
                                                for (int m_j = -l_j; m_j <= l_j;
                                                     m_j++)
                                                  {
                                                    double multipolevalue =
                                                      multipole
                                                        [lQuantumNumber *
                                                           NumRadialProjectors *
                                                           NumRadialProjectors +
                                                         i *
                                                           NumRadialProjectors +
                                                         j];
                                                    double Cijl =
                                                      gaunt(l_i,
                                                            l_j,
                                                            lQuantumNumber,
                                                            m_i,
                                                            m_j,
                                                            mQuantumNumber);
                                                    long unsigned int loc =
                                                      iElemComp *
                                                        (numberQuadraturePoints *
                                                         numProjSq) +
                                                      iQuadPoint * (numProjSq) +
                                                      alpha_i * NumProjectors +
                                                      alpha_j;
                                                    bool flag = true;
                                                    if (std::fabs(
                                                          multipolevalue) <
                                                          1E-16 ||
                                                        std::fabs(Cijl) < 1E-16)
                                                      flag = false;



                                                    if (flag)
                                                      {
                                                        if (r <= RmaxAug)
                                                          {
                                                            tempCoeff[loc] +=
                                                              Cijl *
                                                              multipolevalue *
                                                              sphericalFunctionValue;
                                                          }
                                                        else
                                                          tempCoeff[loc] += 0.0;
                                                      }

                                                    alpha_j++;
                                                  } // m_j
                                              }     // j

                                            alpha_i++;
                                          } // m_i
                                      }     // i loop
                                    gLValuesQuadPoints
                                      [Lindex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      fe_values.JxW(iQuadPoint) *
                                      sphericalFunctionValue;
                                    IntegralValue[Lindex] +=
                                      sphericalFunctionValue *
                                      fe_values.JxW(iQuadPoint) *
                                      pow(r, lQuantumNumber) *
                                      sphericalHarmonicVal;
                                  } // inside r <= Rmax


                              } // quad loop

                          } // image atom loop

                        Lindex++;
                      } // mQuantumNumber
                  }     // alpha
                d_gLValuesQuadPoints[std::make_pair(atomId, elementIndex)] =
                  gLValuesQuadPoints;
              }
          } // iElemComp

        d_ProductOfQijShapeFnAtQuadPoints[atomId] = tempCoeff;


      } // iAtom
    for (int L = 0; L < IntegralValue.size(); L++)
      {
        double value =
          dealii::Utilities::MPI::sum(IntegralValue[L], d_mpiCommParent);
        pcout << "PAW: Integral of the shapefn " << L << " is: " << value << " "
              << " with error: " << (value - atomicNumber.size()) << std::endl;
      }
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  pawClass<ValueType, memorySpace>::getCouplingMatrix(CouplingType couplingtype)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        if (couplingtype == CouplingType::HamiltonianEntries)
          {
            if (!d_HamiltonianCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                                                couplingEntriesHost;
                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    unsigned int atomId = atomIdsInProcessor[iAtom];
                    unsigned int Zno    = atomicNumber[atomId];
                    unsigned int numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (unsigned int alpha_i = 0;
                         alpha_i < numberSphericalFunctions;
                         alpha_i++)
                      {
                        for (unsigned int alpha_j = 0;
                             alpha_j < numberSphericalFunctions;
                             alpha_j++)
                          {
                            unsigned int index =
                              alpha_i * numberSphericalFunctions + alpha_j;
                            ValueType V =
                              d_atomicNonLocalPseudoPotentialConstants
                                [CouplingType::HamiltonianEntries][atomId]
                                [index];
                            Entries.push_back(V);
                          }
                      }
                  }
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries[CouplingType::HamiltonianEntries] =
                  couplingEntriesHost;
                d_HamiltonianCouplingMatrixEntriesUpdated = true;
              }
          }
        else if (couplingtype == CouplingType::pawOverlapEntries)
          {
            if (!d_overlapCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                                                couplingEntriesHost;
                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
                  {
                    unsigned int atomId = atomIdsInProcessor[iAtom];
                    unsigned int Zno    = atomicNumber[atomId];
                    unsigned int numberSphericalFunctions =
                      d_atomicProjectorFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                    for (unsigned int alpha_i = 0;
                         alpha_i < numberSphericalFunctions;
                         alpha_i++)
                      {
                        for (unsigned int alpha_j = 0;
                             alpha_j < numberSphericalFunctions;
                             alpha_j++)
                          {
                            unsigned int index =
                              alpha_i * numberSphericalFunctions + alpha_j;
                            ValueType V =
                              d_atomicNonLocalPseudoPotentialConstants
                                [CouplingType::pawOverlapEntries][Zno][index];
                            Entries.push_back(V);
                          }
                      }
                  }
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries[CouplingType::pawOverlapEntries] =
                  couplingEntriesHost;
                d_overlapCouplingMatrixEntriesUpdated = true;
              }
          }
        else if (couplingtype == CouplingType::inversePawOverlapEntries)
          {
            if (!d_inverseCouplingMatrixEntriesUpdated)
              {
                std::vector<ValueType> Entries;
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHost;

                const std::vector<unsigned int> atomIdsInProcessor =
                  d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
                std::vector<unsigned int> atomicNumber =
                  d_atomicProjectorFnsContainer->getAtomicNumbers();
                for (int kPoint = 0; kPoint < d_kpointWeights.size(); kPoint++)
                  {
                    for (int iAtom = 0; iAtom < atomIdsInProcessor.size();
                         iAtom++)
                      {
                        unsigned int atomId = atomIdsInProcessor[iAtom];
                        unsigned int Zno    = atomicNumber[atomId];
                        unsigned int numberSphericalFunctions =
                          d_atomicProjectorFnsContainer
                            ->getTotalNumberOfSphericalFunctionsPerAtom(Zno);
                        for (unsigned int alpha_i = 0;
                             alpha_i < numberSphericalFunctions;
                             alpha_i++)
                          {
                            for (unsigned int alpha_j = 0;
                                 alpha_j < numberSphericalFunctions;
                                 alpha_j++)
                              {
                                unsigned int index =
                                  alpha_i * numberSphericalFunctions + alpha_j;
                                ValueType V =
                                  d_atomicNonLocalPseudoPotentialConstants
                                    [CouplingType::inversePawOverlapEntries]
                                    [atomId][kPoint * numberSphericalFunctions *
                                               numberSphericalFunctions +
                                             index];
                                Entries.push_back(V);
                              } // alpha_j
                          }     // alpha_i
                      }         // iAtom
                  }             // kPoint
                couplingEntriesHost.resize(Entries.size());
                couplingEntriesHost.copyFrom(Entries);
                d_couplingMatrixEntries
                  [CouplingType::inversePawOverlapEntries] =
                    couplingEntriesHost;
                d_inverseCouplingMatrixEntriesUpdated = true;
              }
          }
      }
    else
      {
        AssertThrow(dftfe::utils::MemorySpace::HOST == memorySpace,
                    dealii::ExcMessage(
                      "DFT-FE Error: Not yet implemented on GPUs."));
      }

    return d_couplingMatrixEntries[couplingtype];
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<
    typename dftfe::dataTypes::singlePrecType<ValueType>::type,
    memorySpace> &
  pawClass<ValueType, memorySpace>::getCouplingMatrixSinglePrec(
    CouplingType couplingtype)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        if (couplingtype == CouplingType::HamiltonianEntries)
          {
            if (!d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec)
              {
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHost =
                    getCouplingMatrix(CouplingType::HamiltonianEntries);
                dftfe::utils::MemoryStorage<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                  dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHostSinglePrec;
                couplingEntriesHostSinglePrec.resize(
                  couplingEntriesHost.size());
                d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                  couplingEntriesHostSinglePrec.size(),
                  couplingEntriesHost.data(),
                  couplingEntriesHostSinglePrec.data());

                d_couplingMatrixEntriesSinglePrec
                  [CouplingType::HamiltonianEntries] =
                    couplingEntriesHostSinglePrec;
                d_HamiltonianCouplingMatrixEntriesUpdatedSinglePrec = true;
              }
          }
        else if (couplingtype == CouplingType::pawOverlapEntries)
          {
            if (!d_overlapCouplingMatrixEntriesUpdatedSinglePrec)
              {
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHost =
                    getCouplingMatrix(CouplingType::pawOverlapEntries);
                dftfe::utils::MemoryStorage<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                  dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHostSinglePrec;
                couplingEntriesHostSinglePrec.resize(
                  couplingEntriesHost.size());
                d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                  couplingEntriesHostSinglePrec.size(),
                  couplingEntriesHost.data(),
                  couplingEntriesHostSinglePrec.data());

                d_couplingMatrixEntriesSinglePrec
                  [CouplingType::pawOverlapEntries] =
                    couplingEntriesHostSinglePrec;
                d_overlapCouplingMatrixEntriesUpdatedSinglePrec = true;
              }
          }
        else if (couplingtype == CouplingType::inversePawOverlapEntries)
          {
            if (!d_inverseCouplingMatrixEntriesUpdatedSinglePrec)
              {
                dftfe::utils::MemoryStorage<ValueType,
                                            dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHost =
                    getCouplingMatrix(CouplingType::inversePawOverlapEntries);
                dftfe::utils::MemoryStorage<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                  dftfe::utils::MemorySpace::HOST>
                  couplingEntriesHostSinglePrec;
                couplingEntriesHostSinglePrec.resize(
                  couplingEntriesHost.size());
                d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
                  couplingEntriesHostSinglePrec.size(),
                  couplingEntriesHost.data(),
                  couplingEntriesHostSinglePrec.data());

                d_couplingMatrixEntriesSinglePrec
                  [CouplingType::inversePawOverlapEntries] =
                    couplingEntriesHostSinglePrec;
                d_inverseCouplingMatrixEntriesUpdatedSinglePrec = true;
              }
          }
      }
    else
      {
        AssertThrow(dftfe::utils::MemorySpace::HOST == memorySpace,
                    dealii::ExcMessage(
                      "DFT-FE Error: Not yet implemented on GPUs."));
      }

    return d_couplingMatrixEntriesSinglePrec[couplingtype];
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  pawClass<ValueType, memorySpace>::computeDij(
    const bool         isDijOut,
    const unsigned int startVectorIndex,
    const unsigned int vectorBlockSize,
    const unsigned int spinIndex,
    const unsigned int kpointIndex)
  {
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char transA = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {
        const unsigned int atomId = atomIdsInProcessor[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numberSphericalFunctions =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        if (startVectorIndex == 0)
          {
            D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId] =
              std::vector<double>(numberSphericalFunctions *
                                    numberSphericalFunctions,
                                  0.0);
          }
        std::vector<ValueType> tempDij(numberSphericalFunctions *
                                         numberSphericalFunctions,
                                       0.0);

        if (d_verbosity >= 5)
          {
            std::cout << "U Matrix Entries" << std::endl;
            for (int i = 0; i < numberSphericalFunctions * vectorBlockSize; i++)
              std::cout << *(d_nonLocalOperator
                               ->getCconjtansXLocalDataStructure(atomId) +
                             i)
                        << std::endl;
          }
        d_BLASWrapperHostPtr->xgemm(
          transA,
          transB,
          numberSphericalFunctions,
          numberSphericalFunctions,
          vectorBlockSize,
          &alpha,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(atomId),
          vectorBlockSize,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(atomId),
          vectorBlockSize,
          &beta,
          &tempDij[0],
          numberSphericalFunctions);
        std::transform(
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data() +
            numberSphericalFunctions * numberSphericalFunctions,
          tempDij.data(),
          D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId].data(),
          [](auto &p, auto &q) { return p + dftfe::utils::realPart(q); });
        // pcout << "DEBUG: PAW Dij size: "
        //       << D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In].size()
        //       << std::endl;
      }
  }
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class pawClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
