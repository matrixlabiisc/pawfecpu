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
// @author Shiva Rudraraju, Phani Motamarri
//

//
// source file for generating image atoms
//

//
// round a given fractional coordinate to zero or 1
//
#include <dft.h>
#include <linearAlgebraOperations.h>
#include <dftUtils.h>
#include <vectorUtilities.h>

namespace dftfe
{
  namespace internaldft
  {
    double
    roundToCell(double frac)
    {
      double returnValue = 0;
      if (frac < 0)
        returnValue = 0;
      else if (frac >= 0 && frac <= 1)
        returnValue = frac;
      else
        returnValue = 1;

      return returnValue;
    }

    //
    // cross product
    //
    std::vector<double>
    cross(const std::vector<double> &v1, const std::vector<double> &v2)
    {
      assert(v1.size() == 3);
      assert(v2.size() == 3);

      std::vector<double> returnValue(3);

      returnValue[0] = v1[1] * v2[2] - v1[2] * v2[1];
      returnValue[1] = -v1[0] * v2[2] + v2[0] * v1[2];
      returnValue[2] = v1[0] * v2[1] - v2[0] * v1[1];
      return returnValue;
    }

    //
    // given surface defined by normal = surfaceNormal and a point = xred2
    // find the point on this surface closest to an arbitrary point = xred1
    // return fractional coordinates of nearest point
    //
    std::vector<double>
    getNearestPointOnGivenSurface(std::vector<double>        latticeVectors,
                                  const std::vector<double> &xred1,
                                  const std::vector<double> &xred2,
                                  const std::vector<double> &surfaceNormal)

    {
      //
      // get real space coordinates for xred1 and xred2
      //
      std::vector<double> P(3, 0.0);
      std::vector<double> Q(3, 0.0);
      std::vector<double> R(3);

      for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
            {
              P[i] += latticeVectors[3 * j + i] * xred1[j];
              Q[i] += latticeVectors[3 * j + i] * xred2[j];
            }
          R[i] = Q[i] - P[i];
        }

      //
      // fine nearest point on the plane defined by surfaceNormal and xred2
      //
      double num = R[0] * surfaceNormal[0] + R[1] * surfaceNormal[1] +
                   R[2] * surfaceNormal[2];
      double denom = surfaceNormal[0] * surfaceNormal[0] +
                     surfaceNormal[1] * surfaceNormal[1] +
                     surfaceNormal[2] * surfaceNormal[2];
      const double t = num / denom;


      std::vector<double> nearestPtCoords(3);
      for (int i = 0; i < 3; ++i)
        nearestPtCoords[i] = P[i] + t * surfaceNormal[i];

      //
      // get fractional coordinates for the nearest point : solve a system
      // of equations
      int N    = 3;
      int NRHS = 1;
      int LDA  = 3;
      int IPIV[3];
      int info;


      dgesv_(&N,
             &NRHS,
             &latticeVectors[0],
             &LDA,
             &IPIV[0],
             &nearestPtCoords[0],
             &LDA,
             &info);


      if (info != 0)
        {
          std::cout << "LU solve in conversion of frac to real coords failed."
                    << std::endl;
          exit(-1);
        }

      //
      // nearestPtCoords is overwritten with the solution = frac coords
      //

      std::vector<double> returnValue(3);

      for (int i = 0; i < 3; ++i)
        returnValue[i] = roundToCell(nearestPtCoords[i]);

      return returnValue;
    }

    //
    // input : xreduced = frac coords of image charge
    // output : min distance to any of the cel surfaces
    //
    double
    getMinDistanceFromImageToCell(const std::vector<double> &latticeVectors,
                                  const std::vector<double> &xreduced)
    {
      const double xfrac = xreduced[0];
      const double yfrac = xreduced[1];
      const double zfrac = xreduced[2];

      //
      // if interior point, then return 0 distance
      //
      if (xfrac >= 0 && xfrac <= 1 && yfrac >= 0 && yfrac <= 1 && zfrac >= 0 &&
          zfrac <= 1)
        return 0;
      else
        {
          //
          // extract lattice vectors and define surface normals
          //
          const std::vector<double> a(&latticeVectors[0],
                                      &latticeVectors[0] + 3);
          const std::vector<double> b(&latticeVectors[3],
                                      &latticeVectors[3] + 3);
          const std::vector<double> c(&latticeVectors[6],
                                      &latticeVectors[6] + 3);

          std::vector<double> surface1Normal = cross(b, c);
          std::vector<double> surface2Normal = cross(c, a);
          std::vector<double> surface3Normal = cross(a, b);

          std::vector<double> surfacePoint(3);
          std::vector<double> dFrac(3);
          std::vector<double> dReal(3);

          //
          // find closest distance to surface 1
          //
          surfacePoint[0] = 0;
          surfacePoint[1] = yfrac;
          surfacePoint[2] = zfrac;

          std::vector<double> fracPtA = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface1Normal);
          //
          // compute distance between fracPtA (closest point on surface A) and
          // xreduced
          //
          for (int i = 0; i < 3; ++i)
            dFrac[i] = xreduced[i] - fracPtA[i];

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distA =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distA = sqrt(distA);

          //
          // find closest distance to surface 2
          //
          surfacePoint[0] = xfrac;
          surfacePoint[1] = 0;
          surfacePoint[2] = zfrac;

          std::vector<double> fracPtB = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface2Normal);

          for (int i = 0; i < 3; ++i)
            {
              dFrac[i] = xreduced[i] - fracPtB[i];
              dReal[i] = 0.0;
            }

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distB =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distB = sqrt(distB);

          //
          // find min distance to surface 3
          //
          surfacePoint[0] = xfrac;
          surfacePoint[1] = yfrac;
          surfacePoint[2] = 0;

          std::vector<double> fracPtC = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface3Normal);

          for (int i = 0; i < 3; ++i)
            {
              dFrac[i] = xreduced[i] - fracPtC[i];
              dReal[i] = 0.0;
            }

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distC =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distC = sqrt(distC);

          //
          // fine min distance to surface 4
          //
          surfacePoint[0] = 1;
          surfacePoint[1] = yfrac;
          surfacePoint[2] = zfrac;

          std::vector<double> fracPtD = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface1Normal);

          for (int i = 0; i < 3; ++i)
            {
              dFrac[i] = xreduced[i] - fracPtD[i];
              dReal[i] = 0.0;
            }

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distD =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distD = sqrt(distD);

          //
          // find min distance to surface 5
          //
          surfacePoint[0] = xfrac;
          surfacePoint[1] = 1;
          surfacePoint[2] = zfrac;

          std::vector<double> fracPtE = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface2Normal);

          for (int i = 0; i < 3; ++i)
            {
              dFrac[i] = xreduced[i] - fracPtE[i];
              dReal[i] = 0.0;
            }

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distE =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distE = sqrt(distE);


          //
          // find min distance to surface 6
          //
          surfacePoint[0] = xfrac;
          surfacePoint[1] = yfrac;
          surfacePoint[2] = 1;

          std::vector<double> fracPtF = getNearestPointOnGivenSurface(
            latticeVectors, xreduced, surfacePoint, surface3Normal);

          for (int i = 0; i < 3; ++i)
            {
              dFrac[i] = xreduced[i] - fracPtF[i];
              dReal[i] = 0.0;
            }

          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              dReal[i] += latticeVectors[3 * j + i] * dFrac[j];

          double distF =
            dReal[0] * dReal[0] + dReal[1] * dReal[1] + dReal[2] * dReal[2];
          distF = sqrt(distF);

          return std::min(
            distF,
            std::min(distE,
                     std::min(distD, std::min(distC, std::min(distB, distA)))));
        }
    }
  } // namespace internaldft

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::generateImageCharges(
    const double                      pspCutOff,
    std::vector<int> &                imageIds,
    std::vector<double> &             imageCharges,
    std::vector<std::vector<double>> &imagePositions)
  {
    const double tol       = 1e-4;
    const bool   periodicX = d_dftParamsPtr->periodicX;
    const bool   periodicY = d_dftParamsPtr->periodicY;
    const bool   periodicZ = d_dftParamsPtr->periodicZ;

    //
    // get the magnitude of lattice Vectors
    //
    double magnitude1 =
      sqrt(d_domainBoundingVectors[0][0] * d_domainBoundingVectors[0][0] +
           d_domainBoundingVectors[0][1] * d_domainBoundingVectors[0][1] +
           d_domainBoundingVectors[0][2] * d_domainBoundingVectors[0][2]);
    double magnitude2 =
      sqrt(d_domainBoundingVectors[1][0] * d_domainBoundingVectors[1][0] +
           d_domainBoundingVectors[1][1] * d_domainBoundingVectors[1][1] +
           d_domainBoundingVectors[1][2] * d_domainBoundingVectors[1][2]);
    double magnitude3 =
      sqrt(d_domainBoundingVectors[2][0] * d_domainBoundingVectors[2][0] +
           d_domainBoundingVectors[2][1] * d_domainBoundingVectors[2][1] +
           d_domainBoundingVectors[2][2] * d_domainBoundingVectors[2][2]);

    //
    // get the maximum of the magnitudes
    //
    double minMagnitude = magnitude1;
    if (magnitude1 >= magnitude2)
      minMagnitude = magnitude2;
    else if (magnitude1 >= magnitude3)
      minMagnitude = magnitude3;

    if (magnitude2 >= magnitude3)
      {
        if (magnitude1 >= magnitude3)
          minMagnitude = magnitude3;
      }

    //
    // compute the ratio between pspCutOff and maxMagnitude and multiply by a
    // factor 2 to decide number of image atom layers
    //
    double ratio        = pspCutOff / minMagnitude;
    int    numberLayers = std::ceil(ratio * 2);



    //
    // get origin/centroid of the cell
    //
    std::vector<double> shift(3, 0.0);

    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            shift[i] += d_domainBoundingVectors[j][i] / 2.0;
          }
      }

    std::vector<double> latticeVectors(9, 0.0);
    int                 count = 0;
    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            latticeVectors[count] = d_domainBoundingVectors[i][j];
            count++;
          }
      }

    imageIds.clear();
    imagePositions.clear();
    imageCharges.clear();

    std::vector<int>    imageIdsKptPool;
    std::vector<int>    imageIdsGathered;
    std::vector<double> imagePositionsFlattenedKptPool;
    std::vector<double> imagePositionsFlattened;

    // kpoint group parallelization data structures
    const unsigned int numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);

    // std::cout<<"hello: "<<numberKptGroups<<std::endl;
    const unsigned int kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    std::vector<int> kptGroupLowHighPlusOneIndices;
    dftUtils::createKpointParallelizationIndices(interpoolcomm,
                                                 atomLocations.size(),
                                                 kptGroupLowHighPlusOneIndices);
    // std::cout<<"hello2: "<<kptGroupLowHighPlusOneIndices[2 *
    // kptGroupTaskId]<<std::endl; std::cout<<"hello2:
    // "<<kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1]<<std::endl;
    for (int i = 0; i < atomLocations.size(); ++i)
      {
        if (i < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
            i >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
          {
            const int    iCharge = i;
            const double fracX   = atomLocations[i][2];
            const double fracY   = atomLocations[i][3];
            const double fracZ   = atomLocations[i][4];

            int izmin = -numberLayers;
            int iymin = -numberLayers;
            int ixmin = -numberLayers;

            int izmax = numberLayers + 1;
            int iymax = numberLayers + 1;
            int ixmax = numberLayers + 1;



            for (int iz = izmin; iz < izmax; ++iz)
              {
                if (periodicZ == 0)
                  iz = izmax;
                for (int iy = iymin; iy < iymax; ++iy)
                  {
                    if (periodicY == 0)
                      iy = iymax;
                    for (int ix = ixmin; ix < ixmax; ++ix)
                      {
                        if (periodicX == 0)
                          ix = ixmax;

                        if ((periodicX * ix) != 0 || (periodicY * iy) != 0 ||
                            (periodicZ * iz) != 0)
                          {
                            const double newFracZ = periodicZ * iz + fracZ;
                            const double newFracY = periodicY * iy + fracY;
                            const double newFracX = periodicX * ix + fracX;

                            std::vector<double> newFrac(3);
                            newFrac[0] = newFracX;
                            newFrac[1] = newFracY;
                            newFrac[2] = newFracZ;

                            bool outsideCell  = true;
                            bool withinCutoff = false;


                            if (outsideCell)
                              {
                                const double distanceFromCell =
                                  internaldft::getMinDistanceFromImageToCell(
                                    latticeVectors, newFrac);

                                if (distanceFromCell < pspCutOff)
                                  withinCutoff = true;
                              }

                            std::vector<double> currentImageChargePosition(3,
                                                                           0.0);

                            if (outsideCell && withinCutoff)
                              {
                                imageIdsKptPool.push_back(iCharge);

                                for (int ii = 0; ii < 3; ++ii)
                                  for (int jj = 0; jj < 3; ++jj)
                                    currentImageChargePosition[ii] +=
                                      d_domainBoundingVectors[jj][ii] *
                                      newFrac[jj];

                                for (int ii = 0; ii < 3; ++ii)
                                  currentImageChargePosition[ii] -= shift[ii];

                                // imagePositions.push_back(currentImageChargePosition);

                                imagePositionsFlattenedKptPool.insert(
                                  imagePositionsFlattenedKptPool.end(),
                                  currentImageChargePosition.begin(),
                                  currentImageChargePosition.end());
                                /*if((newFracX >= -tol && newFracX <= 1+tol) &&
                                  (newFracY >= -tol && newFracY <= 1+tol) &&
                                  (newFracZ >= -tol && newFracZ <= 1+tol))
                                  outsideCell = false;*/
                              }
                          }
                      }
                  }
              }
          }
      }


    std::vector<int> recvCounts(numberKptGroups, 0);
    int              sendCount = imageIdsKptPool.size();
    if (sendCount == 0)
      {
        sendCount == 1;
        imageIdsKptPool.resize(1, -1);
        imagePositionsFlattenedKptPool.resize(3, 0.0);
      }
    int ierr = MPI_Allgather(
      &sendCount, 1, MPI_INT, &recvCounts[0], 1, MPI_INT, interpoolcomm);

    if (ierr)
      AssertThrow(false,
                  dealii::ExcMessage(
                    "DFT-FE Error: MPI Error in generate image charges"));


    const int numImageCharges =
      std::accumulate(recvCounts.begin(), recvCounts.end(), 0);

    if (numImageCharges > 0)
      {
        // std::cout<<"num image charges: "<<numImageCharges<<std::endl;

        imageIdsGathered.resize(numImageCharges, 0);
        imagePositionsFlattened.resize(numImageCharges * 3, 0);

        std::vector<int> displacementsImageIds(numberKptGroups, 0);
        std::vector<int> displacementsImagePos(numberKptGroups, 0);
        std::vector<int> recvCountsPos(numberKptGroups, 0);
        int              disp = 0;
        for (int i = 0; i < numberKptGroups; ++i)
          {
            displacementsImageIds[i] = disp;
            displacementsImagePos[i] = disp * 3;
            disp += recvCounts[i];
            recvCountsPos[i] = recvCounts[i] * 3;
          }


        ierr = MPI_Allgatherv(&imageIdsKptPool[0],
                              sendCount,
                              MPI_INT,
                              &imageIdsGathered[0],
                              &recvCounts[0],
                              &displacementsImageIds[0],
                              MPI_INT,
                              interpoolcomm);

        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage(
                        "DFT-FE Error: MPI Error in generate image charges"));
        const int sendCountPos = sendCount * 3;
        ierr = MPI_Allgatherv(&imagePositionsFlattenedKptPool[0],
                              sendCountPos,
                              MPI_DOUBLE,
                              &imagePositionsFlattened[0],
                              &recvCountsPos[0],
                              &displacementsImagePos[0],
                              MPI_DOUBLE,
                              interpoolcomm);

        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage(
                        "DFT-FE Error: MPI Error in generate image charges"));

        int              numNonTrivialImageCharges = 0;
        std::vector<int> nonTrivialToFullIndexMap;
        for (int i = 0; i < numImageCharges; ++i)
          {
            if (imageIdsGathered[i] != -1)
              {
                imageIds.push_back(imageIdsGathered[i]);
                nonTrivialToFullIndexMap.push_back(i);
                numNonTrivialImageCharges++;
              }
          }

        imageCharges.resize(numNonTrivialImageCharges);
        imagePositions.resize(numNonTrivialImageCharges,
                              std::vector<double>(3, 0.0));
        for (int i = 0; i < numNonTrivialImageCharges; ++i)
          {
            double atomCharge;
            if (d_dftParamsPtr->isPseudopotential)
              atomCharge = atomLocations[imageIds[i]][1];
            else
              atomCharge = atomLocations[imageIds[i]][0];

            imageCharges[i] = atomCharge;
            imagePositions[i][0] =
              imagePositionsFlattened[3 * nonTrivialToFullIndexMap[i] + 0];
            imagePositions[i][1] =
              imagePositionsFlattened[3 * nonTrivialToFullIndexMap[i] + 1];
            imagePositions[i][2] =
              imagePositionsFlattened[3 * nonTrivialToFullIndexMap[i] + 2];
          }
      }
    MPI_Barrier(interpoolcomm);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    createMasterChargeIdToImageIdMaps(
      const double                            pspCutOff,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      std::vector<std::vector<int>> &         globalChargeIdToImageIdMap)
  {
    const unsigned int numImageCharges     = imageIds.size();
    const unsigned int numberGlobalCharges = atomLocations.size();
    globalChargeIdToImageIdMap.clear();
    globalChargeIdToImageIdMap.resize(numberGlobalCharges);

    dealii::BoundingBox<3> boundingBoxTria(
      vectorTools::createBoundingBoxTriaLocallyOwned(dofHandler));
    dealii::Tensor<1, 3, double> tempDisp;
    tempDisp[0] = pspCutOff;
    tempDisp[1] = pspCutOff;
    tempDisp[2] = pspCutOff;
    for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
      {
        dealii::Point<3> atomCoord;
        atomCoord[0] = atomLocations[iCharge][2];
        atomCoord[1] = atomLocations[iCharge][3];
        atomCoord[2] = atomLocations[iCharge][4];


        std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
          boundaryPoints;
        boundaryPoints.first  = atomCoord - tempDisp;
        boundaryPoints.second = atomCoord + tempDisp;
        dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

        if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
            dealii::NeighborType::not_neighbors)
          globalChargeIdToImageIdMap[iCharge].push_back(iCharge);
      }

    for (int iImage = 0; iImage < numImageCharges; ++iImage)
      {
        dealii::Point<3> atomCoord;
        atomCoord[0] = imagePositions[iImage][0];
        atomCoord[1] = imagePositions[iImage][1];
        atomCoord[2] = imagePositions[iImage][2];


        std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
          boundaryPoints;
        boundaryPoints.first  = atomCoord - tempDisp;
        boundaryPoints.second = atomCoord + tempDisp;
        dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

        if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
            dealii::NeighborType::not_neighbors)
          ;
        {
          //
          // Get the masterChargeId corresponding to the current image atom
          //
          const int masterChargeId = imageIds[iImage];

          //
          // insert into the map
          //
          globalChargeIdToImageIdMap[masterChargeId].push_back(
            iImage + numberGlobalCharges);
        }
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
