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


#ifndef meshMovement_H_
#define meshMovement_H_
#include "constants.h"
#include "headers.h"
#include "dftParameters.h"

namespace dftfe
{
  /**
   * @brief Base class to move triangulation vertices
   *
   * @author Sambit Das
   */
  class meshMovementClass
  {
  public:
    /** @brief Constructor
     *
     * @param[in] mpi_comm_parent parent mpi communicator
     *  @param[in] mpi_comm_domain mpi communicator for domain decomposition
     */
    meshMovementClass(const MPI_Comm &     mpi_comm_parent,
                      const MPI_Comm &     mpi_comm_domain,
                      const dftParameters &dftParams);

    virtual ~meshMovementClass()
    {}

    /** @brief Initializes the required data-structures for a given triangulation
     *
     *  @param[in] triangulation triangulation object whose nodes are to be
     * moved
     *  @param[in] serial triangulation to create constraints from serial
     * dofHandler (temporary fix)
     *  @param[in] domainBoundingVectors domain vectors of the domain
     * corresponding to the triangulation object.
     */
    void init(dealii::Triangulation<3, 3> &           triangulation,
              dealii::Triangulation<3, 3> &           serialTriangulation,
              const std::vector<std::vector<double>> &domainBoundingVectors);

    /** @brief Re-initializes the required data-structures for a given triangulation
     *
     *  @param[in] domainBoundingVectors current domain vectors of the domain
     * corresponding to the triangulation object.
     */
    void
    initMoved(const std::vector<std::vector<double>> &domainBoundingVectors);

    /** @brief Finds the closest triangulation vertices to a given vector of position coordinates
     *
     *  @param[in] destinationPoints vector of points in cartesian coordinates
     * (origin at center of the domain) to which closest triangulation vertices
     * are desired.
     *  @param[out] closestTriaVertexToDestPointsLocation vector of positions of
     * the closest triangulation v vertices.
     *  @param[out] dispClosestTriaVerticesToDestPoints vector of displacements
     * of the destinationPoints from the closest triangulation vertices.
     */
    void
    findClosestVerticesToDestinationPoints(
      const std::vector<dealii::Point<3>> &destinationPoints,
      std::vector<dealii::Point<3>> &closestTriaVertexToDestPointsLocation,
      std::vector<dealii::Tensor<1, 3, double>>
        &dispClosestTriaVerticesToDestPoints);

  protected:
    /// Initializes the parallel layout of d_incrementalDisplacementParallel
    void
    initIncrementField();

    /// Takes care of communicating the movement of triangulation vertices on
    /// processor boundaries, and also takes care of hanging nodes and periodic
    /// constraints
    void
    finalizeIncrementField();

    /// Function which updates the locally relevant triangulation vertices
    void
    updateTriangulationVertices();

    /// Function which moves subdivided mesh
    void
    moveSubdividedMesh();

    /// Performs periodic matching sanity check and returns the pair<if negative
    /// jacobian, maximum inverse jacobian magnitude>
    std::pair<bool, double>
    movedMeshCheck();

    // virtual void computeIncrement()=0;

    /// vector of displacements of the triangulation vertices
    // dealii::Vector<double> d_incrementalDisplacement;
    distributedCPUVec<double> d_incrementalDisplacement;

    bool d_isParallelMesh;

    // dealii based FE data structres
    dealii::FESystem<3>                              FEMoveMesh;
    dealii::DoFHandler<3>                            d_dofHandlerMoveMesh;
    dealii::parallel::distributed::Triangulation<3> *d_triaPtr;
    dealii::Triangulation<3, 3> *                    d_triaPtrSerial;
    dealii::IndexSet                                 d_locally_owned_dofs;
    dealii::IndexSet                                 d_locally_relevant_dofs;
    dealii::AffineConstraints<double>                d_constraintsMoveMesh;
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::DoFHandler<3>::cell_iterator>>
                                     d_periodicity_vector;
    std::vector<std::vector<double>> d_domainBoundingVectors;

    const dftParameters &d_dftParams;

    // parallel objects
    MPI_Comm                   d_mpiCommParent;
    MPI_Comm                   mpi_communicator;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };
} // namespace dftfe
#endif
