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

#ifndef dft_H_
#define dft_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <elpaScalaManager.h>
#include <headers.h>
#include <MemorySpaceType.h>
#include <MemoryStorage.h>
#include <FEBasisOperations.h>
#include <BLASWrapper.h>

#include <complex>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifdef DFTFE_WITH_DEVICE
#  include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#  include "deviceKernelsGeneric.h"
#  include <poissonSolverProblemDevice.h>
#  include <kerkerSolverProblemDevice.h>
#  include <linearSolverCGDevice.h>
#  include <deviceDirectCCLWrapper.h>
#endif

#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dealiiLinearSolver.h>
#include <dftParameters.h>
#include <eigenSolver.h>
#include <interpolation.h>
#include <kerkerSolverProblem.h>
#include <KohnShamHamiltonianOperator.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include <poissonSolverProblem.h>
#include <triangulationManager.h>
#include <vselfBinsManager.h>
#include <excManager.h>
#include <dftd.h>
#include <force.h>
#include "dftBase.h"
#ifdef USE_PETSC
#  include <petsc.h>

#  include <slepceps.h>
#endif

#include <mixingClass.h>
#include <oncvClass.h>
#include <pawClass.h>

namespace dftfe
{
  //
  // Initialize Namespace
  //



#ifndef DOXYGEN_SHOULD_SKIP_THIS

  struct orbital
  {
    unsigned int                atomID;
    unsigned int                waveID;
    unsigned int                Z, n, l;
    int                         m;
    alglib::spline1dinterpolant psi;
  };

  /* code that must be skipped by Doxygen */
  // forward declarations
  template <unsigned int T1, unsigned int T2, dftfe::utils::MemorySpace memory>
  class symmetryClass;
  template <unsigned int T1, unsigned int T2, dftfe::utils::MemorySpace memory>
  class forceClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  /**
   * @brief This class is the primary interface location of all other parts of the DFT-FE code
   * for all steps involved in obtaining the Kohn-Sham DFT ground-state
   * solution.
   *
   * @author Shiva Rudraraju, Phani Motamarri, Sambit Das
   */
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  class dftClass : public dftBase
  {
    friend class forceClass<FEOrder, FEOrderElectro, memorySpace>;

    friend class symmetryClass<FEOrder, FEOrderElectro, memorySpace>;

  public:
    /**
     * @brief dftClass constructor
     *
     *  @param[in] mpi_comm_parent parent communicator
     *  @param[in] mpi_comm_domain  mpi_communicator for domain decomposition
     * parallelization
     *  @param[in] interpoolcomm  mpi_communicator for parallelization over k
     * points
     *  @param[in] interBandGroupComm  mpi_communicator for parallelization over
     * bands
     *  @param[in] scratchFolderName  scratch folder name
     *  @param[in] dftParams  dftParameters object containg parameter values
     * parsed from an input parameter file in dftfeWrapper class
     */
    dftClass(const MPI_Comm &   mpiCommParent,
             const MPI_Comm &   mpi_comm_domain,
             const MPI_Comm &   interpoolcomm,
             const MPI_Comm &   interBandGroupComm,
             const std::string &scratchFolderName,
             dftParameters &    dftParams);

    /**
     * @brief dftClass destructor
     */
    ~dftClass();

    /**
     * @brief atomic system pre-processing steps.
     *
     * Reads the coordinates of the atoms.
     * If periodic calculation, reads fractional coordinates of atoms in the
     * unit-cell, lattice vectors, kPoint quadrature rules to be used and also
     * generates image atoms. Also determines orbital-ordering
     */
    void
    set();

    /**
     * @brief Does KSDFT problem pre-processing steps including mesh generation calls.
     */
    void
    init();

    /**
     * @brief Does KSDFT problem pre-processing steps but without remeshing.
     */
    void
    initNoRemesh(const bool updateImagesAndKPointsAndVselfBins = true,
                 const bool checkSmearedChargeWidthsForOverlap = true,
                 const bool useSingleAtomSolutionOverride      = false,
                 const bool isMeshDeformed                     = false);



    /**
     * @brief FIXME: legacy call, move to main.cc
     */
    void
    run();

    /**
     * @brief Writes inital density and mesh to file.
     */
    void
    writeMesh();

    /**
     * @brief compute approximation to ground-state without solving the SCF iteration
     */
    void
    solveNoSCF();
    /**
     * @brief Kohn-Sham ground-state solve using SCF iteration
     *
     * @return tuple of boolean flag on whether scf converged,
     *  and L2 norm of residual electron-density of the last SCF iteration step
     *
     */
    std::tuple<bool, double>
    solve(const bool computeForces                 = true,
          const bool computestress                 = true,
          const bool restartGroundStateCalcFromChk = false);

    void
    computeStress();

    void
    trivialSolveForStress();


    void
    computeOutputDensityDirectionalDerivative(
      const distributedCPUVec<double> &v,
      const distributedCPUVec<double> &vSpin0,
      const distributedCPUVec<double> &vSpin1,
      distributedCPUVec<double> &      fv,
      distributedCPUVec<double> &      fvSpin0,
      distributedCPUVec<double> &      fvSpin1);

    /**
     * @brief Copies the residual residualValues=outValues-inValues
     */
    double
    computeResidualQuadData(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &outValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &inValues,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &residualValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        JxW,
      const bool computeNorm);


    double
    computeResidualNodalData(const distributedCPUVec<double> &outValues,
                             const distributedCPUVec<double> &inValues,
                             distributedCPUVec<double> &      residualValues);


    /**
     * @brief Computes the diagonal mass matrix for rho nodal grid, used for nodal mixing
     */
    void
    computeRhoNodalMassVector(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &massVec);



    void
    computeRhoNodalInverseMassVector();

    void
    computeTotalDensityNodalVector(
      const std::map<dealii::CellId, std::vector<double>> &bQuadValues,
      const distributedCPUVec<double> &                    electronDensity,
      distributedCPUVec<double> &                          totalChargeDensity);

    void
    initializeKohnShamDFTOperator(const bool initializeCublas = true);


    void
    reInitializeKohnShamDFTOperator();


    void
    finalizeKohnShamDFTOperator();


    double
    getInternalEnergy() const;

    double
    getEntropicEnergy() const;

    double
    getFreeEnergy() const;

    const distributedCPUVec<double> &
    getRhoNodalOut() const;

    const distributedCPUVec<double> &
    getRhoNodalSplitOut() const;

    double
    getTotalChargeforRhoSplit();

    void
    resetRhoNodalIn(distributedCPUVec<double> &OutDensity);

    virtual void
    resetRhoNodalSplitIn(distributedCPUVec<double> &OutDensity);

    /**
     * @brief Number of Kohn-Sham eigen values to be computed
     */
    unsigned int d_numEigenValues;

    /**
     * @brief Number of Kohn-Sham eigen values to be computed in the Rayleigh-Ritz step
     * after spectrum splitting.
     */
    unsigned int d_numEigenValuesRR;

    /**
     * @brief Number of random wavefunctions
     */
    unsigned int d_nonAtomicWaveFunctions;

    void
    readkPointData();

    /**
     *@brief Get local dofs global indices real
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalDofIndicesReal() const;

    /**
     *@brief Get local dofs global indices imag
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalDofIndicesImag() const;

    /**
     *@brief Get local dofs local proc indices real
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalProcDofIndicesReal() const;

    /**
     *@brief Get local dofs local proc indices imag
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalProcDofIndicesImag() const;

    /**
     *@brief Get matrix free data object
     */
    const dealii::MatrixFree<3, double> &
    getMatrixFreeData() const;


    /** @brief Updates atom positions, remeshes/moves mesh and calls appropriate reinits.
     *
     *  Function to update the atom positions and mesh based on the provided
     * displacement input. Depending on the maximum displacement magnitude this
     * function decides wether to do auto remeshing or move mesh using Gaussian
     * functions. Additionaly this function also wraps the atom position across
     * the periodic boundary if the atom moves across it beyond a certain
     * magnitude. In case of floating atoms, only the atomic positions are
     * updated keeping the mesh fixed. This function also calls initNoRemesh to
     * reinitialize all the required FEM and KSDFT objects.
     *
     *  @param[in] globalAtomsDisplacements vector containing the displacements
     * (from current position) of all atoms (global).
     *  @return void.
     */
    void
    updateAtomPositionsAndMoveMesh(
      const std::vector<dealii::Tensor<1, 3, double>> &globalAtomsDisplacements,
      const double maxJacobianRatioFactor         = 1.25,
      const bool   useSingleAtomSolutionsOverride = false);


    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files, which are required for
     * geometry relaxation restart

     */
    void
    writeDomainAndAtomCoordinates();

    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files for
     * structural optimization and dynamics restarts. The coordinates are stored
     * as: 1. fractional for semi-periodic/periodic 2. Cartesian for
     * non-periodic.
     * @param[in] Path The folder path to store the atom coordinates required
     * during restart.
     */
    void
    writeDomainAndAtomCoordinates(const std::string Path) const;

    /**
     * @brief writes atomistics data for subsequent post-processing. Related to
     * WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS input parameter.
     * @param[in] Path The folder path to store the atomistics data.
     */
    void
    writeStructureEnergyForcesDataPostProcess(const std::string Path) const;

    /**
     * @brief writes quadrature grid information and associated spin-up
     * and spin-down electron-density for post-processing
     * @param[in] Path The folder path to store the atomistics data.
     */
    virtual void
    writeGSElectronDensity(const std::string Path) const;


    /**
     * @brief Gets the current atom Locations in cartesian form
     * (origin at center of domain) from dftClass
     */
    const std::vector<std::vector<double>> &
    getAtomLocationsCart() const;


    /**
     * @brief Gets the current image atom Locations in cartesian form
     * (origin at center of domain) from dftClass
     */
    const std::vector<std::vector<double>> &
    getImageAtomLocationsCart() const;

    /**
     * @brief Gets the current image atom ids from dftClass
     */
    const std::vector<int> &
    getImageAtomIDs() const;

    /**
     * @brief Gets the current atom Locations in fractional form
     * from dftClass (only applicable for periodic and semi-periodic BCs)
     */
    const std::vector<std::vector<double>> &
    getAtomLocationsFrac() const;



    /**
     * @brief Gets the current cell lattice vectors
     *
     *  @return std::vector<std::vector<double>> 3 \times 3 matrix,lattice[i][j]
     *  corresponds to jth component of ith lattice vector
     */
    const std::vector<std::vector<double>> &
    getCell() const;

    /**
     * @brief Gets the current cell volume
     *
     */
    double
    getCellVolume() const;

    /**
     * @brief Gets the current atom types from dftClass
     */
    const std::set<unsigned int> &
    getAtomTypes() const;

    /**
     * @brief Gets the current atomic forces from dftClass
     */
    const std::vector<double> &
    getForceonAtoms() const;

    /**
     * @brief Gets the current cell stress from dftClass
     */
    const dealii::Tensor<2, 3, double> &
    getCellStress() const;

    /**
     * @brief Get reference to dftParameters object
     */
    dftParameters &
    getParametersObject() const;

  private:
    /**
     * @brief generate image charges and update k point cartesian coordinates based
     * on current lattice vectors
     */
    void
    initImageChargesUpdateKPoints(bool flag = true);

    void
    determineAtomsOfInterstPseudopotential(
      const std::vector<std::vector<double>> &atomCoordinates);


    /**
     *@brief project ground state electron density from previous mesh into
     * the new mesh to be used as initial guess for the new ground state solve
     */
    void
    projectPreviousGroundStateRho();

    /**
     *@brief save triangulation information and rho quadrature data to checkpoint file for restarts
     */
    void
    saveTriaInfoAndRhoNodalData();

    /**
     *@brief load triangulation information rho quadrature data from checkpoint file for restarted run
     */
    void
    loadTriaInfoAndRhoNodalData();

    void
    generateMPGrid();
    void
    writeMesh(std::string meshFileName);

    /// creates datastructures related to periodic image charges
    void
    generateImageCharges(const double                      pspCutOff,
                         std::vector<int> &                imageIds,
                         std::vector<double> &             imageCharges,
                         std::vector<std::vector<double>> &imagePositions);

    void
    createMasterChargeIdToImageIdMaps(
      const double                            pspCutOff,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      std::vector<std::vector<int>> &         globalChargeIdToImageIdMap);

    void
    determineOrbitalFilling();

    //
    // generate mesh using a-posteriori error estimates
    //
    void
    aposterioriMeshGenerate();
    void
    computeTraceXtHX(unsigned int       numberWaveFunctionsEstimate,
                     const unsigned int kpoint = 0);
    double
    computeTraceXtKX(unsigned int numberWaveFunctionsEstimate);


    /**
     *@brief  moves the triangulation vertices using Gaussians such that the all atoms are on triangulation vertices
     */
    void moveMeshToAtoms(dealii::Triangulation<3, 3> &triangulationMove,
                         dealii::Triangulation<3, 3> &triangulationSerial,
                         bool                         reuseFlag      = false,
                         bool                         moveSubdivided = false);

    /**
     *@brief  a
     */
    void
    calculateSmearedChargeWidths();

    /**
     *@brief  a
     */
    void
    calculateNearestAtomDistances();

    /**
     * Initializes the guess of electron-density and single-atom wavefunctions
     * on the mesh, maps finite-element nodes to given atomic positions,
     * initializes pseudopotential files and exchange-correlation functionals to
     * be used based on user-choice. In periodic problems, periodic faces are
     * mapped here. Further finite-element nodes to be pinned for solving the
     * Poisson problem electro-static potential is set here
     */
    void initUnmovedTriangulation(
      dealii::parallel::distributed::Triangulation<3> &triangulation);
    void
    initBoundaryConditions(const bool recomputeBasisData               = true,
                           const bool meshOnlyDeformed                 = false,
                           const bool vselfPerturbationUpdateForStress = false);
    void
    initElectronicFields();
    void
    initPseudoPotentialAll(const bool updateNonlocalSparsity = true);

    /**
     * create a dofHandler containing finite-element interpolating polynomial
     * twice of the original polynomial required for Kerker mixing and
     * initialize various objects related to this refined dofHandler
     */
    void createpRefinedDofHandler(
      dealii::parallel::distributed::Triangulation<3> &triangulation);
    void
    initpRefinedObjects(const bool recomputeBasisData,
                        const bool meshOnlyDeformed,
                        const bool vselfPerturbationUpdateForStress = false);


    void
    updatePRefinedConstraints();

    /**
     *@brief Sets inhomegeneous dirichlet boundary conditions upto quadrupole for total potential constraints on
     * non-periodic boundary (boundary id==0).
     *
     * @param[in] dofHandler
     * @param[out] constraintMatrix dealii::AffineConstraints<double> object
     *with inhomogeneous Dirichlet boundary condition entries added
     */
    void
    applyMultipoleDirichletBC(
      const dealii::DoFHandler<3> &            _dofHandler,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
      dealii::AffineConstraints<double> &      constraintMatrix);


    void
    computeMultipoleMoments(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int densityQuadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  rhoQuadValues,
      const std::map<dealii::CellId, std::vector<double>> *bQuadValues);


    /**
     *@brief interpolate rho nodal data to quadrature data using FEEvaluation
     *
     *@param[in] basisOperationsPtr basisoperationsPtr object
     *@param[in] nodalField nodal data to be interpolated
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void
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
      const bool isEvaluateGradData    = false,
      const bool isEvaluateHessianData = false);

    /**
     *@brief interpolate nodal data to quadrature data using FEEvaluation
     *
     *@param[in] matrixFreeData matrix free data object
     *@param[in] nodalField nodal data to be interpolated
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void
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
      const bool isEvaluateGradData = false);


    /**
     *@brief interpolate rho nodal data to quadrature data using FEEvaluation
     *
     *@param[in] basisOperationsPtr basisoperationsPtr object
     *@param[in] nodalField nodal data to be interpolated
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void
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
      const bool isEvaluateGradData);


    /**
     *@brief add atomic densities at quadrature points
     *
     */
    void
    addAtomicRhoQuadValuesGradients(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureGradValueData,
      const bool isConsiderGradData = false);


    /**
     *@brief Finds the global dof ids of the nodes containing atoms.
     *
     * @param[in] dofHandler
     * @param[out] atomNodeIdToChargeValueMap local map of global dof id to atom
     *charge id
     */
    void
    locateAtomCoreNodes(const dealii::DoFHandler<3> &_dofHandler,
                        std::map<dealii::types::global_dof_index, double>
                          &atomNodeIdToChargeValueMap);

    /**
     *@brief Sets homogeneous dirichlet boundary conditions on a node farthest from
     * all atoms (pinned node). This is only done in case of periodic boundary
     *conditions to get an unique solution to the total electrostatic potential
     *problem.
     *
     * @param[in] dofHandler
     * @param[in] constraintMatrixBase base dealii::AffineConstraints<double>
     *object
     * @param[out] constraintMatrix dealii::AffineConstraints<double> object
     *with homogeneous Dirichlet boundary condition entries added
     */
    void
    locatePeriodicPinnedNodes(
      const dealii::DoFHandler<3> &            _dofHandler,
      const dealii::AffineConstraints<double> &constraintMatrixBase,
      dealii::AffineConstraints<double> &      constraintMatrix);

    void
    initAtomicRho();

    double d_atomicRhoScalingFac;

    void
    initRho();
    void
    initCoreRho();
    void
    initZeroPotential();
    void
    computeRhoInitialGuessFromPSI(
      std::vector<std::vector<distributedCPUVec<double>>> eigenVectors);
    void
    clearRhoData();


    /**
     *@brief computes density nodal data from wavefunctions
     */
    void
    computeRhoNodalFromPSI(bool isConsiderSpectrumSplitting);


    void
    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
      distributedCPUVec<double> &fv,
      distributedCPUVec<double> &fvSpin0,
      distributedCPUVec<double> &fvSpin1);

    void
    noRemeshRhoDataInit();
    void
    readPSI();
    void
    readPSIRadialValues();
    void
    loadPSIFiles(unsigned int  Z,
                 unsigned int  n,
                 unsigned int  l,
                 unsigned int &flag);
    void
    initLocalPseudoPotential(
      const dealii::DoFHandler<3> &            _dofHandler,
      const unsigned int                       lpspQuadratureId,
      const dealii::MatrixFree<3, double> &    _matrix_free_data,
      const unsigned int                       _phiExtDofHandlerIndex,
      const dealii::AffineConstraints<double> &phiExtConstraintMatrix,
      const std::map<dealii::types::global_dof_index, dealii::Point<3>>
        &                                              supportPoints,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinManager,
      distributedCPUVec<double> &                      phiExt,
      std::map<dealii::CellId, std::vector<double>> &  _pseudoValues,
      std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
        &_pseudoValuesAtoms);



    /**
     *@brief Sets homegeneous dirichlet boundary conditions for total potential constraints on
     * non-periodic boundary (boundary id==0).
     *
     * @param[in] dofHandler
     * @param[out] constraintMatrix dealii::AffineConstraints<double> object
     *with homogeneous Dirichlet boundary condition entries added
     */
    void
    applyHomogeneousDirichletBC(
      const dealii::DoFHandler<3> &            _dofHandler,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
      dealii::AffineConstraints<double> &      constraintMatrix);



    /**
     *@brief Computes total charge by integrating the electron-density
     */
    double
    totalCharge(const dealii::DoFHandler<3> &    dofHandlerOfField,
                const distributedCPUVec<double> &rhoNodalField);


    double
    totalCharge(
      const dealii::DoFHandler<3> &                        dofHandlerOfField,
      const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues);

    double
    totalCharge(
      const dealii::DoFHandler<3> &dofHandlerOfField,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoQuadValues);


    double
    totalCharge(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                const distributedCPUVec<double> &    rhoNodalField);



    double
    rhofieldl2Norm(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                   const distributedCPUVec<double> &    rhoNodalField,
                   const unsigned int                   dofHandlerId,
                   const unsigned int                   quadratureId);

    double
    rhofieldInnerProduct(
      const dealii::MatrixFree<3, double> &matrixFreeDataObject,
      const distributedCPUVec<double> &    rhoNodalField1,
      const distributedCPUVec<double> &    rhoNodalField2,
      const unsigned int                   dofHandlerId,
      const unsigned int                   quadratureId);


    double
    fieldGradl2Norm(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                    const distributedCPUVec<double> &    field);

    /**
     *@brief l2 projection
     */
    void
    l2ProjectionQuadToNodal(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                                      basisOperationsPtr,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const unsigned int                       dofHandlerId,
      const unsigned int                       quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                        quadratureValueData,
      distributedCPUVec<double> &nodalField);

    /**
     *@brief l2 projection
     */
    void
    l2ProjectionQuadDensityMinusAtomicDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                                      basisOperationsPtr,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const unsigned int                       dofHandlerId,
      const unsigned int                       quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                        quadratureValueData,
      distributedCPUVec<double> &nodalField);

    /**
     *@brief Computes net magnetization from the difference of local spin densities
     */
    double
    totalMagnetization(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &magQuadValues);

    /**
     *@brief normalize the input electron density
     */
    void
    normalizeRhoInQuadValues();
    /**
     *@brief Used in PAW to scale rhoIn
     */
    void
    scaleRhoInQuadValues(double scalingFactor);

    /**
     *@brief normalize the output electron density in each scf
     */
    void
    normalizeRhoOutQuadValues();

    /**
     *@brief normalize the electron density
     */
    void
    normalizeAtomicRhoQuadValues();

    /**
     *@brief Computes output electron-density from wavefunctions
     */
    void
    compute_rhoOut(const bool isConsiderSpectrumSplitting,
                   const bool isGroundState = false);

    /**
     *@brief Mixing schemes for mixing electron-density
     */

    void
    applyKerkerPreconditionerToTotalDensityResidual(
#ifdef DFTFE_WITH_DEVICE
      kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                   kerkerPreconditionedResidualSolverProblemDevice,
      linearSolverCGDevice &CGSolverDevice,
#endif
      kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                 kerkerPreconditionedResidualSolverProblem,
      dealiiLinearSolver &CGSolver,
      const distributedCPUVec<double> &residualRho,
      distributedCPUVec<double> &      preCondTotalDensityResidualVector);

    double
    lowrankApproxScfDielectricMatrixInv(const unsigned int scfIter);

    double
    lowrankApproxScfDielectricMatrixInvSpinPolarized(
      const unsigned int scfIter);

    /**
     *@brief Computes Fermi-energy obtained by imposing constraint on the number of electrons
     */
    void
    compute_fermienergy(
      const std::vector<std::vector<double>> &eigenValuesInput,
      const double                            numElectronsInput);
    /**
     *@brief Computes Fermi-energy obtained by imposing separate constraints on the number of spin-up and spin-down electrons
     */
    void
    compute_fermienergy_constraintMagnetization(
      const std::vector<std::vector<double>> &eigenValuesInput);

    /**
     *@brief compute density of states and local density of states
     */
    void
    compute_tdos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const unsigned int                      highestStateOfInterest,
                 const std::string &                     fileName);

    void
    compute_ldos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const std::string &                     fileName);

    void
    compute_pdos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const std::string &                     fileName);


    /**
     *@brief compute localization length
     */
    void
    compute_localizationLength(const std::string &locLengthFileName);

    /**
     *@brief write wavefunction solution fields
     */
    void
    outputWfc();

    /**
     *@brief write electron density solution fields
     */
    void
    outputDensity();

    /**
     *@brief write the KS eigen values for given BZ sampling/path
     */
    void
    writeBands();

    /**
     *@brief Computes the volume of the domain
     */
    double
    computeVolume(const dealii::DoFHandler<3> &_dofHandler);

    /**
     *@brief Deforms the domain by the given deformation gradient and reinitializes the
     * dftClass datastructures.
     */
    void
    deformDomain(const dealii::Tensor<2, 3, double> &deformationGradient,
                 const bool vselfPerturbationUpdateForStress = false,
                 const bool useSingleAtomSolutionsOverride   = false,
                 const bool print                            = true);

    /**
     *@brief Computes inner Product and Y = alpha*X + Y for complex vectors used during
     * periodic boundary conditions
     */

#ifdef USE_COMPLEX
    std::complex<double>
    innerProduct(distributedCPUVec<double> &a, distributedCPUVec<double> &b);

    void
    alphaTimesXPlusY(std::complex<double>       alpha,
                     distributedCPUVec<double> &x,
                     distributedCPUVec<double> &y);

#endif
    /**
     *@brief Sets dirichlet boundary conditions for total potential constraints on
     * non-periodic boundary (boundary id==0). Currently setting homogeneous bc
     *
     */
    void
    applyPeriodicBCHigherOrderNodes();



    std::shared_ptr<excManager> d_excManagerPtr;
    dispersionCorrection        d_dispersionCorr;

    /**
     * stores required data for Kohn-Sham problem
     */
    unsigned int numElectrons, numElectronsUp, numElectronsDown, numLevels;
    std::set<unsigned int> atomTypes;

    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;

    /// FIXME: remove atom type atributes from atomLocations
    std::vector<std::vector<double>> atomLocations, atomLocationsFractional,
      d_reciprocalLatticeVectors, d_domainBoundingVectors;
    std::vector<std::vector<double>> d_atomLocationsInterestPseudopotential;
    std::map<unsigned int, unsigned int>
                                     d_atomIdPseudopotentialInterestToGlobalId;
    std::vector<std::vector<double>> d_atomLocationsAutoMesh;
    std::vector<std::vector<double>> d_imagePositionsAutoMesh;

    /// Gaussian displacements of atoms read from file
    std::vector<dealii::Tensor<1, 3, double>> d_atomsDisplacementsGaussianRead;

    ///
    std::vector<double> d_netFloatingDispSinceLastBinsUpdate;

    ///
    std::vector<double> d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps;

    bool d_isAtomsGaussianDisplacementsReadFromFile = false;

    /// Gaussian generator parameter for force computation and Gaussian
    /// deformation of atoms and FEM mesh Gaussian generator: Gamma(r)=
    /// exp(-(r/d_gaussianConstant)^2) Stored for all domain atoms
    std::vector<double> d_gaussianConstantsForce;

    /// Gaussian constants for automesh mesh movement stored for all domain
    /// atoms
    std::vector<double> d_gaussianConstantsAutoMesh;

    /// composite generator flat top widths for all domain atoms
    std::vector<double> d_generatorFlatTopWidths;

    /// flat top widths for all domain atoms in case of automesh mesh movement
    /// composite gaussian
    std::vector<double> d_flatTopWidthsAutoMeshMove;

    /// smeared charge widths for all domain atoms
    std::vector<double> d_smearedChargeWidths;

    /// smeared charge normalization scaling for all domain atoms
    std::vector<double> d_smearedChargeScaling;

    /// nearest atom ids for all domain atoms
    std::vector<unsigned int> d_nearestAtomIds;

    /// nearest atom distances for all domain atoms
    std::vector<double> d_nearestAtomDistances;

    ///
    double d_minDist;

    /// vector of lendth number of periodic image charges with corresponding
    /// master chargeIds
    std::vector<int> d_imageIds;
    // std::vector<int> d_imageIdsAutoMesh;


    /// vector of length number of periodic image charges with corresponding
    /// charge values
    std::vector<double> d_imageCharges;

    /// vector of length number of periodic image charges with corresponding
    /// positions in cartesian coordinates
    std::vector<std::vector<double>> d_imagePositions;

    /// globalChargeId to ImageChargeId Map
    std::vector<std::vector<int>> d_globalChargeIdToImageIdMap;

    /// vector of lendth number of periodic image charges with corresponding
    /// master chargeIds , generated with a truncated pspCutoff
    std::vector<int> d_imageIdsTrunc;

    /// vector of length number of periodic image charges with corresponding
    /// charge values , generated with a truncated pspCutoff
    std::vector<double> d_imageChargesTrunc;

    /// vector of length number of periodic image charges with corresponding
    /// positions in cartesian coordinates, generated with a truncated pspCutOff
    std::vector<std::vector<double>> d_imagePositionsTrunc;

    /// globalChargeId to ImageChargeId Map generated with a truncated pspCutOff
    std::vector<std::vector<int>> d_globalChargeIdToImageIdMapTrunc;

    /// distance from the domain till which periodic images will be considered
    double d_pspCutOff = 15.0;

    /// distance from the domain till which periodic images will be considered
    const double d_pspCutOffTrunc = 15.0;

    /// cut-off distance from atom till which non-local projectors are
    /// non-trivial
    double d_nlPSPCutOff = 8.0;

    /// non-intersecting smeared charges of all atoms at quad points
    std::map<dealii::CellId, std::vector<double>> d_bQuadValuesAllAtoms;

    /// non-intersecting smeared charge gradients of all atoms at quad points
    std::map<dealii::CellId, std::vector<double>> d_gradbQuadValuesAllAtoms;

    /// non-intersecting smeared charges atom ids of all atoms at quad points
    std::map<dealii::CellId, std::vector<int>> d_bQuadAtomIdsAllAtoms;

    /// non-intersecting smeared charges atom ids of all atoms (with image atom
    /// ids separately accounted) at quad points
    std::map<dealii::CellId, std::vector<int>> d_bQuadAtomIdsAllAtomsImages;

    /// map of cell and non-trivial global atom ids (no images) for smeared
    /// charges for each bin
    std::map<dealii::CellId, std::vector<unsigned int>>
      d_bCellNonTrivialAtomIds;

    /// map of cell and non-trivial global atom ids (no images) for smeared
    /// charge for each bin
    std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
      d_bCellNonTrivialAtomIdsBins;

    /// map of cell and non-trivial global atom and image ids for smeared
    /// charges for each bin
    std::map<dealii::CellId, std::vector<unsigned int>>
      d_bCellNonTrivialAtomImageIds;

    /// map of cell and non-trivial global atom and image ids for smeared charge
    /// for each bin
    std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
      d_bCellNonTrivialAtomImageIdsBins;

    /// minimum smeared charge width
    const double d_smearedChargeWidthMin = 0.4;

    std::vector<orbital> waveFunctionsVector;
    std::map<unsigned int,
             std::map<unsigned int,
                      std::map<unsigned int, alglib::spline1dinterpolant>>>
      radValues;
    std::map<unsigned int,
             std::map<unsigned int, std::map<unsigned int, double>>>
      outerValues;

    /**
     * meshGenerator based object
     */
    triangulationManager d_mesh;

    double       d_autoMeshMaxJacobianRatio;
    unsigned int d_autoMesh;


    /// affine transformation object
    meshMovementAffineTransform d_affineTransformMesh;

    /// meshMovementGaussianClass object
    meshMovementGaussianClass d_gaussianMovePar;

    std::vector<dealii::Tensor<1, 3, double>>
                                  d_gaussianMovementAtomsNetDisplacements;
    std::vector<dealii::Point<3>> d_controlPointLocationsCurrentMove;

    /// volume of the domain
    double d_domainVolume;

    /// init wfc trunctation radius
    double d_wfcInitTruncation = 5.0;

    /**
     * dealii based FE data structres
     */
    dealii::FESystem<3>   FE, FEEigen;
    dealii::DoFHandler<3> dofHandler, dofHandlerEigen, d_dofHandlerPRefined,
      d_dofHandlerRhoNodal;
    unsigned int d_eigenDofHandlerIndex, d_phiExtDofHandlerIndexElectro,
      d_forceDofHandlerIndex;
    unsigned int                  d_densityDofHandlerIndex;
    unsigned int                  d_densityDofHandlerIndexElectro;
    unsigned int                  d_nonPeriodicDensityDofHandlerIndexElectro;
    unsigned int                  d_baseDofHandlerIndexElectro;
    unsigned int                  d_forceDofHandlerIndexElectro;
    unsigned int                  d_smearedChargeQuadratureIdElectro;
    unsigned int                  d_nlpspQuadratureId;
    unsigned int                  d_lpspQuadratureId;
    unsigned int                  d_feOrderPlusOneQuadratureId;
    unsigned int                  d_lpspQuadratureIdElectro;
    unsigned int                  d_gllQuadratureId;
    unsigned int                  d_phiTotDofHandlerIndexElectro;
    unsigned int                  d_phiPrimeDofHandlerIndexElectro;
    unsigned int                  d_phiTotAXQuadratureIdElectro;
    unsigned int                  d_helmholtzDofHandlerIndexElectro;
    unsigned int                  d_binsStartDofHandlerIndexElectro;
    unsigned int                  d_densityQuadratureId;
    unsigned int                  d_densityQuadratureIdElectro;
    unsigned int                  d_sparsityPatternQuadratureId;
    unsigned int                  d_nOMPThreads;
    dealii::MatrixFree<3, double> matrix_free_data, d_matrixFreeDataPRefined;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrElectroHost;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      d_basisOperationsPtrDevice;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
      d_basisOperationsPtrElectroDevice;
#endif

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperPtrHost;

    std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
      d_oncvClassPtr;
    std::shared_ptr<dftfe::pawClass<dataTypes::number, memorySpace>>
      d_pawClassPtr;

    std::shared_ptr<
#if defined(DFTFE_WITH_DEVICE)
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
#else
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
#endif
      d_BLASWrapperPtr;

    std::map<dealii::types::global_dof_index, dealii::Point<3>> d_supportPoints,
      d_supportPointsPRefined, d_supportPointsEigen;
    std::vector<const dealii::AffineConstraints<double> *> d_constraintsVector;
    std::vector<const dealii::AffineConstraints<double> *>
      d_constraintsVectorElectro;

    /**
     * parallel objects
     */
    const MPI_Comm mpi_communicator;
#if defined(DFTFE_WITH_DEVICE)
    utils::DeviceCCLWrapper *d_devicecclMpiCommDomainPtr;
#endif
    const MPI_Comm     d_mpiCommParent;
    const MPI_Comm     interpoolcomm;
    const MPI_Comm     interBandGroupComm;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    dealii::IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
    dealii::IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen,
      d_locallyRelevantDofsPRefined, d_locallyRelevantDofsRhoNodal;
    std::vector<dealii::types::global_dof_index> local_dof_indicesReal,
      local_dof_indicesImag;
    std::vector<dealii::types::global_dof_index> localProc_dof_indicesReal,
      localProc_dof_indicesImag;
    std::vector<bool> selectedDofsHanging;

    forceClass<FEOrder, FEOrderElectro, memorySpace> *   forcePtr;
    symmetryClass<FEOrder, FEOrderElectro, memorySpace> *symmetryPtr;

    elpaScalaManager *d_elpaScala;

    poissonSolverProblem<FEOrder, FEOrderElectro> d_phiTotalSolverProblem;

    poissonSolverProblem<FEOrder, FEOrderElectro> d_phiPrimeSolverProblem;
#ifdef DFTFE_WITH_DEVICE
    poissonSolverProblemDevice<FEOrder, FEOrderElectro>
      d_phiTotalSolverProblemDevice;

    poissonSolverProblemDevice<FEOrder, FEOrderElectro>
      d_phiPrimeSolverProblemDevice;
#endif

    bool d_kohnShamDFTOperatorsInitialized;

    KohnShamHamiltonianOperator<memorySpace> *d_kohnShamDFTOperatorPtr;

    const std::string d_dftfeScratchFolderName;

    /**
     * chebyshev subspace iteration solver objects
     *
     */
    chebyshevOrthogonalizedSubspaceIterationSolver d_subspaceIterationSolver;
#ifdef DFTFE_WITH_DEVICE
    chebyshevOrthogonalizedSubspaceIterationSolverDevice
      d_subspaceIterationSolverDevice;
#endif

    /**
     * constraint Matrices
     */

    /**
     *object which is used to store dealii constraint matrix information
     *using STL vectors. The relevant dealii constraint matrix
     *has hanging node constraints and periodic constraints(for periodic
     *problems) used in eigen solve
     */
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      constraintsNoneEigenDataInfo;


    /**
     *object which is used to store dealii constraint matrix information
     *using STL vectors. The relevant dealii constraint matrix
     *has hanging node constraints used in Poisson problem solution
     *
     */
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      constraintsNoneDataInfo;


#ifdef DFTFE_WITH_DEVICE
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
      d_constraintsNoneDataInfoDevice;
#endif


    dealii::AffineConstraints<double> constraintsNone, constraintsNoneEigen,
      d_noConstraints;

    dealii::AffineConstraints<double> d_constraintsForTotalPotentialElectro;

    dealii::AffineConstraints<double> d_constraintsForPhiPrimeElectro;

    dealii::AffineConstraints<double> d_constraintsForHelmholtzRhoNodal;

    dealii::AffineConstraints<double> d_constraintsPRefined;

    dealii::AffineConstraints<double> d_constraintsPRefinedOnlyHanging;

    dealii::AffineConstraints<double> d_constraintsRhoNodal;

    dealii::AffineConstraints<double> d_constraintsRhoNodalOnlyHanging;

    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      d_constraintsRhoNodalInfo;

    /**
     * data storage for Kohn-Sham wavefunctions
     */
    std::vector<std::vector<double>> eigenValues;

    std::vector<std::vector<double>> d_densityMatDerFermiEnergy;

    /// Spectrum split higher eigenvalues computed in Rayleigh-Ritz step
    std::vector<std::vector<double>> eigenValuesRRSplit;

    /**
     * The indexing of d_eigenVectorsFlattenedHost and
     * d_eigenVectorsFlattenedDevice [kPoint * numSpinComponents *
     * numLocallyOwnedNodes * numWaveFunctions + iSpin * numLocallyOwnedNodes *
     * numWaveFunctions + iNode * numWaveFunctions + iWaveFunction]
     */
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsFlattenedHost;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsRotFracDensityFlattenedHost;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsDensityMatrixPrimeHost;

    /// device eigenvectors
#ifdef DFTFE_WITH_DEVICE
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsRotFracFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsDensityMatrixPrimeFlattenedDevice;
#endif

    /// parallel message stream
    dealii::ConditionalOStream pcout;

    /// compute-time logger
    dealii::TimerOutput computing_timer;
    dealii::TimerOutput computingTimerStandard;

    /// A plain global timer to track only the total elapsed time after every
    /// ground-state solve
    dealii::Timer d_globalTimer;

    // dft related objects
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      d_densityInQuadValues, d_densityOutQuadValues,
      d_densityResidualQuadValues;
    std::vector<distributedCPUVec<double>> d_densityInNodalValues,
      d_densityOutNodalValues, d_densityResidualNodalValues;

    std::vector<distributedCPUVec<double>> d_totalChargeDensityInNodalValues,
      d_totalChargeDensityOutNodalValues,
      d_totalChargeDensityResidualNodalValues;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_inverseRhoNodalMassVector;

    // std::map<dealii::CellId, std::vector<double>> d_phiInValues,
    // d_phiOutValues;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_phiInQuadValues, d_phiOutQuadValues, d_phitTotQuadPointsCompensation;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                 d_gradPhiInQuadValues, d_gradPhiOutQuadValues, d_gradPhiResQuadValues;
    MixingScheme d_mixingScheme;

    distributedCPUVec<double> d_rhoInNodalValuesRead, d_rhoOutNodalValuesSplit,
      d_preCondTotalDensityResidualVector, d_rhoNodalFieldRefined,
      d_rhoOutNodalValuesDistributed;


    distributedCPUVec<double> d_magInNodalValuesRead;


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_densityTotalOutValuesLpspQuad, d_densityTotalInValuesLpspQuad,
      d_gradDensityTotalOutValuesLpspQuad, d_gradDensityTotalInValuesLpspQuad;

    // For multipole boundary conditions
    double              d_monopole;
    std::vector<double> d_dipole;
    std::vector<double> d_quadrupole;
    std::vector<double> d_smearedChargeMoments;
    bool                d_smearedChargeMomentsComputed;


    /// for low rank jacobian inverse approximation
    std::deque<distributedCPUVec<double>> d_vcontainerVals;
    std::deque<distributedCPUVec<double>> d_fvcontainerVals;
    std::deque<distributedCPUVec<double>> d_vSpin0containerVals;
    std::deque<distributedCPUVec<double>> d_fvSpin0containerVals;
    std::deque<distributedCPUVec<double>> d_vSpin1containerVals;
    std::deque<distributedCPUVec<double>> d_fvSpin1containerVals;
    distributedCPUVec<double>             d_residualPredicted;
    unsigned int                          d_rankCurrentLRD;
    double                                d_relativeErrorJacInvApproxPrevScfLRD;
    double                                d_residualNormPredicted;
    bool                                  d_tolReached;

    /// for xl-bomd
    std::map<dealii::CellId, std::vector<double>> d_rhoAtomsValues,
      d_gradRhoAtomsValues, d_hessianRhoAtomsValues;
    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_rhoAtomsValuesSeparate, d_gradRhoAtomsValuesSeparate,
      d_hessianRhoAtomsValuesSeparate;

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      d_gradDensityInQuadValues, d_gradDensityOutQuadValues,
      d_gradDensityResidualQuadValues;

    // storage for total electrostatic potential solution vector corresponding
    // to input scf electron density
    distributedCPUVec<double> d_phiTotRhoIn;

    // storage for total electrostatic potential solution vector corresponding
    // to output scf electron density
    distributedCPUVec<double> d_phiTotRhoOut;

    // storage for electrostatic potential Gateaux derivate corresponding
    // to electron number preserving electron-density peturbation (required for
    // LRDM)
    distributedCPUVec<double> d_phiPrime;

    // storage for sum of nuclear electrostatic potential
    distributedCPUVec<double> d_phiExt;

    // storage of densities for xl-bomd
    std::deque<distributedCPUVec<double>> d_groundStateDensityHistory;

    std::map<dealii::CellId, std::vector<double>> d_pseudoVLoc;
    std::map<dealii::CellId, std::vector<double>> d_zeroPotential;

    /// Internal data:: map for cell id to Vpseudo local of individual atoms.
    /// Only for atoms whose psp tail intersects the local domain.
    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_pseudoVLocAtoms;
    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_zeroPotentialAtoms;


    std::vector<std::vector<double>> d_localVselfs;

    // nonlocal pseudopotential related objects used only for pseudopotential
    // calculation
    std::map<dealii::CellId, std::vector<double>> d_rhoCore;

    std::map<dealii::CellId, std::vector<double>> d_gradRhoCore;

    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_gradRhoCoreAtoms;

    std::map<dealii::CellId, std::vector<double>> d_hessianRhoCore;

    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_hessianRhoCoreAtoms;



    /// map of atom node number and atomic weight
    std::map<dealii::types::global_dof_index, double> d_atomNodeIdToChargeMap;

    /// vselfBinsManager object
    vselfBinsManager<FEOrder, FEOrderElectro> d_vselfBinsManager;

    /// Gateaux derivative of vself field with respect to affine strain tensor
    /// components using central finite difference. This is used for cell stress
    /// computation
    std::vector<distributedCPUVec<double>> d_vselfFieldGateauxDerStrainFDBins;

    /// Compute Gateaux derivative of vself field in bins with respect to affine
    /// strain tensor components
    void
    computeVselfFieldGateauxDerFD();

    /// dftParameters object
    dftParameters *d_dftParamsPtr;

    /// kPoint cartesian coordinates
    std::vector<double> d_kPointCoordinates;

    /// k point crystal coordinates
    std::vector<double> kPointReducedCoordinates;

    /// k point weights
    std::vector<double> d_kPointWeights;

    /// closest tria vertex
    std::vector<dealii::Point<3>> d_closestTriaVertexToAtomsLocation;
    std::vector<dealii::Tensor<1, 3, double>> d_dispClosestTriaVerticesToAtoms;

    /// global k index of lower bound of the local k point set
    unsigned int lowerBoundKindex = 0;
    /**
     * Recomputes the k point cartesian coordinates from the crystal k point
     * coordinates and the current lattice vectors, which can change in each
     * ground state solve dutring cell optimization.
     */
    void
    recomputeKPointCoordinates();

    /// fermi energy
    double fermiEnergy, fermiEnergyUp, fermiEnergyDown, d_groundStateEnergy;

    double d_freeEnergyInitial;

    double d_freeEnergy;

    /// entropic energy
    double d_entropicEnergy;

    // chebyshev filter variables and functions
    // int numPass ; // number of filter passes

    std::vector<double> a0;
    std::vector<double> bLow;

    /// stores flag for first ever call to chebyshev filtering for a given FEM
    /// mesh vector for each k point and spin
    std::vector<bool> d_isFirstFilteringCall;

    std::vector<double> d_upperBoundUnwantedSpectrumValues;

    distributedCPUVec<double> d_tempEigenVec;

    bool d_isRestartGroundStateCalcFromChk;

    /**
     * @ nscf variables
     */
    bool scfConverged;
    void
    nscf(
      KohnShamHamiltonianOperator<memorySpace> &      kohnShamDFTEigenOperator,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver);
    void
    initnscf(
      KohnShamHamiltonianOperator<memorySpace> &     kohnShamDFTEigenOperator,
      poissonSolverProblem<FEOrder, FEOrderElectro> &phiTotalSolverProblem,
      dealiiLinearSolver &                           CGSolver);

    /**
     * @brief compute the maximum of the residual norm of the highest occupied state among all k points
     */
    double
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const double                            _fermiEnergy);


    /**
     * @brief compute the maximum of the residual norm of the highest state of interest among all k points
     */
    double
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const unsigned int                      highestState);


    void
    kohnShamEigenSpaceCompute(
      const unsigned int s,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
        &                                             kohnShamDFTEigenOperator,
      elpaScalaManager &                              elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
      std::vector<double> &                           residualNormWaveFunctions,
      const bool                                      computeResidual,
      const bool                                      isSpectrumSplit = false,
      const bool                                      useMixedPrec    = false,
      const bool                                      isFirstScf      = false);


#ifdef DFTFE_WITH_DEVICE
    void
    kohnShamEigenSpaceCompute(
      const unsigned int s,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolverDevice
        &                  subspaceIterationSolverDevice,
      std::vector<double> &residualNormWaveFunctions,
      const bool           computeResidual,
      const unsigned int   numberRayleighRitzAvoidancePasses = 0,
      const bool           isSpectrumSplit                   = false,
      const bool           useMixedPrec                      = false,
      const bool           isFirstScf                        = false);
#endif


#ifdef DFTFE_WITH_DEVICE
    void
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int s,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolverDevice
        &subspaceIterationSolverDevice);

#endif

    void
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int s,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala);

    void
    kohnShamEigenSpaceComputeNSCF(
      const unsigned int spinType,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
        &                                             kohnShamDFTEigenOperator,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
      std::vector<double> &                           residualNormWaveFunctions,
      unsigned int                                    ipass);
  };

} // namespace dftfe

#endif
