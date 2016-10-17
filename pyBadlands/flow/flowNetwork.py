##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling application.    ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates functions related to Badlands stream network computation.
"""

import math
import numpy
import warnings
import mpi4py.MPI as mpi
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

from pyBadlands.libUtils import SFDalgo as SFD
import pyBadlands.libUtils.sfd as sfd
from pyBadlands.libUtils import FLOWalgo
from pyBadlands.libUtils import FLWnetwork
from pyBadlands.libUtils.resolve_sink import resolve_sink

class flowNetwork:
    """
    Class for handling flow network computation based on Braun & Willett's
    algorithm.
    """

    def __init__(self):
        """
        Initialization.
        """

        self.xycoords = None
        self.base = None
        self.localbase = None
        self.receivers = None
        self.arrdonor = None
        self.delta = None
        self.donors = None
        self.localstack = None
        self.stack = None
        self.partFlow = None
        self.maxdonors = 0
        self.CFL = None
        self.erodibility = None
        self.m = None
        self.n = None
        self.mindt = None
        self.alluvial = 0.
        self.bedrock = 0.
        self.esmooth = 0.
        self.dsmooth = 0.
        self.spl = False
        self.capacity = False
        self.filter = False
        self.depo = 0

        self.discharge = None
        self.localsedflux = None
        self.maxh = None
        self.maxdep = None
        self.diff_flux = None
        self.diff_cfl = None
        self.chi = None
        self.basinID = None

        self.xgrid = None
        self.xgrid = None
        self.xi = None
        self.yi = None
        self.xyi = None

        self._comm = mpi.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    def SFD_receivers(self, fillH, elev, neighbours, edges, distances, globalIDs, sea):
        """
        Single Flow Direction function computes downslope flow directions by inspecting the neighborhood
        elevations around each node. The SFD method assigns a unique flow direction towards the steepest
        downslope neighbor.

        Parameters
        ----------
        variable : fillH
            Numpy array containing the filled elevations from Planchon & Darboux depression-less algorithm.

        variable : elev
            Numpy arrays containing the elevation of the TIN nodes.

        variable : neighbours
            Numpy integer-type array with the neighbourhood IDs.

        variable : edges
            Numpy real-type array with the voronoi edges length for each neighbours of the TIN nodes.

        variable : distances
            Numpy real-type array with the distances between each connection in the TIN.

        variable: globalIDs
            Numpy integer-type array containing for local nodes their global IDs.

        variable : sea
            Current elevation of sea level.
        """

        # Call the SFD function from libUtils
        if self.depo == 0 or self.capacity or self.filter:
            base, receivers, diff_flux = sfd.directions_base(elev, neighbours, edges, distances, globalIDs, sea)

            # Send local base level globally
            self._comm.Allreduce(mpi.IN_PLACE,base,op=mpi.MAX)

            bpos = numpy.where(base >= 0)[0]
            self.base = base[bpos]
            numpy.random.shuffle(self.base)
            # Send local receivers globally
            self._comm.Allreduce(mpi.IN_PLACE,receivers,op=mpi.MAX)
            self.receivers = receivers

            # Send local diffusion flux globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_flux,op=mpi.MAX)
            self.diff_flux = diff_flux
        else:
            base, receivers, maxh, maxdep, diff_flux = sfd.directions(fillH, elev, neighbours, edges, distances, globalIDs, sea)

            # Send local base level globally
            self._comm.Allreduce(mpi.IN_PLACE,base,op=mpi.MAX)
            bpos = numpy.where(base >= 0)[0]
            self.base = base[bpos]
            numpy.random.shuffle(self.base)

            # Send local receivers globally
            self._comm.Allreduce(mpi.IN_PLACE,receivers,op=mpi.MAX)
            self.receivers = receivers

            # Send local maximum height globally
            self._comm.Allreduce(mpi.IN_PLACE,maxh,op=mpi.MAX)
            self.maxh = maxh

            # Send local maximum deposition globally
            self._comm.Allreduce(mpi.IN_PLACE,maxdep,op=mpi.MAX)
            self.maxdep = maxdep

            # Send local diffusion flux globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_flux,op=mpi.MAX)
            self.diff_flux = diff_flux

    def SFD_nreceivers(self, Sc, fillH, elev, neighbours, edges, distances, globalIDs, sea):
        """
        Single Flow Direction function computes downslope flow directions by inspecting the neighborhood
        elevations around each node. The SFD method assigns a unique flow direction towards the steepest
        downslope neighbor. In addition it compute the hillslope non-linear diffusion

        Parameters
        ----------
        variable : Sc
            Critical slope for non-linear diffusion.

        variable : fillH
            Numpy array containing the filled elevations from Planchon & Darboux depression-less algorithm.

        variable : elev
            Numpy arrays containing the elevation of the TIN nodes.

        variable : neighbours
            Numpy integer-type array with the neighbourhood IDs.

        variable : edges
            Numpy real-type array with the voronoi edges length for each neighbours of the TIN nodes.

        variable : distances
            Numpy real-type array with the distances between each connection in the TIN.

        variable: globalIDs
            Numpy integer-type array containing for local nodes their global IDs.

        variable : sea
            Current elevation of sea level.
        """

        # Call the SFD function from libUtils
        if self.depo == 0 or self.capacity or self.filter:
            base, receivers, diff_flux, diff_cfl = SFD.sfdcompute.directions_base_nl(elev, \
                neighbours, edges, distances, globalIDs, sea, Sc)

            # Send local base level globally
            self._comm.Allreduce(mpi.IN_PLACE,base,op=mpi.MAX)

            bpos = numpy.where(base >= 0)[0]
            self.base = base[bpos]
            numpy.random.shuffle(self.base)
            # Send local receivers globally
            self._comm.Allreduce(mpi.IN_PLACE,receivers,op=mpi.MAX)
            self.receivers = receivers

            # Send local diffusion flux globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_flux,op=mpi.MAX)
            self.diff_flux = diff_flux

            # Send local diffusion CFL condition globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_cfl,op=mpi.MIN)
            self.diff_cfl = diff_cfl
        else:
            base, receivers, maxh, maxdep, diff_flux, diff_cfl = SFD.sfdcompute.directions_nl(fillH, \
                elev, neighbours, edges, distances, globalIDs, sea, Sc)

            # Send local base level globally
            self._comm.Allreduce(mpi.IN_PLACE,base,op=mpi.MAX)
            bpos = numpy.where(base >= 0)[0]
            self.base = base[bpos]
            numpy.random.shuffle(self.base)

            # Send local receivers globally
            self._comm.Allreduce(mpi.IN_PLACE,receivers,op=mpi.MAX)
            self.receivers = receivers

            # Send local maximum height globally
            self._comm.Allreduce(mpi.IN_PLACE,maxh,op=mpi.MAX)
            self.maxh = maxh

            # Send local maximum deposition globally
            self._comm.Allreduce(mpi.IN_PLACE,maxdep,op=mpi.MAX)
            self.maxdep = maxdep

            # Send local diffusion flux globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_flux,op=mpi.MAX)
            self.diff_flux = diff_flux

            # Send local diffusion CFL condition globally
            self._comm.Allreduce(mpi.IN_PLACE,diff_cfl,op=mpi.MIN)
            self.diff_cfl = diff_cfl

    def _donors_number_array(self):
        """
        Creates an array containing the number of donors for each node.
        """

        self.arrdonor = None
        numPts = len(self.receivers)
        self.arrdonor = numpy.zeros(numPts, dtype=int)
        maxID = numpy.max(self.receivers)
        self.arrdonor[:(maxID+1)] = numpy.bincount(self.receivers)
        self.maxdonors = self.arrdonor.max()

        return

    def _delta_array(self):
        """
        Creates the "delta" array, which is a list containing, for each
        node, the array index where that node's donor list begins.
        """

        self.delta = None
        nbdonors = len(self.arrdonor)
        self.delta = numpy.zeros( nbdonors+1, dtype=int)
        self.delta.fill(nbdonors)
        self.delta[-2::-1] -= numpy.cumsum(self.arrdonor[::-1])

        return

    def ordered_node_array(self):
        """
        Creates an array of node IDs that is arranged in order from downstream
        to upstream.
        """

        # Build donors array for each node
        self._donors_number_array()

        # Create the delta array
        self._delta_array()

        # Using libUtils stack create the ordered node array
        self.donors,lstcks = FLWnetwork.fstack.build(self.localbase,self.receivers,self.delta)
        # Create local stack
        stids = numpy.where(lstcks > -1 )[0]
        self.localstack = lstcks[stids]

    def compute_flow(self, Acell, rain):
        """
        Calculates the drainage area and water discharge at each node.

        Parameters
        ----------
        variable : Acell
            Numpy float-type array containing the voronoi area for each nodes (in m2)

        variable : rain
            Numpy float-type array containing the precipitation rate for each nodes (in m/a).
        """

        numPts = len(Acell)

        self.discharge = numpy.zeros(numPts, dtype=float)
        self.discharge[self.stack] = Acell[self.stack] * rain[self.stack]

        # Compute discharge using libUtils
        self.discharge = FLOWalgo.flowcompute.discharge(self.localstack, self.receivers, self.discharge)
        self._comm.Allreduce(mpi.IN_PLACE, self.discharge, op=mpi.MAX)

    def compute_parameters(self):
        """
        Calculates the catchment IDs and the Chi parameter (Willett 2014).
        """

        # Initialise MPI communications
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get basin starting IDs for each local partition
        cumbase = numpy.zeros(size+1)
        for i in range(size):
            cumbase[i+1] = len(numpy.array_split(self.base, size)[i])+cumbase[i]+1

        # Compute discharge using libUtils
        splexp = self.m / self.n
        chi, basinID = FLOWalgo.flowcompute.parameters(self.localstack,self.receivers,
                                               self.discharge,self.xycoords,splexp,cumbase[rank])
        comm.Allreduce(mpi.IN_PLACE,chi,op=mpi.MAX)
        comm.Allreduce(mpi.IN_PLACE,basinID,op=mpi.MAX)

        self.chi = chi
        self.basinID = basinID

        return

    def compute_sedflux(self, Acell, elev, fillH, xymin, xymax, diff_flux, dt, sealevel, cumdiff, neighbours=None):
        """
        Calculates the sediment flux at each node.

        Parameters
        ----------
        variable : Acell
            Numpy float-type array containing the voronoi area for each nodes (in m2)

        variable : elev
            Numpy arrays containing the elevation of the TIN nodes.

        variable : fillH
            Numpy array containing the filled elevations from Planchon & Darboux depression-less algorithm.

        variable : xymin
            Numpy array containing the minimal XY coordinates of TIN grid (excuding border cells).

        variable : xymax
            Numpy array containing the maximal XY coordinates of TIN grid (excuding border cells).

        variable : diff_flux
            Numpy arrays representing the fluxes due to linear diffusion equation.

        variable : dt
            Real value corresponding to the maximal stability time step.

        variable : sealevel
            Real value giving the sea-level height at considered time step.

        variable : cumdiff
            Numpy array containing the cumulative deposit thicknesses.
        """

        # Initialise MPI communications
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Compute sediment flux using libUtils

        diff = None  # not all code paths use this, so return None in the default case

        # Parallel case
        if(size > 1):
            # Purely erosive case
            if self.spl and self.depo == 0:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_ero_only(self.localstack,self.receivers, \
                                      self.xycoords,xymin,xymax,self.discharge,elev, \
                                      diff_flux,self.erodibility,self.m,self.n,sealevel,dt)

            # Stream power law and mass is not conserved
            elif self.spl and self.filter:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_nocapacity_quick(self.localstack,self.receivers, \
                         self.xycoords,Acell,xymin,xymax,self.discharge,elev,diff_flux,self.erodibility, \
                         self.m,self.n,sealevel,dt)

            # Stream power law
            elif self.spl:
                print "PARALLEL NOT IMPLEMENTED"
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_nocapacity(self.localstack,self.receivers,self.xycoords, \
                         Acell,xymin,xymax,self.maxh,self.maxdep,self.discharge,fillH,elev,diff_flux, \
                         self.erodibility,self.m,self.n,sealevel,dt)

            # River carrying capacity case
            else:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_capacity(self.localstack,self.receivers,self.xycoords,\
                         Acell,xymin,xymax,self.discharge,elev,diff_flux,cumdiff,self.erodibility, \
                         self.m,self.n,self.bedrock,self.alluvial,sealevel,dt)

            timestep = numpy.zeros(1)
            timestep[0] = newdt
            comm.Allreduce(mpi.IN_PLACE,timestep,op=mpi.MIN)
            newdt = timestep[0]
            comm.Allreduce(mpi.IN_PLACE,sedflux,op=mpi.MAX)
            tempIDs = numpy.where(sedflux < -9.5e5)
            sedflux[tempIDs] = 0.
            newdt = max(self.mindt,newdt)
            sedrate = sedflux

        # Serial case
        else:
            # Purely erosive case
            if self.spl and self.depo == 0:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_ero_only(self.localstack,self.receivers, \
                                      self.xycoords,xymin,xymax,self.discharge,elev, \
                                      diff_flux,self.erodibility,self.m,self.n,sealevel,dt)

            # Stream power law and mass is not conserved
            elif self.spl and self.filter:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_nocapacity_quick(self.localstack,self.receivers, \
                         self.xycoords,Acell,xymin,xymax,self.discharge,elev,diff_flux,self.erodibility, \
                         self.m,self.n,sealevel,dt)

            # Stream power law
            elif self.spl:
                diff, newdt = self._single_catchment_fill(areas=Acell, xymin=xymin, xymax=xymax, elev=elev, max_dt=dt, sea=sealevel, diff_flux=diff_flux, neighbours=neighbours)
                sedflux = None

            # River carrying capacity case
            else:
                sedflux, newdt = FLOWalgo.flowcompute.sedflux_capacity(self.localstack,self.receivers,self.xycoords,\
                         Acell,xymin,xymax,self.discharge,elev,diff_flux,cumdiff,self.erodibility, \
                         self.m,self.n,self.bedrock,self.alluvial,sealevel,dt)

            tempIDs = numpy.where(sedflux < -9.5e5)
            if sedflux is not None:
                sedflux[tempIDs] = 0.
            newdt = max(self.mindt,newdt)
            sedrate = sedflux

        return newdt, sedrate, diff

    def gaussian_filter(self, diff):
        """
        Gaussian filter operation used to smooth erosion and deposition
        thicknesses for large simulation time steps. Using this operation
        implies that the resulting simulation is not conserving mass.

        Parameters
        ----------
        variable : diff
            Numpy arrays containing the erosion and deposition thicknesses.
        """

        K = 3

        if self.xgrid is None:
            dx = self.xycoords[1,0] - self.xycoords[0,0]
            xmin, xmax = min(self.xycoords[:,0]), max(self.xycoords[:,0])
            ymin, ymax = min(self.xycoords[:,1]), max(self.xycoords[:,1])
            self.xgrid = numpy.arange(xmin,xmax+dx,dx)
            self.ygrid = numpy.arange(ymin,ymax+dx,dx)
            self.xi, self.yi = numpy.meshgrid(self.xgrid, self.ygrid)

            # querying the cKDTree later becomes a bottleneck, so distribute the xyi array across all MPI nodes
            xyi = numpy.dstack([self.xi.flatten(), self.yi.flatten()])[0]
            splits = numpy.array_split(xyi, self._size)
            self.split_lengths = numpy.array(map(len, splits)) * K
            self.localxyi = splits[self._rank]
            self.query_shape = (xyi.shape[0], K)

        depZ = numpy.copy(diff)
        depZ = depZ.clip(0.)

        eroZ = numpy.copy(diff)
        eroZ = eroZ.clip(max=0.)

        tree = cKDTree(self.xycoords[:,:2])

        # Querying the KDTree is rather slow, so we split it across MPI nodes
        # FIXME: the Allgatherv fails if we don't flatten the array first - why?
        nelems = self.query_shape[0] * self.query_shape[1]
        indices = numpy.empty(self.query_shape, dtype=numpy.int64)
        localdistances, localindices = tree.query(self.localxyi, k=K)

        distances_flat = numpy.empty(nelems, dtype=numpy.float64)
        self._comm.Allgatherv(numpy.ravel(localdistances), [distances_flat, (self.split_lengths, None)])

        indices_flat = numpy.empty(nelems, dtype=numpy.int64)
        self._comm.Allgatherv(numpy.ravel(localindices), [indices_flat, (self.split_lengths, None)])

        distances = distances_flat.reshape(self.query_shape)
        indices = indices_flat.reshape(self.query_shape)

        if len(depZ[indices].shape) == 3:
            zd_vals = depZ[indices][:,:,0]
            ze_vals = eroZ[indices][:,:,0]
        else:
            zd_vals = depZ[indices]
            ze_vals = eroZ[indices]
        zdi = numpy.average(zd_vals,weights=(1./distances), axis=1)
        zei = numpy.average(ze_vals,weights=(1./distances), axis=1)

        onIDs = numpy.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zdi[onIDs] = depZ[indices[onIDs,0]]
            zei[onIDs] = eroZ[indices[onIDs,0]]

        depzi = numpy.reshape(zdi,(len(self.ygrid),len(self.xgrid)))
        erozi = numpy.reshape(zei,(len(self.ygrid),len(self.xgrid)))

        smthDep = gaussian_filter(depzi, sigma=self.dsmooth)
        smthEro = gaussian_filter(erozi, sigma=self.esmooth)

        rgi_dep = RegularGridInterpolator((self.ygrid, self.xgrid), smthDep)
        zdepsmth = rgi_dep((self.xycoords[:,1],self.xycoords[:,0]))
        rgi_ero = RegularGridInterpolator((self.ygrid, self.xgrid), smthEro)
        zerosmth = rgi_ero((self.xycoords[:,1],self.xycoords[:,0]))

        return zdepsmth + zerosmth

    def dt_stability(self, elev, locIDs):
        """
        This function computes the maximal timestep to ensure computation stability
        of the flow processes. This CFL-like condition is computed using erodibility
        coefficients, discharges plus elevations and distances between TIN nodes and
        their respective reveivers.

        Parameters
        ----------
        variable : elev
            Numpy arrays containing the elevation of the TIN nodes.

        variable: locIDs
            Numpy integer-type array containing for local nodes their global IDs.
        """

        # Initialise MPI communications
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Compute the local value for time stability
        dt = FLOWalgo.flowcompute.flowcfl(locIDs,self.receivers,self.xycoords,elev, \
                                      self.discharge,self.erodibility,self.m,self.n)

        # Global mimimum value for diffusion stability
        CFL = numpy.zeros(1)
        CFL[0] = dt
        comm.Allreduce(mpi.IN_PLACE,CFL,op=mpi.MIN)
        self.CFL = CFL[0]

    def sedflux_nocapacity_proposal(self, elev, xymin, xymax, dt, sea, areas, fillH, diff_flux):
        '''
        Returns sedflux, newdt
        '''

        newdt = dt
        change = numpy.empty_like(elev)
        change.fill(-1.e6)

        sedFluxes = numpy.zeros_like(elev)

        # for each node in RANDOM ORDER, we iterate over FLOWS from donor to receiver
        print 'there are %s nodes' % len(self.localstack)
        for stack_index in range(len(self.localstack) - 1, -1, -1):
            # print 'index %s' % stack_index
            SPL = 0.
            donor = self.stack[stack_index]  # get our CURRENT NODE INDEX
            recvr = self.receivers[donor]  # get the corresponding RECEIVER NODE INDEX
            dh = 0.95 * (elev[donor] - elev[recvr])

            if elev[donor] > sea and elev[recvr] < sea:
                dh = elev[donor] - sea

            if dh < 0.001:
                dh = 0.
            waterH = fillH[donor] - elev[donor]

            # Compute stream power law
            if recvr != donor and dh > 0.:
                if waterH == 0. and elev[donor] >= sea:
                    dist = math.sqrt((self.xycoords[donor, 0] - self.xycoords[recvr, 0]) ** 2.0 + (self.xycoords[donor, 1] - self.xycoords[recvr, 1]) ** 2.0)
                    if dist > 0.:
                        SPL = -self.erodibility[donor] * self.discharge[donor] ** self.m * (dh / dist) ** self.n

            maxh = self.maxh[donor]
            if elev[donor] < sea:
                maxh = sea - elev[donor]
            elif waterH > 0.:
                maxh = min(waterH, maxh)

            maxh *= 0.95
            Qs = 0.

            # Deposition case
            if SPL == 0. and areas[donor] > 0.:  # if no stream power AND donor has finite drainage area ??? what does this mean?
                if maxh > 0. and elev[donor] < sea:  # if there's water AND we are under the sea - *** this sounds like an obscure case
                    if sedFluxes[donor] * newdt / areas[donor] < maxh:  # will the sediment deposition stay under the water/sea level?
                        SPL = sedFluxes[donor] / areas[donor]  # set new stream power (?) what does this mean? 
                        Qs = 0.  # no new flow on this node
                    else:
                        SPL = maxh / newdt  # maximum SPL?
                        Qs = sedFluxes[donor] - SPL * areas[donor]  # maximum flow/deposition on receiver?

                # Fill depression
                elif waterH > 0.0001 and donor != recvr:  # if there is water AND flow is between two different nodes
                    dh = 0.95 * waterH
                    if sedFluxes[donor] * newdt / areas[donor] < dh:  # if deposition stays under water level
                        SPL = sedFluxes[donor] / areas[donor]  # stream power is determined by ???
                        Qs = 0.  # no new flow on receiver
                    else:
                        # same as above - use maximum flow
                        SPL = dh / newdt
                        Qs = sedFluxes[donor] - SPL * areas[donor]

                # Base-level (sink)
                elif donor == recvr and areas[donor] > 0.:  # if this is a sink
                    SPL = sedFluxes[donor] / areas[donor]
                    Qs = 0.
                else:
                    Qs = sedFluxes[donor]

            # Erosion case
            elif SPL < 0.:  # if we are ERODING i.e. negative power (?)
                if elev[donor] > sea and elev[recvr] < sea:  # if donor is above the sea and receiver is under the sea
                    # print *, "case A", newdt, mtime
                    newdt = min(newdt, -0.99 * (elev[donor] - sea) / SPL)  # dt is limited because we don't want to reverse gradient (?)

                if -SPL * newdt > elev[donor] - elev[recvr]:  # dt is limited because we don't want to reverse gradient (?)
                    # print *, "case B", newdt, mtime
                    newdt = min(newdt, -0.99 * (elev[donor] - elev[recvr]) / SPL)

                Qs = -SPL * areas[donor] + sedFluxes[donor]  # receiver gets additional deposit depending on spl/area of donor and all sediment on donor

            # Update sediment flux in receiver node
            sedFluxes[recvr] += Qs  # receiver gets more sediment
            # print 'apply %s to %s which now has %s' % (Qs, recvr, sedFluxes[recvr])
            change[donor] = SPL + diff_flux[donor]  # set donor spl (happens once per donor)

            # Update borders
            if self.xycoords[donor, 0] < xymin[0] or self.xycoords[donor, 1] < xymin[1] or self.xycoords[donor, 0] > xymax[0] or self.xycoords[donor, 1] > xymax[1]:
                change[donor] = 0.

            # Update base levels
            if donor == recvr and change[donor] > 0.:  # this is a sink node AND pyChange is set
                if waterH == 0.:  # no water
                    mtime = self.maxdep[donor] / change[donor]  # pymaxd is depth of flow (difference in height from donor to receiver). TODO what is pychange?
                    # i guess pychange is rate of change - maxdepth / rate of change = max years - don't fill past level?
                    # this determines mtime (maxtime)
                    # this MIGHT BE the main constraint *********
                    # print *, "case C", newdt, mtime
                    newdt = min(newdt, mtime)
                else:
                    mtime = waterH / change[donor]  # pymaxd
                    # this is WATER case
                    # don't fill past water?
                    # THIS IS THE MAIN CONSTRAINT. The other cases are not touched!
                    print "case D, node %s %s %s" % (newdt, mtime, donor)
                    newdt = min(newdt, mtime)

        # return sedFluxes, newdt
        return change, newdt

    def _volume_of_nodes(self, areas, node_id_set, limit_elev, elev):
        """
        Returns the volume of the nodes specified in node_id_set up to the
        limit elevation.
        """

        volume = 0.0
        for nid in node_id_set:
            volume += areas[nid] * abs(limit_elev - elev[nid])

        return volume

    def _pop_lowest_node(self, elev, unvisited_ids):
        """
        Determine which node from unvisited_ids is lowest, remove it from
        unvisited_ids and return it.
        """
        id_array = numpy.array(list(unvisited_ids))
        unvisited_elevs = elev[id_array]
        lowest_index = numpy.argmin(unvisited_elevs)
        lowest_id = id_array[lowest_index]
        unvisited_ids.remove(lowest_id)
        return lowest_id

    def _get_pythonic_neighbours(self, nid, neighbours):
        """
        Returns an iterable list of neighbour nodes to 'nid'.

        This is useful because the format used for the FORTRAN code is
        difficult to manipulate from Python code.
        """
        neigh = neighbours[nid]
        return neigh[neigh >= 0]

    def _catchment_capacity(self, sink_id, elev, max_volume_needed, areas, neighbours):
        """
        Determine the catchment volume and sill node for the catchment
        originating at a given sink node.

        Returns (volume, sill_node_id)
        """
        volume = 0.0
        sill_node_id = None
        sill_node_elev = None

        # TODO: obviously, some prioqueues are going to make this a lot faster
        nodes_in_catchment = set([sink_id])
        visited_nodes = set()
        unvisited_nodes = set([sink_id])

        while len(unvisited_nodes) and volume < max_volume_needed and sill_node_id is None:
            lowest_id = self._pop_lowest_node(elev=elev, unvisited_ids=unvisited_nodes)
            visited_nodes.add(lowest_id)
            lowest_elev = elev[lowest_id]

            # NOTE: this scales O(N^2) with node count. There are definitely more efficient algorithms!
            volume = self._volume_of_nodes(areas, nodes_in_catchment, lowest_elev, elev)
            if not(volume > 0.0 or len(nodes_in_catchment) == 1):
                import pdb; pdb.set_trace()
            assert volume > 0.0 or len(nodes_in_catchment) == 1

            # does it drain into the same catchment?
            if self.receivers[lowest_id] in nodes_in_catchment:
                nodes_in_catchment.add(lowest_id)
                assert lowest_id not in unvisited_nodes
                # print 'readd %s' % lowest_id
                if lowest_id not in visited_nodes:
                    unvisited_nodes.add(lowest_id)

                # add any neighbours to our 'unvisited' list
                for nid in self._get_pythonic_neighbours(lowest_id, neighbours):
                    if nid not in visited_nodes:
                        unvisited_nodes.add(nid)

            elif sill_node_id is None or lowest_elev < sill_node_elev:  # It drains elsewhere. Could it be our sill node?
                sill_node_id = lowest_id
                sill_node_elev = lowest_elev

        return volume, sill_node_id

    def _resolve_sink(self, sinks, nid, elev, sea):
        """
        For a given node id 'nid', figure out its sink node id. Update 'sinks'
        and return the resolved sink node id.

        If we reach an undersea node, deposition occurs there.

        Updates 'sinks' in-place.

        Returns the sink node id.

        This calls out to a fast C implementation.
        """

        assert nid >= 0

        # This is the common case, so short-circuit if possible
        if sinks[nid] >= 0:
            return sinks[nid]

        # Call the C code to do a search
        sink_id = resolve_sink(self.receivers, sinks, nid, sea, elev)
        assert sink_id >= 0
        return sink_id

    def _distribute_sediment_land(self, sink_id, elev, sill_id, deposition_change, deposition_volume, sinks, areas, neighbours, sea):
        """
        For a given node is 'sink_id' and its deposition rate 'rate', determine
        how to distribute the sediment volume within the catchment.

        Modifies 'deposition_change' in-place.

        elev is the elevation of each node including any erosion.

        Discards any excess sediment that does not fit within the catchment.
        """

        # Strategy:
        #
        # Starting from the lowest node and working our way up to the highest
        # within the catchment (bounded by the sill), we raise one node at a
        # time to the elevation of the next highest (minus epsilon, to ensure
        # we still have the same sink and slope characteristics after filling).
        # Each time we raise a node, we recalculate the volume consumed by the
        # newly raised catchment. All nodes that have been raised are raised at
        # the same time.
        #
        # IF we have consumed less than the requisite volume of sediment, we
        # raise the next-lowest node.
        # IF we have consumed more than enough sediment, we lower all of the
        # touched nodes by a small amount so as to consume exactly the right
        # amount of sediment.
        # IF we run out of nodes to raise and there is still sediment
        # remaining, we 'top off' the catchment so as to eliminate the
        # depression/sink in it. We then FOR THIS VERSION OF THE ALGORITHM ONLY
        # discard any excess sediment.

        raised_ids = set()  # nodes which have been raised to perform this deposition
        # touched_ids = set()
        visited_ids = set([sink_id])
        unvisited_ids = set([sink_id])
        dv = deposition_volume[sink_id]

        volume = 0.0  # volume of sediment that we have distributed

        while len(unvisited_ids) and volume < dv:
            # again, prioqueues would help here
            lowest_id = self._pop_lowest_node(elev=elev, unvisited_ids=unvisited_ids)
            lowest_elev = elev[lowest_id]

            # add any potential neighbours to our 'unvisited' list
            # this guarantees that the next-lowest node in the catchment is available to be visited
            for nid in self._get_pythonic_neighbours(lowest_id, neighbours):
                if nid not in visited_ids:
                    visited_ids.add(nid)
                    if self._resolve_sink(sinks, nid, elev, sea) == sink_id:
                        unvisited_ids.add(nid)

            # NOTE: this scales O(N^2) with node count. There are definitely more efficient algorithms!
            volume = self._volume_of_nodes(areas, raised_ids, lowest_elev, elev)

            if elev[lowest_id] < sea:
                assert False, 'tried to raise a sea node'
                # If you reach this, all excess sediment should go into the sea. Don't think it's reachable, though.

            raised_ids.add(lowest_id)

        if len(unvisited_ids) == 0:
            # we ran out of nodes to raise
            print 'DISCARDING SEDIMENT volume %s on top of %s' % (dv - volume, volume)
        # else, we did not fill the catchment

        # work out how to fill the nodes
        # we go from highest to lowest and assign them all to the elevation of the highest, minus epsilon. This maintains the same drainage structure.
        # NOTE: there is a slight error in the resulting volume calculation; we will log it for now and try to improve it later.
        # NOTE: we haven't yet handled the case where the next iteration will try to fill these same nodes as we haven't truly filled the depression.

        # pull the raised ids into a list so we can sort them descending by elevation
        raised_ids_array = numpy.array(list(raised_ids))
        raised_ids_elevs = elev[raised_ids_array]
        # sort
        descending_elev_nodes = raised_ids_array[numpy.argsort(raised_ids_elevs)[::-1]]
        highest_elev = elev[descending_elev_nodes[0]]

        # work out how much to change their deposition amount by
        # we want to raise them all to (highest_elev - count * epsilon)
        epsilon = 0.00001  # NOTE: we could use next_after, but that is likely to lead to numerical stability issues; we really want the flow network to be preserved
        allocated_volume = 0.0
        for i in range(len(descending_elev_nodes)):
            nid = descending_elev_nodes[i]

            new_elev = highest_elev - i * epsilon

            assert(numpy.all(deposition_change[nid] == 0.0))  # we have no way to deal with interconnected flow networks yet

            delta = new_elev - elev[nid]
            delta_volume = delta * areas[nid]

            if allocated_volume + delta_volume > dv:
                # scale it back
                remaining_volume = dv - allocated_volume
                delta = remaining_volume / areas[nid]
                delta_volume = remaining_volume
                break
            else:
                allocated_volume += delta_volume

            deposition_change[nid] = delta

    def _distribute_sediment_sea(self, elev, deposition_change, deposition_volume, areas, neighbours, sea):
        '''
        Resolves ALL of the deposits under sea. We treat them as an isolated
        system as there's no way for sediment to 'build up' and become land.

        For under-sea deposition, we fill the sink node up to the sea level
        (less epsilon). If there's too much, we send any excess to the
        neighbour with the steepest slope.

        deposition_volume reflects the initial or 'requested' deposition state
        where most nodes will have too much sediment. deposition_change is
        updated after resolution.
        '''

        unresolved_ids = numpy.where(numpy.logical_and(deposition_volume > 0.0, elev < sea))[0]

        epsilon = 0.000001

        for sid in numpy.nditer(unresolved_ids):
            # how much sediment do we need to deposit on it?
            # dv tracks how much sediment remains to be deposited
            dv = deposition_volume[sid]

            maxh = sea  # at the start of the chain, raise to sea level (minus epsilon)

            last_sid = None
            while dv > 0.0:
                assert sid != last_sid, '%s %s' % (last_sid, sid)
                last_sid = sid

                maxh -= epsilon  # on each step, raise to almost as high to retain slopes everywhere
                a = areas[sid]

                # Determine the highest absolute value we can raise this node
                # Don't raise beyond the sea level OR the donor node. This ensures
                # there is always a downslope. It also ensures we do not create new
                # undersea depressions.

                maxraise = (maxh - elev[sid] - deposition_change[sid])  # the most we can raise this node
                capacity = maxraise * a

                if capacity < 0.0:
                    # just send excess to its receiver
                    # receiver id that will take excess
                    rid = self.receivers[sid]
                    if sid == rid:
                        print 'found undersea sink a, discarding %s' % dv
                        dv = 0.0
                    else:
                        sid = rid
                    continue

                if capacity - dv > 0:  # if it all fits
                    # fill a bit and carry on
                    deposition_change[sid] += dv / a
                    dv = 0.0
                else:
                    # It doesn't all fit. Assign what we can and look for somewhere else to deposit.
                    deposition_change[sid] += maxraise

                    # receiver id that will take excess
                    rid = self.receivers[sid]
                    if sid == rid:
                        print 'found undersea sink b, discarding %s' % dv
                        dv = 0.0
                    else:
                        dv -= capacity
                        sid = rid

        # TODO: make sure erosion doesn't push a node under the sea level
        # TODO: how do you ensure that node ordering is maintained? how does the existing code handle this?

    def _single_catchment_fill(self, elev, xymin, xymax, max_dt, sea, areas, diff_flux, neighbours):
        '''
        Deposition algorithm as described at ...

        Sets the timestep such that a single catchment is filled on each
        timestep. This is faster than the existing algorithm which fills a
        little bit up to the water level but requires many more timesteps to
        progress the simulation to the same point.

        PARAMETERS
        max_dt: maximum timestep in A, determined by previous steps in the model
        elev: numpy array giving elevation of each TIN node
        sea: scalar sea level

        RETURN VALUES
        elev_change: numpy array giving change in elevation for each node
        dt: length of the timestep in A. This must not be changed by subsequent
            steps in the model.
        '''

        dt = max_dt

        # rate of change on each node
        change = numpy.empty_like(elev)
        change.fill(-1.e6)

        erosion_rate = numpy.empty_like(elev)
        erosion_rate.fill(-1.e6)

        deposition_volume_rate = numpy.zeros_like(elev)

        # For each node, we track the sink node (endpoint of any water runoff).
        # -1 means 'unknown', 0+ are node ids
        # so sinks[123] is the sink node id for node 123
        sinks = numpy.empty(elev.shape, dtype=int)
        sinks.fill(-1)

        # for each flow...
        # we're going to iteratively calculate sediment flow down hills, so we iterate from highest to lowest node
        ordered_ids = numpy.argsort(elev)
        # '[::-1]' reverses sort order
        for donor, recvr in [(donor, self.receivers[donor]) for donor in ordered_ids[::-1]]:
            # we have a sink node (bottom of the drainage network) where donor drains into itself
            is_sink = (donor == recvr)

            dh = elev[donor] - elev[recvr]
            if elev[donor] > sea and elev[recvr] < sea:
                dh = elev[donor] - sea

            if dh < 0.001:
                dh = 0.0  # FLOWalgo.f90:347

            # 1. CALCULATE EROSION/DEPOSITION ON EACH NODE AND ANY TIMESTEP CONSTRAINTS
            rate = 0.0
            # TODO what do we do if these conditions are not met?
            if not is_sink and dh > 0.0 and elev[donor] >= sea:
                dist = math.sqrt((self.xycoords[donor, 0] - self.xycoords[recvr, 0]) ** 2.0 + (self.xycoords[donor, 1] - self.xycoords[recvr, 1]) ** 2.0)

                # Node erosion rate (SPL): Braun 2013 eqn (2), measured in HEIGHT per year
                rate = -self.erodibility[donor] * self.discharge[donor] ** self.m * (dh / dist) ** self.n
                rate += diff_flux[donor]   # integrate linear diffusion approximation (?)
                rate = min(rate, 0.0)  # just in case it goes negative due to diff_flux

                # If we fill at this rate, will we fill the depression past level?
                # We don't want to do this as we will change the flow network
                if rate > 0.0:
                    olddt = dt
                    dt = min(dt, -0.999 * dh / rate)
                    if olddt != dt:
                        print 'dt = %s because rate=%s, dh=%s' % (dt, rate, dh)
                '''
                if -rate * dt > elev[donor] - elev[recvr]:
                    # print *, "case B", newdt, mtime
                    # TODO: you could improve this by choosing the rate so as to fit halfway between the two nodes with closest elevation
                    # FIXME: inconsistency between elev[donor]-elev[recvr] and then use of dh?
                    # dt = min(dt, -0.999 * dh / rate)
                    dt = min(dt, -0.999 * (elev[donor] - elev[recvr]) / (rate * areas[donor]))
                    assert dt > 0
                    print 'DEBUG: flow reversal constraint on node %s. new dt is %s' % (donor, dt)
                    # BUT FIXME: the receiver node will probably erode as well, changing the dh, so it might be worth waiting before we determine the dh
                    # so then, shouldn't we work out net rate of change on every node, and then for any nodes with net erosion, set dt to ensure that we don't change flow network? but this a simultaneous equation type situation; how much erosion is too much? unless you just solve it iteratively over timesteps
                    '''

                # TODO: what about the same constraint but where you erode a node so far that the gradient reverses?

                erosion_rate[donor] = rate  # we erode material from the donor...  (HEIGHT per year)
                # TODO: this should also include the diff_flux parameter
                # erosion_rate[donor] = rate * areas[donor]  # we erode material from the donor...  (HEIGHT per year)
                assert erosion_rate[donor] <= 0.0

                # if donor == 37610:
                    # print 'ero %s = %s/year' % (donor, erosion_rate[donor] / dt)

                # print donor, rate, areas[donor], erosion_rate[donor]

                # what's happening is that we're getting too much erosion on a few nodes and the final elev_change is really huge. 
                # how much is too much erosion?

                # We will deposit the same amount, but we need to work out where to deposit it.
                # Do we already know the sink node (bottom point) for the receiver in question?
                sink_id = self._resolve_sink(sinks, recvr, elev, sea)
                assert sink_id >= 0
                deposition_volume_rate[sink_id] -= erosion_rate[donor] * areas[donor]  # VOLUME per year
                # we verify that deposition_rate is always positive below
                # assert deposition_rate[sink_id] >= 0.0
            else:
                erosion_rate[donor] = 0.0  # DEBUG ONLY

        # From this point, dt is fixed for the rest of the tick

        # elev_change is the elevation change given the known timestep
        elev_change = erosion_rate * dt
        new_elev = elev + elev_change

        deposition_change = numpy.zeros_like(elev_change)

        # We know the *rate* of deposition on each node, but it is likely that
        # that will overfill some of the catchments. For any nodes with
        # positive deposition, work out the catchment size and redistribute any
        # excess sediment.

        # The only nodes with positive deposition should be sink nodes.
        # assert(numpy.all(deposition_volume_rate >= 0.0))
        land_sinks = numpy.argwhere(numpy.logical_and(deposition_volume_rate > 0.0, elev >= sea))
        # assert numpy.all(self.receivers[deposition_sinks] == deposition_sinks)
        catchment_volume = {}  # sink_id: catchment volume
        catchment_sill = {}  # sink_id: sill node id for that catchment

        # Work out the catchment volume of each sink - land only
        for sink_id_array in land_sinks:
            sink_id = int(sink_id_array)
            if elev[sink_id] < sea:
                continue  # skip if it's undersea

            # NOTE: if we spill deposition later, this will need to be changed as we could have EVEN MORE deposition to apply
            max_volume_needed = deposition_volume_rate[sink_id] * dt
            catchment_volume[sink_id], catchment_sill[sink_id] = self._catchment_capacity(sink_id, new_elev, max_volume_needed, areas, neighbours)

        # TODO FUTURE: At this point, you want to work out how to distribute excess sediment that does not fit in a catchment
        # Right now we just discard any excess sediment
        # We also resolve all deposition in a single pass

        assert(numpy.all(new_elev <= elev))

        deposition_volume = deposition_volume_rate * dt

        for sink_id_array in land_sinks:
            sink_id = int(sink_id_array)

            if elev[sink_id] < sea:
                assert deposition_change[sink_id] == 0
                self._distribute_sediment_land(sink_id, deposition_change=deposition_change, elev=new_elev, deposition_volume=deposition_volume, sill_id=catchment_sill, sinks=sinks, areas=areas, neighbours=neighbours, sea=sea)

        self._distribute_sediment_sea(deposition_change=deposition_change, elev=new_elev, deposition_volume=deposition_volume, areas=areas, neighbours=neighbours, sea=sea)

        elev_change += deposition_change

        return elev_change, dt

    # NOTE: we're using mindt, but that doesn't really make sense given we're passing back the absolute different in elevations. We should be adhering to mindt.
