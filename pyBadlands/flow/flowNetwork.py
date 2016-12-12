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

    def compute_sedflux(self, Acell, elev, fillH, xymin, xymax, diff_flux, dt, rivqs, sealevel, cumdiff, perc_dep, slp_cr, FVmesh, globalIDs, neighbours=None):
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

        variable : rivqs
            Numpy arrays representing the sediment fluxes from rivers.

        variable : sealevel
            Real value giving the sea-level height at considered time step.

        variable : cumdiff
            Numpy array containing the cumulative deposit thicknesses.

        variable : slp_cr
            Critical slope used to force aerial deposition for alluvial plain.

        variable : perc_dep
            Maximum percentage of deposition at any given time interval.
        """

        # Initialise MPI communications
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Compute sediment flux using libUtils
        # Purely erosive case
        if self.spl and self.depo == 0:
            sedflux, newdt = FLOWalgo.flowcompute.sedflux_ero_only(self.localstack,self.receivers, \
                     self.xycoords,xymin,xymax,self.discharge,elev, \
                     diff_flux,self.erodibility,self.m,self.n,sealevel,dt)

        # Stream power law and mass is not conserved
        elif self.spl and self.filter:
            sedflux, newdt = FLOWalgo.flowcompute.sedflux_nocapacity_quick(self.localstack,self.receivers, \
                     self.xycoords,Acell,xymin,xymax,self.discharge,elev,rivqs,diff_flux,self.erodibility, \
                     self.m,self.n,sealevel,dt)

        # Stream power law
        elif self.spl:
            sedflux, newdt = FLOWalgo.flowcompute.sedflux_nocapacity(self.localstack,self.receivers,self.xycoords, \
                     Acell,xymin,xymax,self.maxh,self.maxdep,self.discharge,fillH,elev,rivqs,diff_flux, \
                     self.erodibility,self.m,self.n,perc_dep,slp_cr,sealevel,dt)

        # River carrying capacity case
        else:
            sedflux, newdt = FLOWalgo.flowcompute.sedflux_capacity(self.localstack,self.receivers,self.xycoords,\
                     Acell,xymin,xymax,self.discharge,elev,rivqs,diff_flux,cumdiff,self.erodibility, \
                     self.m,self.n,self.bedrock,self.alluvial,sealevel,dt)

        # Parallel case
        if(size > 1):
            timestep = numpy.zeros(1)
            timestep[0] = newdt
            comm.Allreduce(mpi.IN_PLACE,timestep,op=mpi.MIN)
            newdt = timestep[0]
            comm.Allreduce(mpi.IN_PLACE,sedflux,op=mpi.MAX)
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
                diff, newdt = self._single_catchment_fill(areas=Acell, xymin=xymin, xymax=xymax, pre_elev=elev, max_dt=dt, sea=sealevel, diff_flux=diff_flux, neighbours=neighbours, fillH=fillH, FVmesh=FVmesh, globalIDs=globalIDs)
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
        # print 'there are %s nodes' % len(self.localstack)
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

        If any nodes have elevations above the limit, they are not considered.
        """

        # FIXME: this is copy-pasted from the land sedimentation function

        if len(node_id_set) == 0:
            return 0.0

        nids = numpy.array(list(node_id_set))
        nids = nids[elev[nids] < limit_elev]  # remove nodes which are above limit

        elevs = elev[nids]
        areas = areas[nids]

        assert numpy.all(elevs <= limit_elev), 'elevs %s, limit %s' % (elevs, limit_elev)

        # we sort the nodes so we can keep the flow network the same
        descending_indices = numpy.argsort(elevs)[::-1]

        # calculate the offset from max that we assign each node (offsets)
        epsilon = 0.001
        offsets = numpy.zeros_like(elevs)
        # TODO this would be faster with an argsort then index * epsilon
        offset = 0.0
        for index in numpy.nditer(descending_indices, flags=('zerosize_ok',)):
            offsets[index] = offset
            offset -= epsilon

        new_elevs = limit_elev + offsets

        dh = new_elevs - elevs

        # if any nodes change by less than epsilon, just zero it; nothing we can do
        dh[numpy.absolute(dh) <= -offset] = 0.0

        assert numpy.all(dh >= 0.0), 'new %s, old %s offset %s' % (new_elevs, elevs, offsets)

        return numpy.sum(dh * areas)

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

    def _each_neighbour_of(self, nid, neighbours):
        neigh = neighbours[nid]
        return numpy.nditer(neigh[neigh >= 0])

    def _distribute_sediment_land(self, sink_id, elev, deposition_volume, areas, neighbours, sea):
        """
        For a given node is 'sink_id' and its deposition rate 'rate', determine
        how to distribute the sediment volume within the catchment.

        Modifies 'deposition_change' in-place.

        elev is the elevation of each node including any erosion.

        Discards any excess sediment that does not fit within the catchment.

        Returns (overflow_node_id, overflow_volume). If there is too much 
        deposition for this sill, this function can return a node id and volume
        where the excess should be deposited.
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
        # Once we know which nodes we will raise, we figure out by how much we
        # raise them. If the capacity of the nodes is sufficient for the amount
        # of deposition, we raise them exactly the amount required to hold the
        # deposition.
        #
        # NOT IMPLEMENTED:
        # If we run out of nodes to raise and there is still sediment
        # remaining, we 'top off' the catchment so as to eliminate the
        # depression/sink in it. We then FOR THIS VERSION OF THE ALGORITHM ONLY
        # discard any excess sediment.

        # nodes which have been raised to perform this deposition
        raised_ids = set()
        visited_ids = set([sink_id])
        # TODO: unvisited_ids = prioqueue of nodes ordered from lowest to highest
        unvisited_ids = set([sink_id])
        dv = deposition_volume[sink_id]

        # Create a new sinks array which is local to this deposition action. We deform the TIN as we go, so it's not safe to reuse across depositions.
        sinks = numpy.empty(elev.shape, dtype=int)
        sinks.fill(-1)

        exploration_maxh = elev[sink_id]
        sill_maxh = 10000000.0  # FIXME: use a better constant for infinity
        sill_id = None  # we use this in the overfill case

        volume = 0.0  # volume of sediment that we have distributed

        overfill_id = None
        overfill_volume = None

        while len(unvisited_ids) and volume < dv:
            # TODO: prioqueues would help here
            # pop lowest id in queue
            lowest_id = self._pop_lowest_node(elev=elev, unvisited_ids=unvisited_ids)
            lowest_elev = elev[lowest_id]

            '''
            we iterate over nodes from lowest to highest
            when we visit a node, we check all of its neighbours
                if a neighbour is part of the catchment, we add it to the prioqueue to examine later
                if it is NOT part of the catchment, it is a sill
                    we determine our lowest sill elev based on its elev

                our minraise is now the elev of this node - we can raise to AT MOST this high based on what we know about
                our maxraise is the LOWEST sill elevation

                if we can accomodate all sediment using the minraise constraint, we stop searching
                if we run out of nodes to search, the lowest sill determines the raise height
            '''

            # TODO: I'm waiving this constraint temporarily, but there's something odd going on
            # Sometimes node receivers go uphill (the elevation increases from donor->recvr) which I don't think should happen
            # if not(lowest_elev >= exploration_maxh):
                # import pdb; pdb.set_trace()
            # assert lowest_elev >= exploration_maxh, 'le %s, em %s' % (lowest_elev, exploration_maxh)
            # exploration_maxh = lowest_elev
            exploration_maxh = max(lowest_elev, exploration_maxh)

            # add any potential neighbours to our 'unvisited' list
            # this guarantees that the next-lowest node in the catchment is
            # available to be visited
            for nid in self._each_neighbour_of(lowest_id, neighbours):
                inid = int(nid)
                if inid not in visited_ids:
                    visited_ids.add(inid)
                    # We're changing the flow network as we go, so the previous sink calculations are probably not reliable
                    if self._resolve_sink(sinks, inid, elev, sea) == sink_id:
                        unvisited_ids.add(inid)
                    else:
                        # it's a sill node; does it restrict our maxh?
                        if elev[inid] < sill_maxh:
                            sill_maxh = elev[inid]
                            sill_id = inid

            maxh = min(sill_maxh, exploration_maxh)

            # What is the new capacity of the catchment based on our known constraints?
            # NOTE: this scales O(N^2) with node count. There are definitely
            # more efficient algorithms!
            volume = self._volume_of_nodes(areas, raised_ids, maxh, elev)

            if volume < dv:  # we will need to find more nodes
                raised_ids.add(lowest_id)
            # if not, this node is our limit

        # work out how to fill the nodes

        # pull the raised ids into a list so we can sort them descending by elevation
        raised_ids_array = numpy.array(list(raised_ids))

        if dv > volume:
            # We're going to overfill the catchment, so set up a slope away from the drain point.
            # We use the straight line distance from the sill to the current node to determine the new height of the node.
            # This isn't appropriate for all models.

            # TODO: this could be vectorised
            for nid in raised_ids:
                # only used for assertion later
                ri_areas = areas[raised_ids_array]

                # what's the distance from nid to sill_id?
                dist = numpy.sqrt(numpy.sum(numpy.power(self.xycoords[nid] - self.xycoords[sill_id], 2)))
                nid_raise = dist * 0.000005  # guess factor - we want about 0.001 raise per node
                assert nid_raise >= 0.0
                elev[nid] += nid_raise

            # There is a slight chance that there is not enough sediment to fill the new structure. This will be detected and flagging at the end of this function.
            # FIXME: above is not true!
        else:
            # We fill all of the nodes to the same elevation, but subtract a small offset based on their current elevation order. This ensures that the drainage network stays intact and we do not introduce any new depressions.

            ri_elevs = elev[raised_ids_array]
            ri_areas = areas[raised_ids_array]

            # we sort the nodes so we can keep the flow network the same
            descending_indices = numpy.argsort(ri_elevs)[::-1]

            # calculate the offset from max that we assign each node (ri_offset)
            epsilon = 0.001
            ri_offset = numpy.zeros_like(ri_elevs)
            offset = 0.0
            for index in numpy.nditer(descending_indices):
                ri_offset[index] = offset
                offset -= epsilon

            # raise the selected nodes by a constant factor
            # the equations for this are:
            # new_elevation = raise_factor * current_elevation - offset
            # fill_amount = new_elevation - current_elevation
            # where fill_amount is equal to dv
            # we seek raise_factor
            # TODO: there's some working to show here
            newh = (dv - numpy.sum(ri_areas * ri_offset) + numpy.sum(ri_areas * ri_elevs)) / numpy.sum(ri_areas)

            if newh > (maxh - epsilon):
                print 'WARNING: overfill; newh was %s max %s' % (newh, maxh)
                newh = maxh - epsilon

                # we're going to overfill to another catchment
                # we know the sill node
                # what is the sill node's sink? that's where the surplus will end up

                # assert sill_id is not None
                if sill_id is None:
                    print 'WARNING: discarded overfill of volume %s' % '???'
                else:
                    overfill_sink_id = self._resolve_sink(sinks, sill_id, elev, sea)
                    assert overfill_sink_id is not None
                    print 'overflow to dest %s sill id %s' % (overfill_sink_id, sill_id)
                    print "OVERFILL - doesn't look fully implemented"

            ri_new_elev = newh + ri_offset

            # determine how the depositions change as a result
            # TODO: check that allocated volume matches what you expect
            dchange = ri_new_elev - ri_elevs

            dchange[numpy.absolute(dchange) <= -offset] = 0.0  # don't bother with tiny changes

            if numpy.any(dchange < 0.0):
                print 'WARNING: adjusted dchange'
                dchange[dchange < 0.0] = 0.0

            assert elev[raised_ids_array].shape == dchange.shape

            elev[raised_ids_array] += dchange

            assert numpy.all(dchange >= 0.0), 'new %s, old %s, offset %s, newh %s' % (ri_new_elev, ri_elevs, ri_offset, newh)

            # DEBUG ONLY: check that we allocated exactly what we expect
            assert dchange.shape == ri_areas.shape
            allocated_volume = numpy.sum(dchange * ri_areas)
            # FIXME: there are still differences between the catchment volume calculation and the final assignment
            depo_error = allocated_volume - dv

            if abs(depo_error) > 0.01:
                print 'WARNING: land discard allocated %s, total %s' % (allocated_volume, dv)

        return overfill_id, overfill_volume

    def _lowest_neighbour(self, nid, neighbours, elev):
        # taking into account new deposits, what is the lowest neighbour node?
        # we can't use the existing flow network as our deposits have changed that
        # 'nid' could be the lowest

        neigh = neighbours[nid]
        neigh = neigh[neigh >= 0]
        neigh = numpy.append(neigh, nid)
        nelevs = elev[neigh]
        min_index = nelevs.argmin()
        return neigh[min_index], nelevs[min_index]

    def _distribute_sediment_sea(self, elev, deposition_volume, areas, neighbours, sea):
        '''
        Resolves ALL of the deposits under sea. We treat them as an isolated
        system as there's no way for sediment to 'build up' and become land.

        For under-sea deposition, we fill the sink node up to the sea level
        (less epsilon). If there's too much, we send any excess to the
        neighbour with the steepest slope.

        deposition_volume (inout) reflects the initial or 'requested'
            deposition state where some nodes will have too much sediment.
        elev (inout) is updated after resolution.

        Returns False if the land sediment algorithm needs to be called again
        (because sediment has risen above sea level and cannot be resolved
        here)
        '''

        all_resolved = True  # if there are undersea overfills, set this to False

        def raise_node(nid, max_elev, max_volume):
            """
            Raises a node as high as possible given an elevation and volume
            constraint. Updates the elev array and returns the volume of
            sediment consumed. Does nothing if the new elevation is below the
            existing elevation.
            """
            a = areas[nid]

            # TODO this should take dv as a parameter and only fill to at most dv
            old_elev = elev[nid]

            # how much volume will we consume?
            trial_elev = old_elev + max_volume / a
            # print '\t\ti want trial %s' % trial_elev

            # what's the new elevation given our constraints?
            new_elev = min(trial_elev, max_elev)

            # print '\t\traise id %s from %s to max %s, volume %s; solution %s' % (nid, old_elev, max_elev, max_volume, new_elev)

            if new_elev <= old_elev:
                return 0.0

            elev[nid] = new_elev
            return (new_elev - old_elev) * a

        '''
        for each deposition point
            do a breadth-first search and deposit as much sediment as possible as quickly as possible
                breadth-first on downhill nodes
            on each iteration of the search, only go to 95% of the previous iteration's elevation
            if the dh is less than 5%, don't bother depositing, but traverse over it (this might get slow)
            find your notes to see what to do about raising nodes above sea level; perhaps just the shore node gets raised to sea level? 

            on each iteration, find ALL of the dest nodes. if they can accomodate the flow, distribute evenly among them; otherwise continue to the next level of traversal
            '''

        unresolved_ids = numpy.where((deposition_volume > 0.0) & (elev < sea))[0]
        for unresolved_id_array in numpy.nditer(unresolved_ids, flags=('zerosize_ok',)):
            unresolved_id = int(unresolved_id_array)

            unresolved_elev = elev[unresolved_id]
            # Nodes can be pushed above sea level before we get to resolve them. The land depo algorithm must deal with them.
            if unresolved_elev >= sea:
                continue

            max_distance = 0.0
            max_distance_id = unresolved_id

            # dv tracks how much sediment remains to be deposited
            dv = deposition_volume[unresolved_id]

            # We assume that these (unresolved) points are where the land->sea
            # transition occurs. We can raise them above sea level to model
            # coastline migration.
            # Diffusion might try to push isolated points above the sea level,
            # but if there's no next-highest neighbour, we stay under sea.

            # What's the next highest neighbour?
            nids = self._get_pythonic_neighbours(unresolved_id, neighbours)
            nelevs = elev[nids]
            # discard elevs which are below the current node
            nelevs = nelevs[nelevs > unresolved_elev]

            if len(nelevs) > 0:
                # there's a higher neighbour, so we can raise this node (at
                # least to sea level)
                max_elev = max(numpy.min(nelevs) - 0.0001, sea)
                volume = raise_node(nid=unresolved_id, max_elev=max_elev, max_volume=dv)
                dv -= volume

            # to prevent loops, we don't visit the same node twice
            seen_ids = set([unresolved_id])
            donor_ids = [unresolved_id]

            # Each iteration of this loop represents one more node traversal
            # from the initial deposition point.
            #
            # Sometimes we get trivially small (1e-10) volume of sediment that
            # is difficult to deposit; the threshold reduces the chance that
            # we will waste time searching for a destination
            while dv > 0.1:
                # on this iteration, this is the list of ids that we can deposit to
                receiver_ids = set()

                # maximum elev for this iteration is the worse-case elev of all donor nodes
                # TODO: slightly more optimistic would be, for each receiver node, to take the minimum of the connected donor nodes, not all nodes in the iteration
                iteration_elev = numpy.min(elev[list(donor_ids)])
                iteration_elev = min(sea - 0.0001, iteration_elev)  # receivers can't come above sea level until catchment is full
                # print '\titer up to %s' % iteration_elev

                # find all of the potential receivers
                # we're depositing to anything underwater; this is simpler and works out roughly the same
                # NOT we can deposit to a node which is downslope
                # TODO: we should probably only go downhill
                for did in donor_ids:
                    potential_ids = self._get_pythonic_neighbours(did, neighbours)
                    potential_elevs = elev[potential_ids]
                    # downslope_ids = potential_ids[potential_elevs < iteration_elev]
                    # NOTE: we're just diffusing to anything that is undersea without regard for the slope underneath
                    downslope_ids = potential_ids[potential_elevs < sea]

                    # store any nodes that we haven't already seen
                    for nid in downslope_ids:
                        if nid not in seen_ids:
                            receiver_ids.add(nid)
                            seen_ids.add(nid)

                            # We track the maximum distance from deposition point to receiver. If the catchment fills, we use this to determine the land slope.
                            dist = numpy.sqrt(numpy.sum(numpy.power(self.xycoords[nid] - self.xycoords[unresolved_id], 2)))
                            max_distance = max(max_distance, dist)
                            max_distance_id = nid

                if len(receiver_ids) == 0:
                    # The undersea part of this catchment is full. We fill the
                    # nodes to above sea level and then leave the excess for
                    # the land depo algorithm to worry about.

                    # Technically, this can push catchment nodes above their
                    # neighbours - but this is unlikely to matter as the slope
                    # is so slight

                    # We create a slope between the deposition point and the most distant point
                    dh = elev[unresolved_id] - sea
                    assert dh >= 0.0, 'dh = %s' % dh
                    if len(seen_ids) > 1:
                        assert max_distance > 0.0, 'max_distance = %s' % max_distance

                        for nid in seen_ids:
                            dist = numpy.sqrt(numpy.sum(numpy.power(self.xycoords[nid] - self.xycoords[unresolved_id], 2)))
                            frac = dist / max_distance

                            assert dist <= max_distance

                            new_elev = sea + (1 - frac) * dh

                            volume = raise_node(nid=nid, max_elev=new_elev, max_volume=dv)
                            dv -= volume
                            if dv <= 0.0:
                                break
                    else:
                        assert elev[unresolved_id] >= sea

                    # We've just filled the catchment with a slope from
                    # unresolved_id to the furthest node. The furthest node is
                    # the sink and therefore where we deposit.
                    print 'NOTE: pushed above sea level (dv = %s)' % dv
                    all_resolved = False
                    assert elev[unresolved_id] >= sea
                    assert max_distance_id is not None
                    assert elev[int(max_distance_id)] >= sea
                    deposition_volume[unresolved_id] = 0.0
                    deposition_volume[max_distance_id] = dv

                    dv = 0.0
                    break
                # else, assign sediment to the receivers

                # TODO: we should be figuring out the exact amount to raise each node to correctly deposit the right amount of sediment
                # we'd prefer to deposit evenly among the available receivers
                # for now, we're just going to iterate through and assign as much as we can to each node and worry about the uneven distribution later
                for nid in receiver_ids:
                    e = elev[nid]
                    new_elev = e + 0.95 * (iteration_elev - e)  # remember, raise_node won't lower a node
                    volume = raise_node(nid=nid, max_elev=new_elev, max_volume=dv)

                    dv -= volume
                    if dv <= 0.0:
                        break

                # reset for next iteration
                donor_ids = receiver_ids

            if dv <= 0.1:
                deposition_volume[unresolved_id] = 0.0

        return all_resolved

    def _single_catchment_fill(self, pre_elev, xymin, xymax, max_dt, sea, areas, diff_flux, neighbours, fillH, FVmesh, globalIDs):
        '''
        Deposition algorithm as described at ...

        Sets the timestep such that a single catchment is filled on each
        timestep. This is faster than the existing algorithm which fills a
        little bit up to the water level but requires many more timesteps to
        progress the simulation to the same point.

        PARAMETERS
        max_dt: maximum timestep in A, determined by previous steps in the model
        pre_elev: numpy array giving starting elevation of each TIN node, not modified
        sea: scalar sea level

        RETURN VALUES
        elev_change: numpy array giving change in elevation for each node
        dt: length of the timestep in A. This must not be changed by subsequent
            steps in the model.
        '''

        # dt is known for this whole function and cannot change
        dt = max_dt
        starting_elev = pre_elev.copy()
        elev = pre_elev.copy()  # we will modify elev throughout this function

        # rate of change on each node
        change = numpy.empty_like(elev)
        change.fill(-1.e6)

        erosion_rate = numpy.zeros_like(elev)
        deposition_volume_rate = numpy.zeros_like(elev)

        # For each node, we track the sink node (endpoint of any water runoff).
        # -1 means 'unknown', 0+ are node ids
        # so sinks[123] is the sink node id for node 123
        sinks = numpy.empty(elev.shape, dtype=int)
        sinks.fill(-1)

        # for each flow...
        # this loop doesn't have any interactions and so can be trivially vectorised and parallelised
        for donor, recvr in [(donor, self.receivers[donor]) for donor in range(len(elev))]:
            # do we have a sink node (bottom of the drainage network) where donor drains into itself?
            is_sink = (donor == recvr)

            dh = 0.95 * (elev[donor] - elev[recvr])
            if elev[donor] > sea and elev[recvr] < sea:
                dh = elev[donor] - sea

            if dh < 0.001:
                dh = 0.0  # FLOWalgo.f90:347

            # 1. CALCULATE EROSION/DEPOSITION ON EACH NODE
            # TODO: flowalgo:352: we don't calculate erosion at all on nodes that are within the water height (i.e. might be deposited on)
            rate = diff_flux[donor]  # rate of deposition (negative for erosion)
            if not is_sink and dh > 0.0 and elev[donor] >= sea:
                dist = math.sqrt((self.xycoords[donor, 0] - self.xycoords[recvr, 0]) ** 2.0 + (self.xycoords[donor, 1] - self.xycoords[recvr, 1]) ** 2.0)
                # Node erosion rate (SPL): Braun 2013 eqn (2), measured in HEIGHT per year
                rate -= self.erodibility[donor] * self.discharge[donor] ** self.m * (dh / dist) ** self.n

            if rate <= 0.0:
                erosion_rate[donor] = rate  # we erode material from the donor...  (HEIGHT per year)

            # We will deposit the same amount, but we need to work out where to deposit it.
            # Find the sink node (bottom point) for the receiver in question?
            sink_id = self._resolve_sink(sinks, donor, elev, sea)
            deposition_volume_rate[sink_id] += abs(rate) * areas[donor]  # VOLUME per year

        # No erosion or deposition on the borders
        # border_flags = (self.xycoords[:, 0] < xymin[0]) | (self.xycoords[:, 1] < xymin[1]) | (self.xycoords[:, 0] > xymax[0]) | (self.xycoords[:, 1] > xymax[1])
        # erosion_rate[border_flags] = 0.0
        # deposition_volume_rate[border_flags] = 0.0

        pre_deposition_elev = elev
        elev = pre_deposition_elev + erosion_rate * dt
        # Note that pre_deposition_elev is before erosion, while elev is after. This is useful later for nodes that might transition from land->sea.

        # 2. APPLY LAND DEPOSITION
        all_resolved = False
        first_pass = True

        while not all_resolved:
            # The only land nodes with positive deposition should be sink nodes.
            # Note that we use the pre-erosion elevation as there might be enough deposition to keep a node above sea level.
            deposition_volume = deposition_volume_rate * dt

            # TODO: try to remove the following fork, or justify its existence
            if first_pass:
                land_sinks = list(numpy.where((deposition_volume > 0.0) & (pre_deposition_elev >= sea))[0])
            else:
                land_sinks = list(numpy.where((deposition_volume > 0.0) & (elev >= sea))[0])

            # Track which sink_ids have been completely filled. This is used to prevent infinite loops where overfill bounces between two nodes.
            filled_sinks = set()

            print 'resolving %s land nodes' % len(land_sinks)
            for sink_id in land_sinks:
                if sink_id in filled_sinks:
                    print 'WARNING: overfill loop; discarding volume %s' % deposition_volume[sink_id]
                    # TODO: you should try finding the next sill node and trying to pass the deposition down the hill
                    deposition_volume[sink_id] = 0.0
                    continue

                # We use the *after* erosion figure to calculate capacity
                overfill_id, overfill_volume = self._distribute_sediment_land(sink_id, elev=elev, deposition_volume=deposition_volume, areas=areas, neighbours=neighbours, sea=sea)

                if overfill_id is not None:
                    # the sink must have filled, so add it to the filled list
                    filled_sinks.add(sink_id)

                    if elev[overfill_id] >= sea:
                        # put at at the end of the land depo list - we deal with it in this loop
                        # undersea deposition is dealt with in the next step
                        land_sinks.append(overfill_id)
                        deposition_volume[overfill_id] += overfill_volume

            # 3. APPLY SEA DEPOSITION
            print 'resolving %s sea nodes' % len(deposition_volume[deposition_volume > 0.0])
            all_resolved = self._distribute_sediment_sea(elev=elev, deposition_volume=deposition_volume, areas=areas, neighbours=neighbours, sea=sea)

            if not all_resolved:
                # rebuild receiver network
                # import ipdb; ipdb.set_trace()
                self.SFD_receivers(fillH, elev, neighbours, FVmesh.vor_edges, FVmesh.edge_length, globalIDs, sea)

                # invalidate cached sinks; they are probably wrong now
                sinks.fill(-1)

                # The land depo algorithm expects deposition to be at the sink
                # node. Push any land depositions down to the sink.
                ids = numpy.where((deposition_volume > 0.0) & (elev >= sea))[0]
                for nid_array in numpy.nditer(ids, flags=('zerosize_ok',)):
                    nid = int(nid_array)
                    sink_id = self._resolve_sink(sinks, nid, elev, sea)
                    vol = deposition_volume[nid]
                    deposition_volume[sink_id] += vol
                    deposition_volume[nid] -= vol

                first_pass = False

        elev_change = elev - starting_elev

        # No erosion or deposition on the borders
        # elev_change[border_flags] = 0.0

        return elev_change, dt

    # NOTE: we're using mindt, but that doesn't really make sense given we're passing back the absolute different in elevations. We should be adhering to mindt.
