##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling application.    ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates functions related to Badlands surface elevation.
"""

import time
import numpy
from pyBadlands.libUtils import PDalgo
import warnings

from scipy.interpolate import interpn
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

def _boundary_elevation(elevation, neighbours, edge_length, boundPts, btype):
    """
    This function defines the elevation of the TIN surface edges for 2 different types of conditions:
        1. Infinitely flat condition,
        2. Continuous slope condition.

    Parameters
    ----------
    variable : elevation
        Numpy arrays containing the internal nodes elevation.

    variable : neighbours
        Numpy integer-type array containing for each nodes its neigbhours IDs.

    variable : edge_length
        Numpy float-type array containing the lengths to each neighbour.

    variable : boundPts
        Number of nodes on the edges of the TIN surface.

    variable : btype
        Integer defining the type of boundary: 0 for flat and 1 for slope condition.

    Return
    ----------
    variable: elevation
        Numpy array containing the updated elevations on the edges.
    """

    # Flat
    if btype == 0:
        missedPts = []
        for id in range(boundPts):
            ngbhs = neighbours[id,:]
            ids = numpy.where(ngbhs >= boundPts)[0]
            if len(ids) == 1:
                elevation[id] = elevation[ngbhs[ids]]
            elif len(ids) > 1:
                lselect = edge_length[id,ids]
                picked = numpy.argmin(lselect)
                elevation[id] = elevation[ngbhs[ids[picked]]]
            else:
                missedPts = numpy.append(missedPts,id)

        if len(missedPts) > 0 :
            for p in range(len(missedPts)):
                id = int(missedPts[p])
                ngbhs = neighbours[id,:]
                ids = numpy.where((elevation[ngbhs] < 9.e6) & (ngbhs >= 0))[0]
                if len(ids) == 0:
                    raise ValueError('Error while getting boundary elevation for point ''%d''.' % id)
                lselect = edge_length[id,ids]
                picked = numpy.argmin(lselect)
                elevation[id] = elevation[ngbhs[ids[picked]]]

    # Slope
    elif btype == 1:
        missedPts = []
        for id in range(boundPts):
            ngbhs = neighbours[id,:]
            ids = numpy.where(ngbhs >= boundPts)[0]
            if len(ids) == 1:
                # Pick closest non-boundary vertice
                ln1 = edge_length[id,ids[0]]
                id1 = ngbhs[ids[0]]
                # Pick closest non-boundary vertice to first picked
                ngbhs2 = neighbours[id1,:]
                ids2 = numpy.where(ngbhs2 >= boundPts)[0]
                lselect = edge_length[id1,ids2]
                if len(lselect) > 0:
                    picked = numpy.argmin(lselect)
                    id2 = ngbhs2[ids2[picked]]
                    ln2 = lselect[picked]
                    elevation[id] = (elevation[id1]-elevation[id2])*(ln2+ln1)/ln2 + elevation[id2]
                else:
                    missedPts = numpy.append(missedPts,id)
            elif len(ids) > 1:
                # Pick closest non-boundary vertice
                lselect = edge_length[id,ids]
                picked = numpy.argmin(lselect)
                id1 = ngbhs[ids[picked]]
                ln1 = lselect[picked]
                # Pick closest non-boundary vertice to first picked
                ngbhs2 = neighbours[id1,:]
                ids2 = numpy.where(ngbhs2 >= boundPts)[0]
                lselect2 = edge_length[id1,ids2]
                if len(lselect2) > 0:
                    picked2 = numpy.argmin(lselect2)
                    id2 = ngbhs2[ids2[picked2]]
                    ln2 = lselect2[picked2]
                    elevation[id] = (elevation[id1]-elevation[id2])*(ln2+ln1)/ln2 + elevation[id2]
                else:
                    missedPts = numpy.append(missedPts,id)
            else:
                missedPts = numpy.append(missedPts,id)

        if len(missedPts) > 0 :
            for p in range(0,len(missedPts)):
                id = int(missedPts[p])
                ngbhs = neighbours[id,:]
                ids = numpy.where((elevation[ngbhs] < 9.e6) & (ngbhs >= 0))[0]
                if len(ids) == 0:
                    raise ValueError('Error while getting boundary elevation for point ''%d''.' % id)
                lselect = edge_length[id,ids]
                picked = numpy.argmin(lselect)
                elevation[id] = elevation[ngbhs[ids[picked]]]

    return elevation

def update_border_elevation(elev, neighbours, edge_length, boundPts, btype='flat'):
    """
    This function computes the domain boundary elevation for 3 different types of conditions:
        1. Infinitely flat condition,
        2. Continuous slope condition,
        3. Wall boundary (closed domain).

    Parameters
    ----------
    variable : elev
        Numpy arrays containing the internal nodes elevation.

    variable : neighbours
        Numpy integer-type array containing for each nodes its neigbhours IDs.

    variable : edge_length
        Numpy float-type array containing the lengths to each neighbour.

    variable : boundPts
        Number of nodes on the edges of the TIN surface.

    variable : btype
        Integer defining the type of boundary. Possible conditions are:
            1. wall
            2. flat: this is the default condition
            3. slope

    Return
    ----------
    variable: newelev
        Numpy array containing the updated elevations on the edges.
    """

    newelev = elev

    if btype == 'wall':
        newelev[:boundPts] = 1.e7

    elif btype == 'flat' or btype == 'slope':
        newelev[:boundPts] = 1.e7
        thetype = 0
        if btype == 'slope':
            thetype = 1

        newelev = _boundary_elevation(elev, neighbours, edge_length, boundPts, thetype)
    else:
        raise ValueError('Unknown boundary type ''%s''.' % btype)

    return newelev

def getElevation(rX, rY, rZ, coords, interp='linear'):
    """
    This function interpolates elevation from a regular grid to a cloud of points using SciPy interpolation.

    Parameters
    ----------
    variable : rX, rY, rZ
        Numpy arrays containing the X, Y & Z coordinates from the regular grid.

    variable : coords
        Numpy float-type array containing X, Y coordinates for the TIN nodes.

    variable : interp
        Define the interpolation technique as in SciPy interpn function. The default is 'linear'

    Return
    ----------
    variable: elev
        Numpy array containing the updated elevations for the local domain.
    """

    # Set new elevation to 0
    elev = numpy.zeros(len(coords[:,0]))

    # Get the TIN points elevation values using the regular grid dataset
    elev = interpn( (rX, rY), rZ, (coords[:,:2]), method=interp)

    return elev

def assign_parameter_pit(neighbours, boundPts, fillTH=1., epsilon=0.01):
    """
    This function defines global variables used in the pit filling algorithm.

    Parameters
    ----------
    variable : neighbours
        Numpy integer-type array containing for each nodes its neigbhours IDs.

    variable : boundPts
        Number of nodes on the edges of the TIN surface.

    variable : fillTH
        Limit the filling algorithm to a specific height to prevent complete filling of depression.
        Default is set to 1.0 metre.

    variable : epsilon
        Force a minimal slope to form the depression instead of a flat area to build continuous flow
        pathways. Default is set to 0.01 metres.
    """

    PDalgo.pdstack.pitparams(neighbours, fillTH, epsilon, boundPts)


def pit_stack_PD(elev, sea):
    """
    This function calls a depression-less algorithm from Planchon & Darboux to compute the flow
    pathway using stack.

    Parameters
    ----------
    variable : elev
        Numpy arrays containing the nodes elevation.

    variable : sea
        Current elevation of sea level.

    Return
    ----------
    variable: fillH
        Numpy array containing the filled elevations.
    """

    # Call stack based pit filling function from libUtils
    fillH = PDalgo.pdstack.pitfilling(elev, sea)

    return fillH

def pit_priority_flood(elev, sea, neighbours, boundPts, fillTH=1., epsilon=0.00001):
    '''
    Priority-flood+epsilon (Algorithm 3) from Barnes et al 2014

    elev: Numpy arrays containing the nodes elevation
    sea: sea level

    neighbours: map of node id->list of connected nodes?
    boundPts: number of nodes that are on the edge of the mesh; these are the first 0:boundPts nodes in 'elev'
    fillTH: ??? fill thickness?
    epsilon: minimum vertical step between nodes

    Returns pit-filled equivalent of elev

    Note that many of the parameters are set above in assign_parameter_pit, which is somewhat awkward. YOu can probably remove it as you're passing them in directly to here from buildFLux.py

    TODO: make sure you test this on multiple iterations in case the parameters are being changed as the run progresses

    None of the inputs are modified.
    '''

    pit_priority_flood.push_count = 0
    dem = numpy.copy(elev)

    def push(prio_queue, node_index, height):
        ''' push node onto priority queue '''

        # node_index is index into elev array
        item = (height, pit_priority_flood.push_count, node_index)
        heapq.heappush(prio_queue, item)
        pit_priority_flood.push_count += 1

    def peek(prio_queue):
        ''' return index of highest priority (lowest elevation) node '''
        if len(prio_queue):
            return prio_queue[0][2]
        else:
            return None

    def pop(prio_queue):
        ''' pop highest priority (lowest elevation) node '''
        return heapq.heappop(prio_queue)[2]


    # we're going to use heapq to manipulate opn; it's a prioqueue of (height, push_count (tiebreaker), node index (which is irrelevant to priority))
    # 'top' of queue is LOWEST priority, so pop() removes the LOWEST elevation item
    opn = []

    # this is just a list of indices
    # we're treating the left of the queue as 'bottom' and the right as 'top'
    pit = deque()

    closed = numpy.zeros_like(dem, dtype=numpy.bool_)

    # 5. for all c on the edges of DEM do
    for index in range(boundPts):
        z = dem[index]
        push(opn, index, z)
        closed[index] = True

    # 8: while either Open or Pit is not empty do
    while len(opn) or len(pit):
        if len(pit) and peek(opn) == pit[0]:
            c = pop(opn)
            pit_top = None
        elif len(pit):
            c = pit.pop()
            if pit_top is None:
                pit_top = dem[c]
        else:
            c = pop(opn)
            pit_top = None

        # 19: for all neighbours n of c do
        neighbour_index = 0
        while neighbours[c][neighbour_index] >= 0:
            n = neighbours[c][neighbour_index]

            if closed[n]:
                neighbour_index += 1
                continue

            closed[n] = True

            # NOTE: skip lines 22 and 23: elev is always defined in this model
            # HOWEVER, we don't want to fill things that are underwater. We use the NODATA case to handle underwater nodes; we pretend that they don't exist.
            # NEW 22
            # print 'test %s %s' % (dem[n], sea)
            #if dem[n] <= sea:
            #    pit.appendleft(n)
            #    print 'AL'
            #elif dem[n] <= dem[c] + epsilon:  # line 24
            if dem[n] <= dem[c] + epsilon:  # line 24
                if pit_top < dem[n] and dem[c] + epsilon >= dem[n]:  # TODO: second clause is always true!
                    # If you get this, probably your epsilon is too high; nodes elevations have been reordered.
                    print "A significant alteration of the DEM has occurred. The inside of the pit is now higher than the terrain surrounding it."
                dem[n] = dem[c] + epsilon
                pit.appendleft(n)
            else:
                push(opn, n, elev[n])

            neighbour_index += 1

    return dem

def pit_filling_PD(elev, neighbours, boundPts, sea, fillTH=1., epsilon=0.01):
    """
    This function calls a depression-less algorithm from Planchon & Darboux to compute the flow pathway.

    Parameters
    ----------
    variable : elev
        Numpy arrays containing the nodes elevation.

    variable : neighbours
        Numpy integer-type array containing for each nodes its neigbhours IDs.

    variable : boundPts
        Number of nodes on the edges of the TIN surface.

    variable : sea
        Current elevation of sea level.

    variable : fillTH
        Limit the filling algorithm to a specific height to prevent complete filling of depression.
        Default is set to 1.0 metre.

    variable : epsilon
        Force a minimal slope to form the depression instead of a flat area to build continuous flow
        pathways. Default is set to 0.01 metres.

    Return
    ----------
    variable: fillH
        Numpy array containing the filled elevations.
    """

    # Call pit filling function from libUtils
    fillH = PDalgo.pdcompute.filling(elev, neighbours, fillTH, epsilon, boundPts, sea )

    return fillH
