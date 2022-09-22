import itertools as itr
import math
import copy
from collections import deque

import numpy as np
import shapely.geometry




'''

Number Constants

'''

int_types = [int,np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64]
float_types = [float,np.float_,np.float16,np.float32,np.float64]
non_complex_types = [*int_types, *float_types]
complex_types = [complex,np.complex_,np.complex64,np.complex128]
num_types = [*int_types,*float_types,*complex_types]


''' Round numbers '''
r1 = lambda n: round(n, 1)
r2 = lambda n: round(n, 2)
r3 = lambda n: round(n, 3)
r4 = lambda n: round(n, 4)
r5 = lambda n: round(n, 5)
r6 = lambda n: round(n, 6)
r7 = lambda n: round(n, 7)
r8 = lambda n: round(n, 8)
r9 = lambda n: round(n, 9)
r10 = lambda n: round(n, 10)


'''

Array (Numpy Arrays) Operations

    Verification Log:
    - "is_Nx2_array" verified on 2/8/2022 (see macro "is_Nx2_array_2_8_2022)
    - "vstack_Nx2_arrays" verified on 2/8/2022 (see macro "vstack_Nx2_arrays_2_8_2022")
    - "is_N_array" verified along with "is_Nx2_array" on 3/8/2022 (see macro "is_Nx1_is_Nx2_tests_3_8_2022")
    - "N_array_subsets" and "Nx2_array_subsets" verified on 3/8/2022 (see macro "N_array_Nx2_array_subsets_test_3_8_2022")
    - "is_2_array" verified on 3/11/2022 (see macro "is_2_array_tests_3_11_2022")

'''


''' if at least equal to one item, return True else False'''
def is_any(obj, *items):
    for item in items:
        if obj==item: return True
    return False


''' will check if input is a 2 array (numpy shape (2,)); transposing has no effect '''
def is_2_array(p):
    if isinstance(p, np.ndarray): # checks if its array
        if len(p.shape) == 1: # checks if it has one dimension
            if len(p) == 2: return True # checks if it is size 2
    return False


''' will check if input is a 2 array (numpy shape (2,)) of only real non-complex types; transposing has no effect '''
def is_2_real_array(p):
    if is_2_array(p):
        if isinstance(p[0], tuple(non_complex_types)) and \
            isinstance(p[1], tuple(non_complex_types)):
            return True
    return False


''' will check if input is a 3 array (numpy shape (3,)); transposing has no effect '''
def is_3_array(p):
    if isinstance(p, np.ndarray): # checks if its array
        if len(p.shape) == 1: # checks if it has one dimension
            if len(p) == 3: return True # checks if it is size 2
    return False


''' will check if input is a 3 array (numpy shape (3,)) of only real non-complex types; transposing has no effect '''
def is_3_real_array(p):
    if is_3_array(p):
        if isinstance(p[0], tuple(non_complex_types)) and \
            isinstance(p[1], tuple(non_complex_types)) and \
            isinstance(p[2], tuple(non_complex_types)):
            return True
    return False


''' will check if input is a N array (numpy shape (N,)); transposing has no effect '''
def is_N_array(v):
    if isinstance(v, np.ndarray): # checks if its array
        if len(v.shape) == 1: return True  # checks if it has one dimension
    return False


''' will check if input is a N x 2 array (numpy shape (N,2))'''
def is_Nx2_array(v):
    if isinstance(v, np.ndarray): # checks if its array
        if len(v.shape) == 2: # checks if it has two dimensions
            if v.shape[1] == 2: return True # checks if second dimension is size 2
    return False


''' will check if input is a N x 3 array (numpy shape (N,3))'''
def is_Nx3_array(v):
    if isinstance(v, np.ndarray): # checks if its array
        if len(v.shape) == 2: # checks if it has two dimensions
            if v.shape[1] == 3: return True # checks if second dimension is size 3
    return False


''' will append a third column with ones; useful to multiply with 3x3 matrices '''
def Nx2_to_Nx3(v):
    assert is_Nx2_array(v)

    col3 = np.array([np.linspace(1, 1, len(v))]).T

    return np.hstack((v, col3))


''' will discard third column; useful for projection of points on xy-plane or if working with 2x2 matrices '''
def Nx3_to_Nx2(v):
    assert is_Nx3_array(v)

    return v[:,0:2]


''' will stack variable Nx2 arrays together'''
def vstack_Nx2_arrays(*v, chk=True):
    if chk:
        for p in v: 
            if not is_Nx2_array(p): raise TypeError('all v must by variable N x 2 arrays')

    v_stk = v[0]
    for p in v[1:]: v_stk = np.vstack((v_stk, p))

    return v_stk

''' will stack variable N arrays together '''
def hstack_N_arrays(*v, chk=True):
    if chk:
        for p in v:
            if not is_N_array(p): raise TypeError('all v must be variable N arrays.')

    h_stk = v[0]
    for p in v[1:]: h_stk = np.hstack((h_stk, p))

    return h_stk


''' will generate subsets of v (an N array) based on "slices" (an Nx2 array) where each element [a,b]
    is a slice interval to produce subset "v[a:b]"; option to stack all subsets and option to prohibit 
    overlapping slice intervals (not implemented yet). '''
def N_array_subsets(v, slices, stack=False, overlap=True, chk=False):
    if chk:
        if not is_N_array(v): raise TypeError('v must be an N array.')
        if not is_Nx2_array(slices): raise TypeError('slices must be an Nx2 array.')
    
    if overlap==False: pass # implement later... (throw error if any slices overlap)

    subsets = []
    for slice_ in slices: subsets.append(v[slice_[0]:slice_[1]])

    if stack: return hstack_N_arrays(*subsets, chk=False)
    else: return subsets


''' will generate subsets of v (an Nx2 array) along the first dimension (length N) based on "slices" 
    (an Nx2 array) where each element [a,b] is a slice interval to produce subset "v[a:b]"; option 
    to stack all subsets and option to prohibit overlapping slice intervals (not implemented yet). '''
def Nx2_array_subsets(v, slices, stack=False, overlap=True, chk=False):
    if chk:
        if not is_Nx2_array(v): raise TypeError('v must be an Nx2 array.')
        if not is_Nx2_array(slices): raise TypeError('slices must be an Nx2 array.')
    
    if overlap==False: pass # implement later... (throw error if any slices overlap)

    subsets = []
    for slice_ in slices: subsets.append(v[slice_[0]:slice_[1]])

    if stack: return vstack_Nx2_arrays(*subsets, chk=False)
    else: return subsets


''' will sort v (Nx2 array) based on column (0 or 1) '''
def Nx2_sort(v, col, chk=True):
    if chk:
        if not is_Nx2_array(v): raise TypeError('v must be an Nx2 array.')
        if not (col==0 or col==1): raise ValueError('col can only be 0 or 1.')

    return v[v[:, col].argsort()]


''' will sort v (Nx2 list) based on column (0 or 1) '''
def Nx2_list_sort(v, col, chk=True):
    if chk:
        if not isinstance(v, list): raise TypeError('v must be an Nx2 list.') # chk incomplete
    if col==0: return sorted(v, key=lambda x: (x[0],x[1]))
    elif col==1: return sorted(v, key=lambda x: (x[1],x[0]))
    else: raise ValueError('col can only be 0 or 1.')


''' will swap columns of v (Nx2) array '''
def swap_x_y(v, chk=True):
    if chk:
        if not is_Nx2_array(v): raise TypeError('v must be an Nx2 array.')

    return np.vstack((v[:,1], v[:,0])).T


''' let v be a list or numpy array '''
''' will get slice [a, b] of v if a is greater than b such
    that it concatenates a-to-end of v and beginning-to-b of v '''
''' useful for constructs that have a cyclic nature (i.e. a closed path) '''
def cyclic_slice(v, a, b):
    # chk for a and b?
    if isinstance(v, list):
        if a<=b: return v[a:b]
        else: return [*v[a:], *v[:b]]
    elif isinstance(v, np.ndarray):
        if a<=b: return v[a:b]
        else: return np.array([*v[a:], *v[:b]])
    else: raise TypeError('v must be a list or numpy array.')


''' let v be a list or numpy array, and shift be an integer '''
''' will shift the elements of v in reference to a new start index
    (specified relative to the previous '''
''' useful for constructs that have a cyclic nature (i.e. a closed path) '''
def cyclic_shift(v, shift):
    print(shift)
    
    # needs chks
    if isinstance(v, np.ndarray):
        v = deque(v)
        v.rotate(shift)
        return np.array(v)
    elif isinstance(v, list):
        v = deque(v)
        v.rotate(shift)
        return list(v)
    else: raise TypeError('v must be a list or numpy array.')


''' radians to degrees '''
def rad2deg(rad):
    return rad*180/np.pi


''' degrees to radians '''
def deg2rad(deg):
    return deg*np.pi/180


'''let x and y be N arrays. Will convert to list of N x 2 array of 2D points;
   first column is x and second column is y.'''
def to_points(x, y):
    return np.vstack((x,y)).T


''' v must be Nx2 array; will convert into N-1x4 array where the ith row is     
    the concatenation of the ith and (i+1)-th rows of v 
    (concatenation of successive points, making an edge) '''
def points_to_edges(v):
    assert is_Nx2_array(v)
    return np.hstack((v[:len(v)-1], v[1:]))


def end_ind(arr):
    return len(arr)-1


def inds(arr):
    return range(len(arr))


def tri_center(x1,y1,x2,y2,x3,y3): 
    return [(x1+x2+x3)/3,(y1+y2+y3)/3]


# let v1 and v2 be N=2 or Nx2 arrays
def distance(v1, v2): 
    err_msg = 'v1 and v2 must be N=2 or Nx2 arrays.'
    if is_2_array(v1):
        if is_2_array(v2): return np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)
        elif is_Nx2_array(v2): return np.sqrt((v2[:,0]-v1[0])**2 + (v2[:,1]-v1[1])**2)
        else: raise TypeError(err_msg)
    elif is_Nx2_array(v1):
        if is_2_array(v2): return np.sqrt((v1[:,0]-v2[0])**2 + (v1[:,1]-v2[1])**2)
        elif is_Nx2_array(v2): 
            if len(v1)!=len(v2): raise ValueError('If v1 and v2 are Nx2 arrays, they must have same length.')
            return np.sqrt((v2[:,0]-v1[:,0])**2 + (v2[:,1]-v1[:,1])**2)
        else: raise TypeError(err_msg)
    else: raise TypeError(err_msg)


# let v be N=2 or Nx2 arrays
def magnitude(v):
    if is_2_array(v): return np.sqrt(v[0]**2 + v[1]**2)
    elif is_Nx2_array(v): return np.sqrt(v[:,0]**2 + v[:,1]**2)
    else: raise TypeError('v must be N=2 or Nx2 arrays.')


# let v be a Nx2 array (N=M), N>=2
# output is a Nx2 array (N=M-1) of difference vectors between adjacent points (segments)
# 'normalize' is optional to normalize these segments
# 'add' option will take addition instead of difference of adjacent points
def segments(v,normalize=False,add=False, chk=False):
    if chk:
        if not is_Nx2_array(v): raise TypeError('v must be a Nx2 array.')
        if len(v)<2: raise ValueError('v must have two or more points; N>=2')

    if normalize:
        if add: v = v[1:] + v[:len(v)-1]
        else: v = v[1:] - v[:len(v)-1]
        d = distance(v)
        v[:,0] = v[:,0]/d
        v[:,1] = v[:,1]/d
        return v
    else: 
        if add: return v[1:] + v[:len(v)-1]
        else: return v[1:] - v[:len(v)-1]

'''

2D Affine matrices

'''

# 2 x 2, about origin, ang is rads
rot_2x2 = lambda ang: np.array([
    [np.cos(ang), -np.sin(ang)],
    [np.sin(ang), np.cos(ang)]
]).astype('float64')

rotT_2x2 = lambda ang: np.array([
    [np.cos(ang), np.sin(ang)],
    [-np.sin(ang), np.cos(ang)]
]).astype('float64')
# 3 x 3, about origin, ang is rads
rot_3x3 = lambda ang: np.array([
    [np.cos(ang), -np.sin(ang), 0],
    [np.sin(ang), np.cos(ang), 0],
    [0, 0, 1]
]).astype('float64')
rotT_3x3 = lambda ang: np.array([
    [np.cos(ang), np.sin(ang), 0],
    [-np.sin(ang), np.cos(ang), 0],
    [0, 0, 1]
]).astype('float64')
# 3 x 3, about p, ang is rads
rotp_3x3 = lambda ang, px, py: np.array([
    [np.cos(ang), -np.sin(ang), -px*np.cos(ang)+py*np.sin(ang)+px],
    [np.sin(ang), np.cos(ang), -px*np.sin(ang)-py*np.cos(ang)+py],
    [0, 0, 1]
]).astype('float64')
rotpT_3x3 = lambda ang, px, py: np.array([
    [np.cos(ang), np.sin(ang), 0],
    [-np.sin(ang), np.cos(ang), 0],
    [-px*np.cos(ang)+py*np.sin(ang)+px, -px*np.sin(ang)-py*np.cos(ang)+py, 1]
]).astype('float64')
# 3 x 3, translate ds 
trans_3x3 = lambda dx, dy: np.array([
    [1, 0, dx],
    [0, 1, dy],
    [0, 0, 1]
]).astype('float64')
transT_3x3 = lambda dx, dy: np.array([
    [1, 0, 0],
    [0, 1, 0],
    [dx, dy, 1]
]).astype('float64')
# 2 x 2 (then 3 x 3), reflection about vector l passing through origin (transpose is same)
refl_2x2 = lambda lx, ly: np.array([
    [lx**2-ly**2, 2*lx*ly],
    [2*lx*ly, ly**2-lx**2]
]).astype('float64')/np.sqrt(lx**2 + ly**2)
refl_3x3 = lambda lx, ly: np.array([
    [lx**2-ly**2, 2*lx*ly, 0],
    [2*lx*ly, ly**2-lx**2, 0],
    [0, 0, 1]
]).astype('float64')/np.sqrt(lx**2 + ly**2)
# 2 x 2 (then 3 x 3), shear parallel to x then parallel to y (transpose of shear_x is shear_y & vice versa)
shearX_2x2 = lambda k: np.array([
    [1, k],
    [0, 1]
]).astype('float64')
shearY_2x2 = lambda k: np.array([
    [1, 0],
    [k, 1]
]).astype('float64')
shearX_3x3 = lambda k: np.array([
    [1, k, 0],
    [0, 1, 0],
    [0, 0, 1]
]).astype('float64')
shearY_3x3 = lambda k: np.array([
    [1, 0, 0],
    [k, 1, 0],
    [0, 0, 1]
]).astype('float64')
# 2 x 2 (then 3 x 3), scale x and y (transpose is same)
scale_2x2 = lambda sx, sy: np.array([
    [sx, 0],
    [0, sy]
]).astype('float64')
scale_3x3 = lambda sx, sy: np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
]).astype('float64')



# 3 x 3 to scale about a point
scale_3x3_p = lambda sx, sy, px, py: matmuls(
    trans_3x3(px, py),
    scale_3x3(sx, sy),
    trans_3x3(-px, -py)
)


''' Will perform recursive matrix multiplication using numpy matmul '''
def matmuls(*M):
    M = list(M)
    M.reverse()

    matmuls = M[0]
    for Mi in M[1:]: matmuls = np.matmul(Mi, matmuls)

    return matmuls



# let v be N x 2 array; will return N x 3 with third column of zeros
as_3D = lambda v: np.hstack((v, np.zeros((len(v),1))))

# let v be N x 3 array; will return N x 2 of first two columns
as_2D = lambda v: v[:, 0:2]

# let v be an N x 2 numpy 2D array,
# output is an N-2 x 2 numpy 2D array consisting of unit tangent vector
# of all intermediate points; essentially f sum of
# adjacent segment vectors 
# If ends is true, output will be N x 2 numpy 2D array with the unit tangents
# of the endpoints
def tangents(v,ends=False):
    if ends:
        segs = segments(v,norm=True)
        tangs = segments(segs,norm=True,add=True)
        tang0 = 2*segs[0]*np.dot(tangs[0],segs[0]) - tangs[0]
        tang0 = tang0/distance(0,0,*tang0)
        tangn = 2*np.dot(segs[len(segs)-1],tangs[len(tangs)-1])*segs[len(segs)-1]-tangs[len(tangs)-1]
        return np.vstack((tang0,tangs,tangn))
    else: return segments(segments(v,norm=True,add=True),norm=True)

def elevation_angle(x1,y1,x2,y2): 
    return np.pi/2 if abs(x2-x1)==0 else np.arctan((y2-y1)/(x2-x1))

def size_of_legs(x1,y1,x2,y2): 
    return [abs(x2-x1),abs(y2-y1)]

def mid(i,f): 
    return (i+f)/2

def midpoint(x1,y1,x2,y2): 
    return [(x1+x2)/2,(y1+y2)/2]

# let v be a numpy 2D array
def centroid(v):
    length = v.shape[0]
    return np.array([np.sum(v[:,0])/length, np.sum(v[:,1])/length])

# will find the x and y midpoints.
def center(v):
    x, y = v[:, 0], v[:, 1]
    return np.array([(np.amin(x)+np.amax(x))/2, (np.amin(y)+np.amax(y))/2])


# input in radians
def np_rotate(X,ang): 
    return np.matmul(np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]),X)


# v is N x 2 array, rotate v about point p in ang
def rotate(v, p=[0, 0], ang=0, degrees=False):
    assert is_2_array(v) or is_Nx2_array(v)

    v = v.astype('float64')
    if degrees==True: ang = ang*np.pi/180

    [px, py] = p
    vx = v[:,0]
    vy = v[:,1]

    return np.vstack((
        px + np.cos(ang) * (vx - px) - np.sin(ang) * (vy - py),
        py + np.sin(ang) * (vx - px) + np.cos(ang) * (vy - py)
    )).T

# v is N x 2 array, p and bp are 1 x 2 arrays of place point
# and basepoint values. By default basepoint is center.
def place(v, p, bp=None):
    if bp!=None: v = v - bp + p
    else: v = v - centroid(v) + p
    return v

# v is N x 2 array, ds is 1 x 2 array of shift values
def shift(v, ds):
    assert is_2_array(v) or is_Nx2_array(v)
    v = v.astype('float64')

    return v + ds

# v is N x 2 array, p and s are 1 x 2 arrays of scale center
# and scale values
def scale(v, p, s):
    assert is_2_array(v) or is_Nx2_array(v)
    v = v.astype('float64')

    return p + (v - p)*s

# v is numpy Nx2 array
def affine(v, ds, s, ang, degrees=False):
    v = v.astype('float64')

    v = shift(v, ds)
    c = centroid(v)
    return rotate(scale(v, c, s), c, ang, degrees=degrees)

# v can be N=2 or Nx2 array
# will return ccw perpendiculars of corresponding dimension
def perp(v):
    if is_2_array(v): return np.array([-v[1], v[0]])
    elif is_Nx2_array(v): return np.vstack((-1*v[:,1], v[:,0])).T 
    else: raise TypeError('v can only be N=2 or Nx2 array.')


# let v be a numpy N x 2 array
# flip about horizontal reference line. By default,
# reference line is at y coordinate of center.
def flip_vertical(v, ry=None):
    if ry==None:  [_,ry] = centroid(v)
    v[:,1] =  (v[:,1] - ry)*-1 + ry
    return v

# let v be a numpy Nx2 array
# flip about vertical reference line. By default,
# reference line is at x coordinate of center.
def flip_horizontal(v, rx=None):
    if rx==None:  [rx,_] = centroid(v)
    v[:,0] =  (v[:,0] - rx)*-1 + rx
    return v

# v can be N=2 or Nx2 array
# will return unit vector(s) (N=2 for single, Nx2 for plural)
def unit(v):
    if is_2_array(v): return v/np.sqrt(v[0]**2 + v[1]**2)
    elif is_Nx2_array(v):
        ds = distance(v)
        return np.vstack((v[:,0]/ds, v[:,1]/ds)).T
    else: raise TypeError('v can only be N=2 or Nx2 array.')


# let v be a numpy 2D array
def normalize_points(v):
    v = v.astype('float64')
    c = centroid(v)   # find centroid
    v -= c   # shift points so centroid is zero
    s = max(distance(0,0,v))  # scale down points based on furthest point and return
    return scale(v, [0,0], [1/s,1/s])

# normalize reference array and scale,shift other arrays based on reference's scale,shift value
def normalize_points_ref(v_ref,*vi):
    v_ref = v_ref.astype('float64')
    c = centroid(v_ref)   # find centroid
    
    v_ref -= c   # shift points so centroid is zero
    for v in vi: 
        v = v.astype('float64')    
        v -= c

    s = max(distance(v_ref))  # scale down points based on furthest point of reference and return
    vo = [scale(v_ref, [0,0], [1/s,1/s])]
    for v in vi: vo.append(scale(v, [0,0], [1/s,1/s]))
    return vo


# let v be a numpy 2D array, calculate the angle of points from center
def angles(v,append=None,normalize=None):
    v = v.astype('float64')
    if normalize:
        v = normalize_points(v)
        if append:
            a = np.zeros(shape=(len(v),3))
            a[:,0:2] = v
            a[:,2] = np.arctan2(v[:,1],v[:,0])
            return a
        else:
            return np.arctan2(v[:,1],v[:,0])      
    else:
        c = centroid(v)   # find centroid
        v -= c   # shift points to so centroid is zero
        if append:
            a = np.zeros(shape=(len(v),3))
            a[:,0:2] = v
            a[:,2] = np.arctan2(v[:,1],v[:,0])
            return a
        else:
            return np.arctan2(v[:,1],v[:,0])


# let v be a N x 2 array
# will return two N x 1 arrays of x-coords and y-coords respectively
def get_x_y(v):
    return [v[:,0], v[:,1]] 


# let all v be a N x 2 array (variable length)
# returns their corresponding N x 1 array of x-coords as a generator
def get_x(*v):
    return (v_[:,0] for v_ in v)
    
# let all v be a N x 2 array (variable length)
# returns their corresponding N x 1 array of y-coords as a generator
def get_y(*v):
    return (v_[:,1] for v_ in v)


# let v be a N x 2 array
# returns the minimum x-coordinate
def min_x(v):
    return min(v[:,0])

# let v be a N x 2 array
# returns the minimum x-coordinate
def min_y(v):
    return min(v[:,1])

# let v be a N x 2 array
# returns the minimum x-coordinate
def max_x(v):
    return max(v[:,0])

# let v be a N x 2 array
# returns the minimum x-coordinate
def max_y(v):
    return max(v[:,1])

# let v be a N x 2 array
# returns the point with first instance of minimum x-coordinate
def Nx2_min_x(v):
    pts = v[v[:, 0]==min(v[:, 0]), :]
    if len(pts)>1: return pts[0]
    else: return pts

# let v be a N x 2 array
# returns the point with  first instance of minimum y-coordinate
def Nx2_min_y(v):
    pts = v[v[:, 1]==min(v[:, 1]), :]
    if len(pts)>1: return pts[0]
    else: return pts

# let v be a N x 2 array
# returns the point with  first instance of maximum x-coordinate
def Nx2_max_x(v):
    pts = v[v[:, 0]==max(v[:, 0]), :]
    if len(pts)>1: return pts[0]
    else: return pts

# let v be a N x 2 array
# returns the point with  first instance of maximum y-coordinate
def Nx2_max_y(v):
    pts = v[v[:, 0]==min(v[:, 0]), :]
    if len(pts)>1: return pts[0]
    else: return pts

# lets arrs be 1D size-N arrays (all same size)
# will output a 1D size-N array with elements being the minimum of 
# all corresponding elements between the arrays
def elements_minima(*arrs):
    # check size consistency later
    arr_mins = arrs[0]
    for arr in arrs[1:]: arr_mins = np.minimum(arr_mins, arr)

    return arr_mins


# lets arrs be 1D size-N arrays (all same size)
# will output a 1D size-N array with elements being the maximum of 
# all corresponding elements between the arrays
def elements_maxima(*arrs):
    # check size consistency later
    arr_maxs = arrs[0]
    for arr in arrs[1:]: arr_maxs = np.maximum(arr_maxs, arr)

    return arr_maxs


# let v be a N x 2 array
# returns the distance between largest and smallest x-value from points
def x_spread(v):
    x=v[:,0]
    return max(x) - min(x)

# let v be a N x 2 array
# returns the distance between largest and smallest y-value from points
def y_spread(v):
    y=v[:,1]
    return max(y) - min(y)

# let v be a N x 2 array
# will return a 2 x 1 list of the x and y spreads
def x_y_spreads(v):
    x, y = v[:,0], v[:,1]
    return [max(x)-min(x), max(y)-min(y)]


# let v be a N x 2 array
# returns the midpoint between largest and smallest x-value from points
def x_mid(v):
    x=v[:,0]
    return (min(x) + max(x))/2

# let v be a N x 2 array
# returns the midpoint between largest and smallest y-value from points
def y_mid(v):
    y=v[:,1]
    return (min(y) + max(y))/2


# let v be a numpy 2D array, sort 2D array based on its angle from center
def angle_sort(v,normalize=None):
    if normalize:
        v = angles(v,append=True, normalize=True)
        return v[np.argsort(v[:, 2])]
    else:
        v = angles(v,append=True)
        return v[np.argsort(v[:, 2])]


# Linearly transform coordinates based on starting and end window with reflection about x-axis when normalized)
# let v be 2D numpy array 
def lin_trans_reflect_x(v,fx1,fx2,fy1,fy2,sx1,sx2,sy1,sy2):
    return (2*(((2*v - [sx1+sx2,sy1+sy2])/[sx2-sx1,sy1-sy2])*[fx2-fx1,fy2-fy1]/2)+[fx1+fx2,fy1+fy2])/2

def limits_to_pairs(x1,x2,y1,y2):
    return [[x1,y1],[x2,y2]]

def pairs_to_limits(x1,y1,x2,y2):
    return [[x1,x2],[y1,y2]]

def all_between(i1, i2, *i, i1_bound=True, i2_bound=True, chk=False):
    err_msg = "All inputs must be real numbers."
    if chk:
        for p in i:
            if not isinstance(p, tuple(non_complex_types)): raise TypeError(err_msg)
        if not isinstance(i1, tuple(non_complex_types)): raise TypeError(err_msg)
        if not isinstance(i2, tuple(non_complex_types)): raise TypeError(err_msg)
        if i2 <= i1: raise ValueError("i2 must be greater than i1.")
        if not (isinstance(i1_bound, bool) and isinstance(i2_bound, bool)): 
            raise TypeError("'i1_bound' and 'i2_bound' must be bools only.")

    if i1_bound==True: 
        if i2_bound==True: 
            for p in i: # [i1, i2]
                if not (p >= i1 and p <= i2): return False
            return True
            
        else: 
            for p in i: # [i1, i2)
                if not (p >= i1 and p < i2): return False
            return True
    else:
        if i2_bound==True: 
            for p in i: # (i1, i2]
                if not (p > i1 and p <= i2): return False
            return True
            
        else: 
            for p in i: # (i1, i2)
                if not (p > i1 and p < i2): return False
            return True


# p must be numpy 1 x 2 arrays, will implement later...
def pairs_between(x1, y1, x2, y2, *p, x1_bound=True, y1_bound=True, x2_bound=True, y2_bound=True, chk=False):
    pass


# will return array with True everywhere r is within a, b (False elsewhere)
''' needs thorough check for all inputs '''
def arraysBetween(r, a, b, startClose=True, endClose=True):
    if startClose: lb = np.greater_equal(r, a)
    else: lb = np.greater(r, a)
    if endClose: ub = np.less_equal(r, b)
    else: ub = np.less(r, b)  

    return np.logical_and(lb, ub)      


# will return array with True everywhere r is "almost" within a, b (False elsewhere)
''' needs thorough check for all inputs '''
def arraysBetweenAlmost(r, a, b, startClose=True, endClose=True, rtol=0, atol=1e-9):
    if startClose: lb = np.logical_or(np.greater(r, a), np.isclose(r, a, rtol=rtol, atol=atol))
    else: lb = np.logical_and(np.greater(r, a), np.logical_not(np.isclose(r, a, rtol=rtol, atol=atol))) 
    if endClose: ub = np.logical_or(np.less(r, b), np.isclose(r, b, rtol=0, atol=atol))
    else: ub = np.logical_and(np.less(r, b), np.logical_not(np.isclose(r, b, rtol=rtol, atol=atol))) 

    return np.logical_and(lb, ub) 


# finds scale over specified duration based on scale over reference duration
def scale_per(r, tr, ts):
    return np.exp(ts*np.log(r)/tr)

# transform array values relative to start range to being relative to an end range
# let v be a numpy 1D array
# chk later
def rangeform(v, ri1, ri2, rf1, rf2):
    c1 = (ri1+ri2)/2
    c2 = (rf1+rf2)/2
    s2 = abs(rf2-rf1)
    s1 = abs(ri2-ri1)
    return (v - c1)*s2/s1 + c2

# like rangeform, but end range is restricted to -1 to 1
def rangebinorm(v,ri1,ri2):
    c1 = (ri1+ri2)
    s1 = abs(ri2-ri1)
    return (2*v - c1)/s1    

# like rangeform, but end range is restricted to 0 to 1
def rangenorm(v, ri1, ri2):
    c1 = (ri1+ri2)/2
    s1 = abs(ri2-ri1)
    # return round((v - c1)/s1 + 1/2, 8)
    return (v - c1)/s1 + 1/2

# like rangeform, but instead of a v, inputs a path function (function of t)
# and 'rangeforms' its input range from being relative to a start range
# to being relative to an end range.
def pathform(path_,ri1,ri2,rf1,rf2):
    def path_inside(t):
        t = rangeform(t,ri1,ri2,rf1,rf2)
        return path_(t)
    return path_inside


# i1 and i2 are the desired start and end points at the input side of the time curve
# o1 and o2 are the desired start and end points at the output side of the time curve
# t1 and t2 are the animation start and end points
def rateform(path_,i1,i2,o1,o2):
    def rate_inside(t,t1,t2):
        # step 1: go from animation time to input side of time curve
        t = rangeform(t,t1,t2,i1,i2)
        # step 2: morph time path by applying time curve
        t = path_(t)
        # step 3: go from output side of time curve back to animation time
        t = rangeform(t,o1,o2,t1,t2) 

        return t
    return rate_inside


# let v be a 1D numpy array
def nearest(v, val):
    ind = (np.abs(v - val)).argmin()
    return v[ind]

''' Following have to do with complex numbers '''


# numpy Nx2 array v to complex N array
def points_to_complex(v):
    return v[:,0] + v[:,1]*1j

# complex N array c to points Nx2 array
def complex_to_points(c):
    return np.vstack((c.real,c.imag)).T

# numpy Nx2 arrays arrs to complex N arrays c_arrs
def points_to_complex_arrs(*arrs):
    c_arrs=[]
    for v in arrs: 
        c_arrs.append(points_to_complex(v))
    return c_arrs

# complex N arrays c_arrs to points Nx2 arrays arrs
def complex_to_points_arrs(*c_arrs):
    arrs=[]
    for c in c_arrs: 
        arrs.append(complex_to_points(c))
    return arrs

# let pths be a list of N x 2 (N variable) arrays.
# output will be vstack of all arrays (path) and start indices needed to
# break path back into original pths.
def paths_to_path(*pths):
            
    inds=[0]
    ind=0
    for ln in [len(pth) for pth in pths[:len(pths)-1]]: 
        ind+=ln
        inds.append(ind)

    v=pths[0]
    for pth in pths[1:]: v=np.vstack((v,pth))

    return [v,inds]

# let pth be an N x 2 array and inds to be any indices in ascending order
# within the length of path. Output will be segments of path starting at
# the indices and ending at the next.
def path_to_paths(pth,*inds):
            
    pths = []
    for i in range(len(inds)):
        if i==len(inds)-1: pths.append(pth[inds[i]:])
        else: pths.append(pth[inds[i]:inds[i+1]])
        
    return pths




''' Following have to do with lines '''

# will find smallest possible positive integer which
# is multiples of span away from "align"
def gen_pos_align(align,span):
    if align<0: return span - (abs(align)%span)
    else: return align

# will generate a N array of values between x1 and x2,
# such that adjacent values are a "span" apart and all values
# are aligned to the nearest multiple of spans  from "align"
# for now let align only be integers...
# function needs optimization... (may have extra or time-consuming steps)
def align_span(x1,x2,align,span,chk=None):
    if chk:
        # this check needs to be tested and include checks for "align" and "span"
        try:
            if x1 > x2: raise Exception
        except: raise ValueError('"x2" must be greater than "x1.')

    rem1 = align%span
    divs = math.floor((x2-x1)/span)
    arr = np.linspace(0,divs+1,divs+2)*span + x1
    if x1 < 0:
        if x2 < 0:
            rem2 =  abs(arr[0])%span
            arr = arr + rem2 + rem1 - span
        else: arr = arr + rem1 - nearest(arr,rem1)
    else: 
        rem2 =  abs(arr[0])%span
        arr = arr + rem1 - rem2

    arr=arr[arr>=x1]   
    arr=arr[arr<=x2]
    return arr    

# will generate an N array with n elements such that adjacent elements 
# a 'span' amount apart, and whole array is centered around zero.
def extend_span(span,n):
    return np.linspace(0,(n-1)*span,n) -  (n-1)*span/2


# let v be a numpy Nx2 array
def points_to_lines_gen(v,chk=None):
    if chk:
        try:
            if v.ndim != 2 or len(v)<2: raise Exception
        except: raise TypeError('Input must be a 2D numpy array and have more than 2 points.')

    for i in range(len(v)): yield [*v[len(v)-1], *v[0]] if i==len(v)-1 else [*v[i], *v[i+1]]


# let v be a numpy Nx2 array
# then this function returns two numpy (N-1)x2 arrays, first being the start control points,
# second being the end control points both normalized for cubic bezier and then 
# being scaled such that zero is no handles and 1 is their handles touch. Both handles
# of a curve segment are of the same size.
def c_bezier_tangents(v,ctrl_scale=0.5):
    tangs = tangents(v,ends=True)
    starts = tangs[:len(tangs)-1]
    ends = tangs[1:]*-1
    scales = ctrl_scale*distance(segments(v))/2
    starts = np.vstack((starts[:,0]*scales,starts[:,1]*scales)).T
    ends = np.vstack((ends[:,0]*scales,ends[:,1]*scales)).T
    # scales = ctrl_scale*distances(segments(v))/(2*cosines)
    # starts = np.vstack((starts[:,0]*scales,starts[:,1]*scales)).T
    # ends = np.vstack((ends[:,0]*scales,ends[:,1]*scales)).T
    return [starts,ends] # start and end control points normalized respectively

# let v be a numpy Nx2 array
def points_to_lines_vec(v,chk=None):
    if chk:
        try:
            if v.ndim != 2 or len(v)<2: raise Exception
        except: raise TypeError('Input must be a 2D numpy array and have more than 2 points.')

    return np.hstack((v[:len(v)-1],v[1:]))

# input flattened xx, yy mesh_grid arrays of x and y, as well as the slopes
# evaluated from flattened array, and the center deviation for lines
# All inputs must be an numpy array.
def slope_to_lines(xx,yy,slope_angs,dev):
    xc = np.cos(slope_angs)
    yc = np.sin(slope_angs)
    return np.vstack((
        xx - dev*xc,
        yy - dev*yc,
        xx + dev*xc,
        yy + dev*yc
    )).T


# Return true if pair of line segments v1-v2 and v3-v4 intersect
def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4,chk=None):
    def ccw(x1,y1,x2,y2,x3,y3):
        return (y3-y1)*(x2-x1) > (y2-y1)*(x3-x1)
    
    return ccw(x2,y2,x3,y3,x4,y4) != ccw(x1,y1,x3,y3,x4,y4) and ccw(x1,y1,x2,y2,x4,y4) != ccw(x1,y1,x2,y2,x3,y3) 




# Return true if any pair from n line segments intersect
# This function needs optimization (currently generates all combinations and then iteraters through...
# need it to interpret combination during the iteration)


# let lns be a 2D 4 x N numpy array
def N_segments_intersect(segs, chk=None):
    if chk:
        try: 
            if segs.ndim != 2: raise Exception
            if len(segs[0]) != 4 or len(segs)<2: raise Exception
        except: raise TypeError('Input must be an N x 4 numpy array with N >= 2 (min. 2 lines).')

    for pr in itr.combinations(segs, 2):
        if segments_intersect(*pr[0],*pr[1]): return True
    return False


''' NEEDS UNITTEST... '''
''' will take variable shapes (as Nx2 arrays; arrs) and resize/reposition them
    such they're in constructed tiles. Count of shapes must be <= size of tiles.
'''
''' All Shapes will be positioned so that their centroid is at the center of 
    their respective tile '''
''' bx, by are bottom-left position of tiling '''
''' w is width of tiling '''
''' h is height of tiling '''
''' rows is number of rows in tiling '''
''' cols is number of columns in tiling '''
''' rw is the width of row (if specified, will override w to rw*cols) '''
''' ch is the height of columns (if specified, will override to h to ch*rows '''
def shapes_tile(*arrs, bx, by, w, h, rows, cols, rw=None, ch=None):
    # input checks later...
    if not rw is None: w = rw*cols
    if not ch is None: h = ch*rows

    if len(arrs)>rows*cols:
        raise ValueError('number of shapes (length of "arrs") must be less than size of tiling (rows*cols)')

    dw, dh = w/cols, h/rows
    cx, cy = bx+dw, by+dh
    ctr=0
    shapes = []
    for i in range(0, rows-1):
        for k in range(0, cols-1):
            if ctr==len(arrs): return shapes

            shape = copy.deepcopy(arrs[ctr])
            shape = Nx2_to_Nx3(shape)

            [prev_cx, prev_cy] = centroid(shape)
            shape = np.matmul(shape, transT_3x3(prev_cx*-1, prev_cy*-1)) # shift to origin about point
            shape = np.matmul(shape, transT_3x3(cx, cy)) # shift to new center

            shape = Nx3_to_Nx2(shape)
            shapes.append(shape)            
            ctr+=1
            cx+=dw
        cy+=dh


''' will take a shape (as Nx2 array) and convert it to an SVG path string'''
def shape_to_SVG_paths(shape): 
    # chk later 
    return ' '.join(['%s%d %d' % (['M', 'L'][i>0], x, y) for i, [x, y] in enumerate(shape)])


'''********************************************************************


Parametrics...


********************************************************************'''



# linear interpolate (credit to Freya Holmer from her YouTube video "The Beauty of Bezier Curves" for the name)
# p0, p1 are endpoints (N=2 array)
# t is interpolation position; 0 at p0 and 1 at p1 (real scalar or N array)
def lerp_func(p0, p1, t, chk=False):
    err_msg = 'p0 and p1 must be N=2 arrays.'
    if chk:
        if not is_2_array(p0): raise TypeError(err_msg)
        if not is_2_array(p1): raise TypeError(err_msg)

    if is_N_array(t):
        x = p0[0]+(p1[0]-p0[0])*t
        y = p0[1]+(p1[1]-p0[1])*t
        return np.vstack((x,y)).T
    elif isinstance(t, tuple(non_complex_types)): return p0+(p1-p0)*t
    else: raise TypeError('t can only be a real number or an N array.')


def lerp_wrapper(p0, p1):
    def lerp_inside(t):
        return lerp_func(p0, p1, t, chk=True)
    return lerp_inside




# linear interpolate between two MxN arrays (same dimensions)
# t is interpolation position; 0 at v0 and 1 at v1 (real scalar or N array)
def array_lerp_func(v0, v1, t):
    assert isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray) # must be numpy array
    shp0, shp1 = v0.shape, v1.shape
    assert len(shp0)==2 and len(shp1)==2 # must have 2 dimensions
    assert shp0[0]==shp1[0] and shp0[1]==shp1[1] # dimensions must be the same

    if is_N_array(t): return np.array([v0+(v1-v0)*ti for ti in t]) # PxMxN array where P is length of t
    elif isinstance(t, tuple(non_complex_types)): return v0+(v1-v0)*t # MxN
    else: raise TypeError('t can only be a real number or an N array.')


def array_lerp_wrapper(v0, v1):
    def array_lerp_inside(t):
        return array_lerp_func(v0, v1, t, chk=True)
    return array_lerp_inside




# linear interpolate between two scalar values s0 and s1.
# t is interpolation position; 0 at s0 and 1 at s1 (real scalar or N array)
def scalerp_func(s0, s1, t, chk=False):
    err_msg = 's0 and s1 must be real numbers.'
    if chk:
        if not isinstance(s0, tuple(non_complex_types)): raise TypeError(err_msg)
        if not isinstance(s1, tuple(non_complex_types)): raise TypeError(err_msg)

    return s0+(s1-s0)*t # needs a check for t (can be any array of real numbers within [0, 1])


def scalerp_lerp_wrapper(s0, s1):
    def scalerp_lerp_inside(t):
        return scalerp_func(s0, s1, t, chk=True)
    return scalerp_lerp_inside




# quadratic bezier
# p0 and p2 be endpoints, and p1 be control point (all N=2 arrays)
# t is interpolation position; 0 at p0 and 1 at p2 (real scalar or N array)
def q_bezier_func(p0, p1, p2, t, chk=False):
    err_msg = 'p0, p1 and p2 must be N=2 arrays.'
    if chk:
        if not is_2_array(p0): raise TypeError(err_msg)
        if not is_2_array(p1): raise TypeError(err_msg)
        if not is_2_array(p2): raise TypeError(err_msg)

    if is_N_array(t):
        x = ((1-t)**2)*p0[0] + 2*(1-t)*t*p1[0] + (t**2)*p2[0] 
        y = ((1-t)**2)*p0[1] + 2*(1-t)*t*p1[1] + (t**2)*p2[1] 
        return np.vstack((x,y)).T
    elif isinstance(t, non_complex_types): 
        return lerp_func(lerp_func(p0, p1, t), lerp_func(p1, p2, t), t)
    else: raise TypeError('t can only be a real number or an N array.')


def q_bezier_wrapper(p0, p1, p2):
    def q_bezier_inside(t):
        return q_bezier_func(p0, p1, p2, t, chk=True)
    return q_bezier_inside





# cubic bezier
# p0 and p3 be endpoints, and p1 and p2 be control points (all N=2 arrays)
# t is interpolation position; 0 at p0 and 1 at p3 (real scalar or N array)
def c_bezier_func(p0, p1, p2, p3, t, chk=False):
    err_msg = 'p0, p1, p2 and p3 must be N=2 arrays.'
    if chk:
        if not is_2_array(p0): raise TypeError(err_msg)
        if not is_2_array(p1): raise TypeError(err_msg)
        if not is_2_array(p2): raise TypeError(err_msg)
        if not is_2_array(p3): raise TypeError(err_msg)

    if is_N_array(t):
        x = ((1-t)**3)*p0[0] + 3*((1-t)**2)*t*p1[0] + 3*(1-t)*(t**2)*p2[0] + (t**3)*p3[0]
        y = ((1-t)**3)*p0[1] + 3*((1-t)**2)*t*p1[1] + 3*(1-t)*(t**2)*p2[1] + (t**3)*p3[1]
        return np.vstack((x,y)).T
    elif isinstance(t, non_complex_types): 
        return lerp_func(lerp_func(lerp_func(p0, p1, t), lerp_func(p1, p2, t), t), lerp_func(lerp_func(p1, p2, t), lerp_func(p2, p3, t), t), t)
    else: raise TypeError('t can only be a real number or an N array.')


def c_bezier_wrapper(p0, p1, p2, p3):
    def c_bezier_inside(t):
        return c_bezier_func(p0, p1, p2, p3, t, chk=True)
    return c_bezier_inside




# logistic curve with input t and parameters L (height), k (horizontal scaling), t0 (horizontal positioning)
# t can be a scalar or a N array, will return the corresponding type
def logistic_func(L, k, t0, t, chk=False):
    err_msg = 'L, k, and t0 must be real numbers.'
    if chk:
        if not isinstance(L, non_complex_types): raise TypeError(err_msg)
        if not isinstance(k, non_complex_types): raise TypeError(err_msg)
        if not isinstance(t0, non_complex_types): raise TypeError(err_msg)
        if not (isinstance(t, non_complex_types) or is_N_array(t)):
            raise TypeError('t must only be a real numbe or an N array.')

    return L/(1 + np.exp(-k*(t-t0)))


def logistic_wrapper(L, k, t0):
    def logistic_inside(t):
        return logistic_func(L, k, t0, t, chk=True)
    return logistic_inside




'''
Ways to classify arc:
1. center, radius, ang0, ang (default)
2. center, start, ang (wrapper)
3. start, end, radius (direction ccw) (wrapper)
4. start, end, ang (positive --> ccw, negative --> cw) (wrapper)


All 4 arc functions verified on 3/12/2022; see macro "arc_tests_3_12_2022"

'''

# arc interpolate.Let c be center (a N=2 array), r be radius,
# ang1 and ang2 be start and end angles, respectively, and  
# t be interpolation position between endpoints (0 at start point and 
# 1 at end point)


''' arc interpolate: center, radius, ang0, ang '''
# c is center (N=2 array)
# r is radius (positive real number)
# ang0 & ang are start and displacement angles (both real numbers)
# t is interpolation position; 0 at ang0 and 1 at ang0+ang 
# (real scalar or N array; will return N=2 or Nx2 respectively)
def arc_func(c, r, ang0, ang, t, chk=False):

    if chk:
        if not is_2_array(c): raise TypeError('c (center) must be a N=2 array.')
        if not (isinstance(r, non_complex_types) and r>0): raise TypeError('r (radius) must be a positive real number.')
        if not isinstance(ang0, non_complex_types): raise TypeError('ang0 and ang must be real numbers.')
        if not isinstance(ang, non_complex_types): raise TypeError('ang0 and ang must be real numbers.')

    if is_N_array(t):
        ang1=ang0+ang
        angt = scalerp_func(ang0, ang1, t)
        x = c[0] + r*np.cos(angt)
        y = c[1] + r*np.sin(angt)
        return np.vstack((x,y)).T # return Nx2
    elif isinstance(t, non_complex_types): 
        ang1=ang0+ang
        angt = scalerp_func(ang0, ang1, t)
        x = c[0] + r*np.cos(angt)
        y = c[1] + r*np.sin(angt)
        return np.array([x, y]) # return N=2
    else: raise TypeError('t can only be a real number or an N array.')





# c and s are N=2 arrays
# ang is a real number
# t is real numbe or N array
def arc_center_start_ang_func(c, s, ang, t, chk=False):
    if chk:
        if not (is_2_array(c) and is_2_array(s)): 
            raise TypeError('c (center) and s (start) must be N=2 arrays.')
        if not isinstance(ang, non_complex_types):
            raise TypeError('ang must be a real number.')

    [drx, dry] = s - c

    r = np.sqrt(drx**2+dry**2)
    if drx>0: # Q1, Q4, positive x axis
        ang0 = np.arctan(dry/drx) 
    elif drx<0: # Q2, Q3, negative x axis
        ang0 = np.arctan(dry/drx) + np.pi
    else: # drx==0
        if dry>0: # positive y axis
            ang0 = np.pi/2
        elif dry<0: # negative y axis
            ang0 = -np.pi/2
        else: # dry==0
            raise Exception('Value is indeterminate.')
        
    return arc_func(c, r, ang0, ang, t)



# e and s are N=2 arrays
# r is positive scalar
# dir is direction (CCWS, CCWL, CWS, & CWL)
# needs a condition for r = infinity
def arc_start_end_rad_func(s, e, r, t, dir='CCWS', chk=False):
    if chk:
        if not is_2_array(s): raise TypeError('s and e must be N=2 arrays.')
        if not is_2_array(e): raise TypeError('s and e must be N=2 arrays.')
        if not (isinstance(r, non_complex_types) and r>0):
            raise Exception('r must be a positive real number.')

    ds = e - s
    sm = ds/2
    sm_mag = magnitude(sm)

    if r < sm_mag: raise ValueError('r must be >= half the distance between start and end points.')

    ds_u = unit(ds)
    mc_u = perp(ds_u)

    theta = np.arcsin(sm_mag/r)
    mc_mag = r*np.cos(theta)

    mc = mc_mag*mc_u

    if dir=='CCWS': 
        c = s + sm + mc
        ang = 2*theta
    elif dir=='CCWL':
        c = s + sm - mc
        ang = 2*np.pi - 2*theta
    elif dir=='CWS':
        c = s + sm - mc
        ang = -2*theta
    elif dir=='CWL':
        c = s + sm + mc
        ang = 2*theta - 2*np.pi
    else: raise ValueError('valid dir values are "CCWS", "CCWL", "CWS", & "CWL".')
    
    return arc_center_start_ang_func(c, s, ang, t)



# e and s are N=2 arrays
# ang is a real number between -2pi and 2pi
def arc_start_end_ang_func(s, e, ang, t, chk=False):
    err_msg='ang must be a real number in the interval (-2pi, 2pi).'
    if chk:
        if not is_2_array(s): raise TypeError('s and e must be N=2 arrays.')
        if not is_2_array(e): raise TypeError('s and e must be N=2 arrays.')
        if not isinstance(ang, non_complex_types): raise TypeError(err_msg)

    ds = e - s

    ds_u = unit(ds)
    mc_u = perp(ds_u)

    sm = ds/2
    sm_mag = magnitude(sm)

    if ang==0: return lerp_func(s, e, t)
    else:
        if ang>-2*np.pi and ang<-np.pi: # CWL
            theta = (2*np.pi - abs(ang))/2
            r = sm_mag/np.sin(theta)
            mc_mag = r*np.cos(theta)
            mc = mc_mag*mc_u
            c = s + sm + mc
        elif ang>=-np.pi and ang<0: # CWS
            theta = abs(ang/2)
            r = sm_mag/np.sin(theta)
            mc_mag = r*np.cos(theta)
            mc = mc_mag*mc_u       
            c = s + sm - mc
            
        elif ang>0 and ang<np.pi: # CCWS
            theta = abs(ang/2)
            r = sm_mag/np.sin(theta)
            mc_mag = r*np.cos(theta)
            mc = mc_mag*mc_u       
            c = s + sm + mc      
        elif ang>=np.pi and ang<2*np.pi: # CCWL
            theta = (2*np.pi - abs(ang))/2
            r = sm_mag/np.sin(theta)
            mc_mag = r*np.cos(theta)
            mc = mc_mag*mc_u       
            c = s + sm - mc  
        else: raise TypeError(err_msg)

        return arc_center_start_ang_func(c, s, ang, t)


'''
Returns parametric function constructed by exponential Fourier Series of 
input vertices (vi, Nx2 array). Parametric function is evaluated over t (N array), 
producing output vertices (vo, Nx2 array)

N is number of frequencies used for F.S
'''
def fourier_series_wrapper(vi, N):

    # make complex:
    vi=points_to_complex(vi) # N array, N=P

    sz = len(vi)
    dt = 1/sz
    ti = np.linspace(0, 1, sz+1)[1:]  # N array, N=P
    n = np.linspace(-N, N, 2*N + 1)   # N array, N=Q

    # Getting coefficients
    M = (tile_by(vi, n).T)*np.exp(matmul_1D(ti, n)*-2*np.pi*1j)*dt  # MxN array, M=P, N=Q
    c = np.sum(M, axis=0)  # N array, N=Q  

    # let t be a N array; allow scalars?

    def fourier_series_func(t):

        # Evaluating Fourier Series over t
        M = tile_by(c, t)*np.exp(matmul_1D(t, n)*2*np.pi*1j) # MxN array, M=R, N=Q
        vo = np.sum(M, axis=1) # R x 1 (as 1D)

        return complex_to_points(vo) # back to points (Nx2 array)
    return fourier_series_func





# let v be Nx2 array, N>=2
def arc_length(v, chk=False):
    if chk:
        if not is_Nx2_array(v): raise TypeError('v must be a Nx2 array.')
        if len(v) < 2: raise ValueError('v must have two or more points; N >= 2')

    return np.sum(magnitude(segments(v)))
        
    
'''
Let S1 be an Nx2 array at t=0 and S2 be an Mx2 array t=1
Then output Si (Px2) is an interpolated array at some t in between,
where P is the rounded interpolation of N and M at t.
'''

def segmentsInterpolate(S1, S2, t, integers=True):
    # print('Printing S1')
    # print(S1)
    # print('Printing S2')
    # print(S2)

    assert is_Nx2_array(S1) and is_Nx2_array(S2)
    assert t>=0 and t<=1

    def pos_inds(column):
        # print(column)
        # print(len(column))
        return np.array([[i, i/(len(column)-1)] for i in range(len(column))])

    def initialize_Si(n1, n2, t):
        ni = round(scalerp_func(n1, n2, t))
        return np.array([np.linspace(0, 0, ni), np.linspace(0, 0, ni)]).T

    ''' boundary as in start or end '''
    def boundary_lerp_calc(pos_ind_i, pos_inds_, S):
        for [k, pos_ind_] in pos_inds_:
            k = int(k)
            if pos_ind_==pos_ind_i:
                b_lerp = S2[k, :]
                break
            elif pos_ind_>pos_ind_i:
                inda = k-1
                indb = k
                pos_ind_a = pos_inds_[inda, 1]

                b_lerp_t = scalerp_func(pos_ind_a, pos_ind_, pos_ind_i)
                b_lerp = lerp_func(S[inda, :], S[indb, :], b_lerp_t)
                break
            else: pass
        return b_lerp


    n1 = len(S1[:, 0])
    n2 = len(S2[:, 0])

    if n1==1:
        if n2==1:
            Si = array_lerp_func(S1, S2, t)
        else:
            Si = initialize_Si(n1, n2, t)

            pos_inds2 = pos_inds(S2[:, 0])
            pos_inds_i = pos_inds(Si[:, 0])

            start_lerp = S1[0] # getting only row of S1
            for i in range(len(Si[:, 0])):
                pos_ind_i = pos_inds_i[i, 1]

                end_lerp = boundary_lerp_calc(pos_ind_i, pos_inds2, S2)
                Si[i, :] = lerp_func(start_lerp, end_lerp, t)
    else:
        if n2==1:
            Si = initialize_Si(n1, n2, t)

            pos_inds1 = pos_inds(S1[:, 0])
            pos_inds_i = pos_inds(Si[:, 0])

            end_lerp = S2[0] # getting only row of S2
            for i in range(len(Si[:, 0])):
                pos_ind_i = pos_inds_i[i, 1]

                start_lerp = boundary_lerp_calc(pos_ind_i, pos_inds1, S1)
                Si[i, :] = lerp_func(start_lerp, end_lerp, t)        
        
        else:
                        
            Si = initialize_Si(n1, n2, t)

            pos_inds1 = pos_inds(S1[:, 0])
            pos_inds2 = pos_inds(S2[:, 0])
            pos_inds_i = pos_inds(Si[:, 0])

            for i in range(len(Si[:, 0])):
                pos_ind_i = pos_inds_i[i, 1]

                start_lerp = boundary_lerp_calc(pos_ind_i, pos_inds1, S1)
                end_lerp = boundary_lerp_calc(pos_ind_i, pos_inds2, S2)
                Si[i, :] = lerp_func(start_lerp, end_lerp, t)

    if integers: Si = np.rint(Si).astype(int)

    return Si



''' credit to Luke Abers '''
#Does linear interpolation of two arrays
#s1 & s2 are numpy arrays, t is a numpy array or tuple; returns a numpy array
def segmentsInterpolate2(s1, s2, t, integers=True):

    #Returns positional ratio based upon length n
    #n is an integer; returns a numpy array;
    def createPosRatio(n):
        return np.linspace(0, 1, n)

    #Returns nearest upper bound to n in arr
    #arr is a numpy array, n is a decimal number; returns a decimal number;
    def findUpperBound(arr, n):
        return arr[arr >= n].min()

    #Returns nearest lower bound to n in arr
    #arr is a numpy array, n is a decimal number; returns a decimal number;
    def findLowerBound(arr, n):
        return arr[arr <= n].max()

    #Returns the nth column of arr
    #arr is a numpy array, n is a decimal number; returns a numpy array;
    def returnColumn(arr, n):
        return arr[:,n]

    t = np.array([t])
    #Get column size for arrays
    l1 = len(s1)
    l2 = len(s2)
   
    #If lengths are equal, resort to standard array lerp
    if l1 == l2:
        Si = array_lerp_func(s1, s2, t)
    
    #If lengths are not equal, create new interpolated array with length based upon lengths of other columns and t
    else:
        #Get new column length
        P = round(lerp_func(l1, l2, t[0]))
        
        #Create positional ratios for the 2 input arrays and for the new column length
        p0 = createPosRatio(P)
        ps1 = createPosRatio(l1)
        ps2 = createPosRatio(l2)
        
        #Get columns from s1 and s2
        #s1 and s2 are assumed to have two columns
        col0s1 = returnColumn(s1, 0)
        col1s1 = returnColumn(s1, 1)
        
        col0s2 = returnColumn(s2, 0)
        col1s2 = returnColumn(s2, 1)

        #Lists to hold new column values
        startLerpCol0 = []
        startLerpCol1 = []
        
        endLerpCol0 = []
        endLerpCol1 = []
        
        for i in p0:
            #Find bounds
            lps1 = findLowerBound(ps1, i)
            ups1 = findUpperBound(ps1, i)
            
            lps2 = findLowerBound(ps2, i)
            ups2 = findUpperBound(ps2, i)
            
            #Get values from columns based on bounds
            vlCol0s1 = col0s1[np.where(ps1 == lps1)][0]
            vuCol0s1 = col0s1[np.where(ps1 == ups1)][0]
            vlCol1s1 = col1s1[np.where(ps1 == lps1)][0]
            vuCol1s1 = col1s1[np.where(ps1 == ups1)][0]
            
            vuCol0s2 = col0s2[np.where(ps2 == ups2)][0]
            vlCol0s2 = col0s2[np.where(ps2 == lps2)][0]
            vuCol1s2 = col1s2[np.where(ps2 == ups2)][0]
            vlCol1s2 = col1s2[np.where(ps2 == lps2)][0]
            
            #Add lerp to new columns
            startLerpCol0.append(lerp_func(vlCol0s1, vuCol0s1, lerp_func(lps1,ups1,i)))
            startLerpCol1.append(lerp_func(vlCol1s1, vuCol1s1, lerp_func(lps1,ups1,i)))
            
            endLerpCol0.append(lerp_func(vuCol0s2, vlCol0s2, lerp_func(lps2,ups2,i)))
            endLerpCol1.append(lerp_func(vuCol1s2, vlCol1s2, lerp_func(lps2,ups2,i)))

        #Combine lerp lists into np arrays
        startLerp = np.column_stack((startLerpCol0,startLerpCol1))
        endLerp = np.column_stack((endLerpCol0,endLerpCol1))
        
        #Return final array lerp
        Si = array_lerp_func(startLerp, endLerp, t)

    if integers: Si = np.rint(Si).astype(int)

    return Si


# let v be a numpy N x 2 array
# output is a N x 3 list (if 'close_path' is True) or N-1 x 3 (if 'close_path' is False) 
# where first element is the 'lerp' wrapper, second and third elements are start and end times.
# Output can be input directly for "crvs" in 'parametric_join'
# def points_to_lerps(v,close_path=False):
#     pth=[]
#     for i in inds(v):
#         if i==end_ind(v):
#             if close_path: 
#                 p0, p1 = v[i], v[0]
#                 pth.append([lerp_wrapper(p0, p1), 0, 1])    
#         else: 
#             p0, p1 = v[i], v[i+1]
#             pth.append([lerp_wrapper(p0, p1), 0, 1])

#     return pth



# let a be N x 1 (as 1D) 
# let b be M x 1 (as 1D)
# tile a so output dimension M x N
def tile_by(a,b):
    return np.tile(a.reshape((1,len(a))),(len(b),1))

# let a be N x 1 (as 1D) 
# let b be M x 1 (as 1D)
# output be N x M where element-ij = element-i(a) * element-j(b) 
def matmul_1D(a,b):
    return np.matmul(a.reshape((len(a),1)),b.reshape((len(b),1)).T)



# Let v be an N x 2 array. Shifts v so the point at ind is
# now the starting point. ind must be a positive integer. If ind
# is larger than length of v, then ind is remainder of the length division
def points_shift(v,ind):
    ind = ind%len(v)
    return np.vstack((v[ind:],v[:ind]))


# let v be an N x 2 array. Will shift points so that the point whose angle
# from the center is nearest to 'ang' is the starting point.
def points_shift_angle(v,ang,degrees=False):
    angs = angles(v)
    if degrees: angs = angs*180/np.pi
    return points_shift(v,abs(angs-ang).argmin())



def neighbors_or(arr,px,py,val):
    return (arr[px,py+1]==val) or   \
    (arr[px+1,py+1]==val) or        \
    (arr[px+1,py]==val) or          \
    (arr[px+1,py-1]==val) or        \
    (arr[px,py-1]==val) or          \
    (arr[px-1,py-1]==val) or        \
    (arr[px-1,py]==val) or          \
    (arr[px-1,py+1]==val)

def neighbors_and(arr,px,py,val):
    return (arr[px,py+1]==val) and   \
    (arr[px+1,py+1]==val) and        \
    (arr[px+1,py]==val) and          \
    (arr[px+1,py-1]==val) and        \
    (arr[px,py-1]==val) and          \
    (arr[px-1,py-1]==val) and        \
    (arr[px-1,py]==val) and          \
    (arr[px-1,py+1]==val)

# checks if point is a convex corner (by checking if any of the corner neighbors 
# and neighbors adjacent to that corner are 'val')
def is_convex_corner(arr,px,py,val):
    def top_left(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def top_right(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def bottom_right(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def bottom_left(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]!=val)
    return top_left(arr,px,py,val) or    \
    top_right(arr,px,py,val) or          \
    bottom_right(arr,px,py,val) or       \
    bottom_left(arr,px,py,val)


# checks if point is a concave corner (by checking if any of the corner neighbors 
# are not 'val' and remaining neighbors are)
def is_concave_corner(arr,px,py,val):
    def top_left(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]!=val)
    def top_right(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def bottom_right(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def bottom_left(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    return top_left(arr,px,py,val) or    \
    top_right(arr,px,py,val) or          \
    bottom_right(arr,px,py,val) or       \
    bottom_left(arr,px,py,val)


# checks if point is a dip (by checking if any of the mid edge neighbors 
# are not val and remaining neighbors are)
def is_dip(arr,px,py,val):
    def left(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]==val)
    def right(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def top(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def bottom(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    return left(arr,px,py,val) or    \
    right(arr,px,py,val) or          \
    top(arr,px,py,val) or       \
    bottom(arr,px,py,val)


# checks if point is a peak (by checking if any of the full or partial edge neighbors 
# are val and remaining neighbors aren't)
def is_peak(arr,px,py,val):
    def left(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def left_top(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]==val)
    def left_bottom(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]==val) and          \
        (arr[px-1,py+1]!=val)

    def right(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def right_top(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def right_bottom(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]==val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)


    def top(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]==val)
    def top_right(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]==val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def top_left(arr,px,py,val):
        return (arr[px,py+1]==val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]!=val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]==val)

    def bottom(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def bottom_right(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]==val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]!=val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)
    def bottom_left(arr,px,py,val):
        return (arr[px,py+1]!=val) and   \
        (arr[px+1,py+1]!=val) and        \
        (arr[px+1,py]!=val) and          \
        (arr[px+1,py-1]!=val) and        \
        (arr[px,py-1]==val) and          \
        (arr[px-1,py-1]==val) and        \
        (arr[px-1,py]!=val) and          \
        (arr[px-1,py+1]!=val)

    return left(arr,px,py,val) or        \
    left_top(arr,px,py,val) or           \
    left_bottom(arr,px,py,val) or        \
    right(arr,px,py,val) or              \
    right_top(arr,px,py,val) or          \
    right_bottom(arr,px,py,val) or       \
    top(arr,px,py,val) or                \
    top_right(arr,px,py,val) or          \
    top_left(arr,px,py,val) or           \
    bottom(arr,px,py,val) or             \
    bottom_right(arr,px,py,val) or       \
    bottom_left(arr,px,py,val) 


# from top neighbor, go clockwise through all, 0 to 7
def neighbors_cw(arr,px,py,pos,coord=False):
    if pos==0: return arr[px,py+1]      if coord==False else [px,py+1]
    elif pos==1: return arr[px+1,py+1]  if coord==False else [px+1,py+1]
    elif pos==2: return arr[px+1,py]    if coord==False else [px+1,py]
    elif pos==3: return arr[px+1,py-1]  if coord==False else [px+1,py-1]
    elif pos==4: return arr[px,py-1]    if coord==False else [px,py-1]
    elif pos==5: return arr[px-1,py-1]  if coord==False else [px-1,py-1]
    elif pos==6: return arr[px-1,py]    if coord==False else [px-1,py]
    elif pos==7: return arr[px-1,py+1]  if coord==False else [px-1,py+1]
    else: raise ValueError('Invalid position, must be an integer from 0 to 7.')


def neighbors_ccw(arr,px,py,pos,coord=False):
    if pos==0: return arr[px,py+1]      if coord==False else [px,py+1]
    elif pos==1: return arr[px-1,py+1]  if coord==False else [px-1,py+1]
    elif pos==2: return arr[px-1,py]    if coord==False else [px-1,py]
    elif pos==3: return arr[px-1,py-1]  if coord==False else [px-1,py-1]
    elif pos==4: return arr[px,py-1]    if coord==False else [px,py-1]
    elif pos==5: return arr[px+1,py-1]  if coord==False else [px+1,py-1]
    elif pos==6: return arr[px+1,py]    if coord==False else [px+1,py]
    elif pos==7: return arr[px+1,py+1]  if coord==False else [px+1,py+1]
    else: raise ValueError('Invalid position, must be an integer from 0 to 7.')


def get_neighbors_coords(arr,px,py,val,chk=False):
    if chk==True:
        if not neighbors_or(arr,px,py,val): raise Exception('There are no neighbors with "val".') 

    coords=[]
    for i in range(8):
        coord = neighbors_cw(arr,px,py,i,coord=True)
        if arr[coord[0],coord[1]]==val:
            coords.append(coord)
    return coords
    


def neighbors_cw_vec(arr,px,py,coord=False):
    vals=np.array([
        arr[px,py+1],
        arr[px+1,py+1],
        arr[px+1,py],
        arr[px+1,py-1],
        arr[px,py-1],
        arr[px-1,py-1],
        arr[px-1,py],
        arr[px-1,py+1]     
    ])
    if coord:
        crds=np.array([
            [px,py+1],
            [px+1,py+1],
            [px+1,py],
            [px+1,py-1],
            [px,py-1],
            [px-1,py-1],
            [px-1,py],
            [px-1,py+1]     
        ])  
        return [vals,crds]
    else: return vals



def neighbors_ccw_vec(arr,px,py,coord=False):
    vals=np.array([
        arr[px,py+1],
        arr[px-1,py+1],
        arr[px-1,py],
        arr[px-1,py-1],
        arr[px,py-1],
        arr[px+1,py-1],
        arr[px+1,py],
        arr[px+1,py+1]     
    ])
    if coord:
        crds=np.array([
            [px,py+1],
            [px-1,py+1],
            [px-1,py],
            [px-1,py-1],
            [px,py-1],
            [px+1,py-1],
            [px+1,py],
            [px+1,py+1]     
        ])  
        return [vals,crds]
    else: return vals



def neighbors_cw_gen(arr,px,py,coord=False):
    vals=np.array([
        arr[px,py+1],
        arr[px+1,py+1],
        arr[px+1,py],
        arr[px+1,py-1],
        arr[px,py-1],
        arr[px-1,py-1],
        arr[px-1,py],
        arr[px-1,py+1]     
    ])
    if coord:
        crds=np.array([
            [px,py+1],
            [px+1,py+1],
            [px+1,py],
            [px+1,py-1],
            [px,py-1],
            [px-1,py-1],
            [px-1,py],
            [px-1,py+1]       
        ])  
        for neighbor,crd in zip(vals,crds): yield [neighbor,crd]
    else:
        for neighbor in vals: yield neighbor


def neighbors_ccw_gen(arr,px,py,coord=False):
    vals=np.array([
        arr[px,py+1],
        arr[px-1,py+1],
        arr[px-1,py],
        arr[px-1,py-1],
        arr[px,py-1],
        arr[px+1,py-1],
        arr[px+1,py],
        arr[px+1,py+1]     
    ])
    if coord:
        crds=np.array([
            [px,py+1],
            [px-1,py+1],
            [px-1,py],
            [px-1,py-1],
            [px,py-1],
            [px+1,py-1],
            [px+1,py],
            [px+1,py+1]       
        ])  
        for neighbor,crd in zip(vals,crds): yield [neighbor,crd]
    else:
        for neighbor in vals: yield neighbor


# Note: NumPy interprets as list of 'columns' when indexing [x][y] or [x,y]
# checks if px,py are in array
def coord_in_arr(arr,px,py):
    for i,col in enumerate(arr):
        for j in range(len(col)):
            if px==i and py==j: return True
    return False

# needs to be vectorized for optimal performance...
def all_coords_in_arr(arr,v):
    for p in v:
        if not coord_in_arr(arr,*p): return False
    return True

# needs to be vectorized for optimal performance...
def any_coords_in_arr(arr,v):
    for p in v:
        if coord_in_arr(arr,*p): return True
    return False

# needs to be vectorized for optimal performance...
def coord_in_coords(crd,crds):
    for crd_ref in crds:
        if crd[0]==crd_ref[0] and crd[1]==crd_ref[1]: return True
    return False

def coord_in_contours(crd,conts):
    for cont in conts:
        if coord_in_coords(crd,cont): return True
    return False

# needs check 
def mesh_grid(x,y,flatten=None):
    if flatten: return np.tile(x,(1,len(y)))[0],np.tile(y,(len(x),1)).T.reshape(1,(len(x)*len(y)))[0]
    else: return np.tile(x,(len(y),1)), np.tile(y,(len(x),1)).T

# please input a 2D Numpy array
def flatten(arr,axis=0):
    [m,n] = np.shape(arr)
    if axis==0: return np.reshape(arr,m*n)
    elif axis==1: return np.reshape(np.transpose(arr),m*n)
    else: raise ValueError('"axis" must be 0 or 1.')





'''

Polygon Operations

'''


''' Checks if Point (N=2 array) is inside Polygon (Nx2 array); Algorithm 1 (uses Ray-Casting) '''
'''
    Raycasting Algorithm to find out whether a point is in a given polygon.
    Performs the even-odd-rule Algorithm to find out whether a point is in a given polygon.
    This runs in O(n) where n is the number of edges of the polygon.
'''
def point_in_polygon(px, py, polyg, chk=False):
    if chk: assert is_Nx2_array(polyg)
    # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    odd = False
    # For each edge (In this case for each point of the polygon and the previous one)
    i = 0
    j = len(polyg) - 1
    while i < len(polyg) - 1:
        i = i + 1
        # If a line from the point into infinity crosses this edge
        # One point needs to be above, one below our y coordinate
        # ...and the edge doesn't cross our Y corrdinate before our x coordinate (but between our x coordinate and infinity)
        if (((polyg[i, 1] > py) != (polyg[j, 1] > py)) and (px < (
                (polyg[j, 0] - polyg[i, 0]) * (py - polyg[i, 1]) / (polyg[j, 1] - polyg[i, 1])) +
                                                                            polyg[i, 0])):
            odd = not odd  # Invert odd
        j = i # If the number of crossings was odd, the point is in the polygon
    return odd


''' Checks if Point (N=2 array) is inside Polygon (Nx2 array); Algorithm 2 (using shapely module) '''
def point_in_polygon_2(px, py, polyg, chk=False):
    if chk: assert is_Nx2_array(polyg)

    p = shapely.geometry.Point(px, py)
    return p.within(shapely.geometry.Polygon(polyg))


''' Checks if polygon 1 is in polygon 2; Algorithm 1 '''
def polygon_in_polygon(polyg1, polyg2, chk=False):
    if chk: assert is_Nx2_array(polyg1) and is_Nx2_array(polyg2)

    ''' this is O(N^2) speed; optimize to sweep-line... '''
    for edge1 in points_to_edges(polyg1):
        for edge2 in points_to_edges(polyg2):
            if segments_intersect(*edge1, *edge2): return False
            if not point_in_polygon(*edge1[0:2], polyg2): return False
    return True


''' Checks if polygon 1 is in polygon 2; Algorithm 2 (using shapely module) '''
def polygon_in_polygon_2(polyg1, polyg2, chk=False):
    if chk: assert is_Nx2_array(polyg1) and is_Nx2_array(polyg2)

    return shapely.geometry.Polygon(polyg1).within(shapely.geometry.Polygon(polyg2))


''' Checks if polygon 1 and polygon 2 intersect (Algorithm 2 (using shapely module) '''
def polygons_intersect_2(polyg1, polyg2, chk=False):
    if chk: assert is_Nx2_array(polyg1) and is_Nx2_array(polyg2)

    return shapely.geometry.Polygon(polyg1).intersects(shapely.geometry.Polygon(polyg2))


''' will take a list of Nx2 arrays (contours; close-paths)
   and will clump sub-groupings together (in lists) if they
   can form a shape with the outer-most contour being first
   in list. Will return list of these lists. Will return error
   if any polygons intersect
'''
'''
Assumption:
    A contour is at-most only inside one contour; this won't work for multi-nested contours.
'''
def contours_to_shapes(*conts):
    # list of all contour pairs (with indices stored)
    conts_pairs = list(itr.combinations([[i, el] for i, el in enumerate(conts)], 2)) 

    shapes = {}
    nested_conts = {}
    
    ''' collecting all shapes with contours inside them '''
    for cont_pair in conts_pairs:
        [ind_1, cont_1], [ind_2, cont_2] = cont_pair
        # print(cont_pair)

        if polygon_in_polygon_2(cont_2, cont_1):
            if str(ind_1) in shapes.keys():
                shapes[str(ind_1)].append(cont_2)
            else: shapes[str(ind_1)] = [cont_1, cont_2]
            nested_conts[str(ind_2)] = cont_2
        elif polygon_in_polygon_2(cont_1, cont_2):
            if str(ind_2) in shapes.keys():
                shapes[str(ind_2)].append(cont_1)
            else: shapes[str(ind_2)] = [cont_2, cont_1]
            nested_conts[str(ind_1)] = cont_1
        elif polygons_intersect_2(cont_1, cont_2): raise Exception('No two contours must intersect.')
        else: pass

    ''' collecting remaining shapes; lone contours'''
    for i, cont in enumerate(conts):
        if not (str(i) in shapes.keys()): 
            if not (str(i) in nested_conts.keys()): shapes[str(i)] = [cont]
    
    ''' sorting shapes based on appropriate index '''
    ''' this section needs optimization... too many for loops.. '''
    shapes_list = [[int(key), value] for key, value in shapes.items()]
    shapes_list.sort(key=lambda x: x[0])
    return [ind_value[1] for ind_value in shapes_list]



'''

Shape-Morpher Interpolate Under Test

'''

''' will interpolate a polygon PI between polygons P1 & P2
    based on a shape-morphing algorithm '''
''' domain of t is [0, 1] where at t=0, Pi = P1 and at t=1, Pi = P2'''
''' P1 and P2 may only be Nx2 arrays '''
def shape2DMorphInterpolate(P1, P2, t):
    assert is_Nx2_array(P1) and is_Nx2_array(P2)
    assert t>=0 and t<=1

    return polyMorph(P1, P2, t) # for now... but you control this


# Morph two polygons
# P1, P2 are Nx2 arrays of closed polygons, t is interpolation position
# Outputs Pi at time = t
def polyMorph(P1, P2, t):
    #Close polygons i.e. add first element to last element, if not done already
    if not np.array_equal(P1[0], P1[-1]):
        P1 = np.append(P1, [P1[0]],axis=0)
    if not np.array_equal(P2[0], P2[-1]):
        P2 = np.append(P2, [P2[0]],axis=0)
        
    #Get column size
    l1 = len(P1)
    l2 = len(P2)

    #Add points to polygon with smallest column size
    if (l1 < l2):
        P1 = addPoints(P1, l2 - l1)
    elif (l2 < l1):
        P2 = addPoints(P2, l1 - l2)
    
    #Align P1 according to least squares optimization
    P1 = wind(P1, P2)
    
    #Linearly interpolate results
    return array_lerp_func(P1, P2, t)
    
# Get polygon perimeter
# ring is a closed polygon
# Outputs a decimal number equal to perimeter 
def getPolyPerimeter(ring):
    d=0
    
    for i in range(len(ring)-1):
        a = ring[i]
        b = ring[i+1]
        d += distanceBetween(a, b)
        
    d += distanceBetween(ring[-1],ring[0])
    
    return d

# Get distance between points
# a, b are coordinate pairs
# Outputs distance between two points 
def distanceBetween(a, b):
    #Had issue with python 'overthinking' floats
    x1 = float(a[0])
    x2 = float(b[0])
    y1 = float(a[1])
    y2 = float(b[1])
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2);
  
# Add points to polygon
# ring is a closed polygon, numPoints is number of points to add
# Outputs ring with numPoints added  
def addPoints(ring, numPoints):

    #Add numPoints to length of ring
    desiredLength = len(ring) + numPoints
    
    #Set step to average length 
    step = getPolyPerimeter(ring) / numPoints;

    #Increment variable
    i = 0
    #Keeps track of current location
    cursor = 0
    #Half the step size
    insertAt = step / 2
    #Convert ring to list for ease of use
    ring = ring.tolist()
    
    #Main loop
    while True:
        #Get coords
        a = ring[i]
        b = ring[(i + 1) % len(ring)]
        
        #Get distance between coords
        segment = distanceBetween(a, b)

        #Insert point
        if insertAt <= cursor + segment:
            ring.insert(i + 1, pointBetween(a, b, (insertAt - cursor) / segment))
            insertAt += step
            continue

        #Increment cursor and increment variable
        cursor += segment
        i+=1
    
        #Break loop if out of bounds
        if len(ring) > desiredLength or i > len(ring) - 1:
            break
            
    #Return ring with added points   
    return np.array(ring)

#Determine location in segment to insert point
#a, b are coordinate pairs, pct is a decimal number
#Outputs new coordinate pair
def pointBetween(a, b, pct):
    point = [a[0] + (b[0] - a[0]) * pct, a[1] + (b[1] - a[1]) * pct]
    return point    

#Rotate ring to best match vs according to least squares
#ring, vs are closed polygons with ring being Pi and vs being Pe
#Outputs a new ring (Pi)
def wind(ring, vs):

    #Variable declaration
    length = len(ring)
    m = float("inf")
    bestOffset = '-'
    s = 0
    offset = 0
    
    #Main loop
    while True:
        #Reset sum
        s = 0
        
        #Cycle through Pe
        for i in range(len(vs)):
            p = vs[i]
            d = distanceBetween(ring[(offset+i)%len(ring)],p)
            
            #Append sum with square of distance
            s += d**2
        
        #Set min
        if s < m:
            m = s
            bestOffset = offset
            
        #Increment offset
        offset+=1
        
        #Terminate loop if out of bounds
        if offset >= len(ring):
            break

    #Shift ring to best location according to min of squares
    ring = ring.tolist()
    ring = ring[bestOffset:] + ring[0:bestOffset]
    
    #Return new ring
    return np.array(ring)
