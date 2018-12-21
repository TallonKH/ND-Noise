# # -*- coding: UTF-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import operator

# NOTICE:
# This is an implementation of Perlin Noise that works for any dimensionality > 1.
# It is mostly a proof of concept, and as a result is not fully optimized.
# If you want to use Perlin Noise, I recommend using a less dynamic implementation.

def average(itr):
	return sum(itr)/len(itr)

def vecLenSquared(vec):
	return sum(comp**2 for comp in vec)

def vecLen(vec):
	return vecLenSquared(vec)**0.5

def unaryVecOp(vec, func):
	return tuple(func(comp) for comp in vec)

def normalize(vec):
	reciprocal = 1/vecLen(vec)
	return unaryVecOp(vec, lambda a:a * reciprocal)

def binaryVecOp(vec1, vec2, func):
	return tuple(func(c1,c2) for c1,c2 in zip(vec1,vec2))

def dotProduct(vec1, vec2):
	return sum(a*b for a,b in zip(vec1,vec2))

def vecSum(vec1, vec2):
	return binaryVecOp(vec1,vec2,operator.add)

def vecDif(vec1, vec2):
	return binaryVecOp(vec1,vec2,operator.sub)

def randomVector(dims):
	# generate random vectors by -
	# generating each component from a normal distribution -
	# and normalizing the result
	return normalize([np.random.normal() for i in range(dims)])


def lerp(x, y, a):
	return a * (x - y) + y

def vectorToSeed(seed, vec):
	return 0

def ease(t):
	return 6*pow(t,5) - 15*pow(t,4) + 10*pow(t,3)

##### Algorithm summary #####
# note: (N refers to the dimensionality of the noise)
# note: (hypercube refers to the N-dimensional analogue of a square)
#
# to get the value at coordiate point [P]:
#  1. find the integral coordinates of the vertices/corners [C] of the hypercube that contains P.
#     there are 2ᴺ of these vertices
#  2. get the random unit vectors [U] at each C
#  2. if any C does not have a U, generate one and save it
#  3. find the vectors [V] pointing from each C to P - DO NOT NORMALIZE
#  4. for each C, find the dot product [D] between U and V
#  5. interpolate between each D in pairs (this is explained in depth later).
#     use easing function to weird borders at hypercube edges;
class NoiseGenND:
	def __init__(self, dims=2, seed=1, tiling=None, easing=ease, seedAlgorithm=vectorToSeed):
		self.dims = dims
		self.seed = seed
		self.tiling = tiling
		self.easing = easing
		self.seedAlgorithm = seedAlgorithm

		self.hvpc = 2**(dims-1) # half of vpc (see next line)
		self.vpc = self.hvpc * 2 # vertices per (hypercube) cell
		self.coordVectors = dict()

		# GENERATE EVERY PERMUTATION OF N 1's AND 0's :
		# count from [0 --> 2ᴺ) and convert to binary
		# this is needed to find every vertex of a hypercube
		self.formatType = str(self.dims) + 'b'
		self.offsets = [tuple((1 if digit=='1' else 0) for digit in format(i,self.formatType)) for i in range(self.vpc)]

		# NORMALIZE VALUE BETWEEN [-1,1]
		# the most extreme possible values arise in the exact middle of a hypercube -
		# when the integral vectors all point at or from the middle.
		# the dot product in this case will equal:
		# (+/-) ||U|| * ||V||;
		# since the magnitudes of the integral vectors are always 1,
		# the resulting value is decided by the magnitudes of the directional vectors,
		# which, in the middle of an N-Dimensional hypercube, are equal to:
		# sqrt(N * ½²);
		# therefore, the dot products can be normalized to [-1,1] by multiplying -
		# by 1/sqrt(N * ½²)
		self.multiplier = 1/((self.dims * 0.25)**0.5)

	def get(self, point):
		rmdrs = unaryVecOp(point, lambda a:a%1) # remainders of point modulo 1
		baseCoord = vecDif(point,rmdrs) # lowest integral point coordinate
		# if the point lays on an int coord, pretend it doesn't
		if(vecLen(rmdrs) == 0):
			rmdrs = tuple(i+0.0000001 for i in rmdrs)
		# list of dot products (U•V)
		dots = []
		# generate the surrounding int coordinates
		for i in range(self.vpc):
			# get surrounding int coordinate;
			# offset the base coord by every permutation of 1's and 0's
			coord = vecSum(baseCoord, self.offsets[i])
			# get V (directional vector from vertex to point)
			dirVec = vecDif(coord, point)
			# tile the corner coords - do this AFTER getting dirVec
			if self.tiling != None:
				coord = binaryVecOp(coord, self.tiling, lambda a,b : (a if b==None else a%b))
			# get the U vector at the int coordinate
			coordVec = self.coordVectors.get(coord, None)
			if coordVec == None:
				coordVec = randomVector(self.dims)
				self.coordVectors[coord] = coordVec
			dots.append(dotProduct(dirVec,coordVec))
		# reverse - flipped bits change end-first and remainders are front-first
		rmdrs = rmdrs[::-1]
		# interpolate between all dot products
		for n in range(self.dims):
			# ease the interpolation
			alpha = self.easing(rmdrs[n])
			# add up each pair of dot products in their original order.
			#   their order is/was decided by the offset array;
			# the last bit of their indices alternates, so -
			# in each iteration, the same axis/bit is flipped between every pair.
			# as the outer loop runs down, the list is split in half by interpolation, and -
			# the last bit is effectively stripped, meaning the axis being -
			# interpolated changes with each iteration
			dots = [lerp(dotB,dotA,alpha) for dotA,dotB in zip(dots[::2], dots[1::2])]
			# eg N=3: 8 initial dot products:
			# notice how the last bit of each coordinate is flipped for each pair
			# *-----------------------------------------------------------------*
			# |                                                                 |
			# |       000--x--001   010--x--011    100--x--101   110--x--111    |
			# |  1:        |             |              |             |         |
			# |           00*-----y-----01*            10*-----y-----11*        |
			# |  2:               |                            |                |
			# |                  0**------------z-------------1**               |
			# |  3:                             |                               |
			# |                            final value                          |
			# |                                                                 |
			# *-----------------------------------------------------------------*

		return dots[0] * self.multiplier # normalize & return

chars = list('.:-=+*#%&@')
gen = NoiseGenND(3)
x=-1
increment = 0.05
nums = []
while x<1:
	y=-2
	while y<2:
		row = []
		nums.append(row)
		z=-2
		while z<2:
			row.append(gen.get((x,y,z)))
			z+=0.05
		y+=0.05
	x+=0.2


# bounds = 2;
# increment = 0.05
# nums = []
# y = -bounds
# while y < bounds:
# 	line = ''
# 	row = []
# 	nums.append(row)
# 	x=-bounds
# 	while x < bounds:
# 		row.append(gen.get((x,y)))
# 		x+=increment
# 	y+=increment
a = np.array(nums)
print(np.amax(a))
print(np.amin(a))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
