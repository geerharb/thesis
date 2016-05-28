from __future__ import division
import numpy as np
from math import sin,cos,tan
import pylab as plt
from random import randint
import random
degree30=30*np.pi/180.0
class Simulator:
	def __init__(self,nxMax,nyMax,temperature,boxScale=1.0):
		self.boxScale=boxScale
		hexss=[]
		alpha=4
		self.nxMax=nxMax
		self.nyMax=nyMax
		self.temperature=temperature
		self.gasDensity=(1-(1-4*temperature/alpha)**0.5)/2.0
		self.liquidDensity=(1-(1+4*temperature/alpha)**0.5)/2.0
		for ny in range(0,nyMax):
			ns=[]
			for nx in range(0,nxMax):
				ns.append(Hexagon(nx,ny,nxMax=nxMax,nyMax=nyMax,color='b',
									density=self.gasDensity))
			hexss.append(ns)
		
		#set the liquid
		centerX=hexss[0][int(nxMax/2)].x#hexss[0][nMax-1].x/2.0+0.5#(2*int(nMax/2)+1)#/2.0-2.5
		centerY=hexss[int(nyMax/2)][0].y#hexss[nMax-1][0].y/2.0+0.5#centerX
		rLiquid=min((2.5/4*centerX)**2.0,(2.5/4*centerY)**2.0)
		for ny in range(0,nyMax):
			for nx in range(0,nxMax):
				r=(hexss[ny][nx].x-centerX)**2.0+(hexss[ny][nx].y-centerY)**2.0
				if r<rLiquid:
					hexss[ny][nx].color='r'
					hexss[ny][nx].density=self.liquidDensity
		self.hexss=hexss
		return
	def validIndex(self,nx,ny):
		while nx<0:
			ny-=int(self.nxMax/2)
			nx+=self.nxMax
		while nx>=self.nxMax:
			ny+=int(self.nxMax/2)
			nx-=self.nxMax
		while ny<0:
			ny+=self.nyMax
		while ny>=self.nyMax:
			ny-=self.nyMax
		return [nx,ny]
	def nearestNeighbors(self,nx,ny):
		neighbors=[]
		neighbors.append(self.validIndex(nx+1,ny))
		neighbors.append(self.validIndex(nx-1,ny))
		neighbors.append(self.validIndex(nx,ny+1))
		neighbors.append(self.validIndex(nx,ny-1))
		neighbors.append(self.validIndex(nx-1,ny+1))
		neighbors.append(self.validIndex(nx+1,ny-1))
		return neighbors
	def energy(self,nx,ny):
		L=2*(3**0.5)/3*self.boxScale
		index=validIndex(nx,ny)
		density0=hexxs[index[1]][index[0]].density
		energyTotal=0
		for index in nearestNeighbors(nx,ny):
			density1=hexxs[index[1]][index[0]].density
			energyTotal+=abs(density1-density0)*L
		return energyTotal
	
	def simulate(self,validMoves=1000,maxMoves=1000000):
		while validMoves>0 and maxMoves>0:
			maxMoves-=1
			nx1=randint(0,self.nxMax)
			ny1=randint(0,self.nyMax)
			neighbor=nearestNeighbors(nx1,nx2)
			neighbor=neighbor[randint(0,len(neighbor))]
			nx2=neighbor[0]
			ny2=neighbor[1]
			if abs(hexss[ny1][nx1]-hexss[ny2][nx2])<1e-7: continue
			energyBefore=energy(nx1,ny1)+energy(nx2,ny2)
			hexss[ny1][nx1],hexss[ny2][nx2]=hexss[ny2][nx2],hexss[ny1][nx1]
			energyAfter=energy(nx1,ny1)+energy(nx2,ny2)
			if energyAfter<energyBefore:
				validMoves-=1
				continue
			acceptanceRate=np.exp(-(energyAfter-energyBefore)/self.temperature)
			if random.uniform(0,1.0)<acceptanceRate: 
				validMoves-=1
				continue
			hexss[ny1][nx1],hexss[ny2][nx2]=hexss[ny2][nx2],hexss[ny1][nx1]
		return
	def draw(self):
		#for index in self.nearestNeighbors(randint(0,self.nxMax),randint(0,self.nyMax)):
		#	self.hexss[index[1]][index[0]].color='g'
		for ny in range(0,self.nyMax):
			for nx in range(0,self.nxMax):
				self.hexss[ny][nx].draw()
class Hexagon:
	def __init__(self,nx,ny,nxMax=1,nyMax=1,density=1.0,color='r'):
		self.color=color
		self.nx=nx
		self.ny=ny
		self.nxMax=nxMax
		self.nyMax=nyMax
		self.density=density
		self.x=2*1.01*(3**0.5)/3 #initial offset
		self.y=1.01	#initial offset
		self.y+=2*ny+sin(degree30)*nx*2
		self.x+=cos(degree30)*nx*2
		if(self.y-nyMax*2>0.05):
			self.y-=nyMax*2
		return
	
	def draw(self):
		polySize=0.8
		L=2*(3**0.5)/3*polySize
		
		xyCoord=[[-sin(degree30)*L,polySize],[sin(degree30)*L,polySize],[L,0],
			[sin(degree30)*L,-polySize],[-sin(degree30)*L,-polySize],[-L,0]]#,[-sin(degree30)*L,1]]
		for i in range(len(xyCoord)):
			xyCoord[i][0]+=self.x
			xyCoord[i][1]+=self.y
		line=plt.Polygon(xyCoord,color=self.color)#, edgecolor=self.color)#fill=None
		plt.gca().add_line(line)
		#xyCoord=np.array(xyCoord)
		#x=xyCoord[:,0]
		#y=xyCoord[:,1]
		
		#poly=Polygon(xyCoord)
		#x,y=poly.exterior.xy
		#plt.plot(x,y,'r')
		return


plt.figure()
nxMax=10 #must be multiple of 2
nyMax=int(nxMax*cos(degree30))+1#-sin(degree30)*nxMax
box=Simulator(nxMax,nyMax,0.5)
box.simulate(validMoves=1)
box.draw()
#plt.xlim((0,2*nMax+1))
#plt.ylim((0,2*nMax+1))
plt.axis('equal')
plt.axis([0,2*nxMax+1,0,2*nyMax+1])
plt.show()
