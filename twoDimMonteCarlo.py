from __future__ import division
import numpy as np
from math import sin,cos,tan
import pylab as plt
degree30=30*np.pi/180.0
class Simulator:
	def __init__(self,nxMax,nyMax,temperature):
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
	def draw(self):
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
		line=plt.Polygon(xyCoord,color=self.color, edgecolor=self.color)#fill=None
		plt.gca().add_line(line)
		#xyCoord=np.array(xyCoord)
		#x=xyCoord[:,0]
		#y=xyCoord[:,1]
		
		#poly=Polygon(xyCoord)
		#x,y=poly.exterior.xy
		#plt.plot(x,y,'r')
		return


plt.figure()
nxMax=40
nyMax=int(nxMax*cos(degree30))#-sin(degree30)*nxMax
box=Simulator(nxMax,nyMax,0.5)
box.draw()
#plt.xlim((0,2*nMax+1))
#plt.ylim((0,2*nMax+1))
plt.axis('equal')
plt.axis([0,2*nxMax+1,0,2*nyMax+1])
plt.show()
