import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import shutil

def initiateDirectories():
    if not os.path.exists('pictures'):
    	os.mkdir('pictures')
    # if not os.path.exists('pictures/pressure'):
    # 	os.mkdir('pictures/pressure')
    if not os.path.exists('pictures/shear'):
    	os.mkdir('pictures/shear')
    # if not os.path.exists('pictures/areaRatio'):
    # 	os.mkdir('pictures/areaRatio')
    if not os.path.exists('pictures/velocity'):
    	os.mkdir('pictures/velocity')
    if not os.path.exists('pictures/surfaces'):
        os.mkdir('pictures/surfaces')

    if not os.path.exists('pictures/consistency'):
        os.mkdir('pictures/consistency')
    # if not os.path.exists('pictures/CSXY'):
    #     os.mkdir('pictures/CSXY')
    # if not os.path.exists('pictures/CSYX'):
    #     os.mkdir('pictures/CSYX')
    # if not os.path.exists('pictures/FXY'):
    #     os.mkdir('pictures/FXY')
    # if not os.path.exists('pictures/FYX'):
    #     os.mkdir('pictures/FYX')
    # if not os.path.exists('pictures/FXY-FYX'):
    #     os.mkdir('pictures/FXY-FYX')
    # if not os.path.exists('pictures/CSXY-CSYX'):
    #     os.mkdir('pictures/CSXY-CSYX')
initiateDirectories()


def readSurfaceMeshes(surfaceMeshesFileName="surfaceMeshes-A.dat"):
    surfaceMeshesFile=open(surfaceMeshesFileName)
    surfaceMeshes=[]
    line=surfaceMeshesFile.readline()
    assert(line=="Number of surfaces\n")
    line=surfaceMeshesFile.readline()
    numSurfaces=int(line.split()[0])
    line=surfaceMeshesFile.readline()
    numNodesPerMesh=int(line.split()[0])
    surfaceIndex=0
    while(surfaceIndex<numSurfaces):
        thisMesh=[]
        while(1):
            line=surfaceMeshesFile.readline()
            if(line=="\n"): break
            thisMesh.append(int(line.split()[0]))
        surfaceIndex+=1
        surfaceMeshes.append(thisMesh)
    return surfaceMeshes
surfaceMeshes=readSurfaceMeshes()
augSurfaceMeshes=readSurfaceMeshes(surfaceMeshesFileName="compressing/augmentedSurfaceMeshes.txt")

def readPerNodesData():
    step = sys.argv[1]
    nodeFile=open("compressing/dataPerNode-"+str(step)+".txt")
    line=nodeFile.readline()
    assert(line.split()[0]== "numNodes")
    numNodes=int(line.split()[1])
    line=nodeFile.readline()
    assert(line.split()[0]== "kTOverOmega")
    kTOverOmega = int(line.split()[1]);
    line=nodeFile.readline()
    assert(line.split()[0]== "phi")
    strPhi = str(line.split()[1])
    line=nodeFile.readline()
    assert(line.split()[0]== "timeStep")
    timeStep = int(line.split()[1])
    line=nodeFile.readline()
    assert(line.split()[0]== "wallsLRBT")
    leftPos = float(line.split()[1])
    rightPos = float(line.split()[2])
    botPos = float(line.split()[3])
    topPos = float(line.split()[4])

    nodesX=np.zeros(numNodes)
    nodesY=np.zeros(numNodes)
    forceX=np.zeros(numNodes)
    forceY=np.zeros(numNodes)
    errForceX=np.zeros(numNodes)
    errForceY=np.zeros(numNodes)

    nodeFile.readline()
    nodeFile.readline()
    for i in range(numNodes):
        line=nodeFile.readline().split()
        nodesX[i]=float(line[1])
        nodesY[i]=float(line[2])
        forceX[i]=float(line[3])
        forceY[i]=float(line[4])
        errForceX[i]=float(line[5])
        errForceY[i]=float(line[6])

    nodeFile.close()
    return step, numNodes,leftPos,rightPos, topPos, botPos, nodesX, nodesY, forceX, forceY, errForceX, errForceY, strPhi, kTOverOmega
step, numNodes,leftPos,rightPos, topPos, botPos, nodesX, nodesY, forceX, forceY, errForceX, errForceY, strPhi, kTOverOmega=readPerNodesData()

def readPeriodicPerNode():

    augNodeFile=open("compressing/dataPerNodePeriodicImages-"+str(step)+".txt")
    line=augNodeFile.readline()
    assert(line.split()[0]== "numNodes")
    augNumNodes=int(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "kTOverOmega")
    kTOverOmega = int(line.split()[1]);
    line=augNodeFile.readline()
    assert(line.split()[0]== "phi")
    strPhi = str(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "timeStep")
    timeStep = int(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "wallsLRBT")
    leftPos = float(line.split()[1])
    rightPos = float(line.split()[2])
    botPos = float(line.split()[3])
    topPos = float(line.split()[4])

    line=augNodeFile.readline()
    assert(line.split()[0]== "cellSizeX")
    cellSizeX = float(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "cellSizeY")
    cellSizeY = float(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "effectiveLeftPos")
    effLeftPos = float(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "effectiveBotPos")
    effBotPos = float(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "numCellsX")
    numCellsX = int(line.split()[1])
    line=augNodeFile.readline()
    assert(line.split()[0]== "numCellsY")
    numCellsY = int(line.split()[1])

    augNodesX=np.zeros(augNumNodes)
    augNodesY=np.zeros(augNumNodes)


    augNodeFile.readline()
    augNodeFile.readline()
    for i in range(augNumNodes):
        line=augNodeFile.readline().split()
        augNodesX[i]=float(line[1])
        augNodesY[i]=float(line[2])
        

    augNodeFile.close()
    return augNumNodes, timeStep, leftPos, rightPos, botPos, topPos, cellSizeX, cellSizeY, numCellsX, numCellsY, effBotPos, effLeftPos, augNodesX, augNodesY






 #### read tri data
augNumNodes, timeStep, leftPos, rightPos, botPos, topPos, cellSizeX, cellSizeY, numCellsX, numCellsY, effBotPos, effLeftPos, augNodesX, augNodesY=readPeriodicPerNode()

def readPerEleData():
    triFile=open("compressing/dataPerEle-"+str(step)+".txt")
    line=triFile.readline()
    assert(line.split()[0]== "numElements")
    numElements = int(line.split()[1])

    for i in range(9):
    	triFile.readline().split()

    refArea=np.zeros(numElements)
    areaRatio=np.zeros(numElements)
    PK1StressXX=np.zeros(numElements)
    PK1StressXY=np.zeros(numElements)
    PK1StressYX=np.zeros(numElements)
    PK1StressYY=np.zeros(numElements)
    CStressXX=np.zeros(numElements)
    CStressXY=np.zeros(numElements)
    CStressYX=np.zeros(numElements)
    CStressYY=np.zeros(numElements)
    FXX=np.zeros(numElements)
    FXY=np.zeros(numElements)
    FYX=np.zeros(numElements)
    FYY=np.zeros(numElements)


    for i in range(numElements):
        line = triFile.readline().split()
        refArea[i]=float(line[1])
        areaRatio[i]=float(line[2])
        PK1StressXX[i]=float(line[7])
        PK1StressXY[i]=float(line[8])
        PK1StressYX[i]=float(line[9])
        PK1StressYY[i]=float(line[10])
        CStressXX[i]=float(line[11])
        CStressXY[i]=float(line[12])
        CStressYX[i]=float(line[13])
        CStressYY[i]=float(line[14])
        FXX[i]=float(line[3])
        FXY[i]=float(line[4])
        FYX[i]=float(line[5])
        FYY[i]=float(line[6])

    triFile.close()

    elements=[]
    elementsFile=open("compressing/elements.txt")

    for i in range(numElements):
    	line=elementsFile.readline().split()
    	elements.append([int(line[0]),int(line[1]),int(line[2])])
    elementsFile.close()
    return numElements, refArea, areaRatio,PK1StressXX, PK1StressXY, PK1StressYX, PK1StressYY, CStressXX, CStressXY, CStressYX, CStressYY, elements
numElements, refArea, areaRatio,PK1StressXX, PK1StressXY, PK1StressYX, PK1StressYY, CStressXX, CStressXY, CStressYX, CStressYY, elements=readPerEleData()

def plotVerletGrid():
    plt.plot([effLeftPos+numCellsX*cellSizeX,effLeftPos+numCellsX*cellSizeX],[effBotPos,effBotPos+numCellsY*cellSizeY],'grey')
    plt.plot([effLeftPos+numCellsX*cellSizeX,effLeftPos],[effBotPos+numCellsY*cellSizeY,effBotPos+numCellsY*cellSizeY],'grey')
    for i in range(numCellsX):
        plt.plot([effLeftPos+i*cellSizeX,effLeftPos+i*cellSizeX],[effBotPos,effBotPos+numCellsY*cellSizeY],'grey')
    for i in range(numCellsY):
        plt.plot([effLeftPos,effLeftPos+numCellsX*cellSizeX],[effBotPos+i*cellSizeY,effBotPos+i*cellSizeY],'grey')

def plotSurfaceBoundaries(meshes,j=3):
    for mesh in meshes:  
        plt.plot([augNodesX[mesh[0]], augNodesX[mesh[-1]]],[augNodesY[mesh[0]], augNodesY[mesh[-1]]], c=colors[j])
        for k in range(len(mesh)-1):
            plt.plot([augNodesX[mesh[k]], augNodesX[mesh[k+1]]],[augNodesY[mesh[k]], augNodesY[mesh[k+1]]], c=colors[j])
        j+=1

def plotWalls():
    plt.plot([leftPos,rightPos],[botPos,botPos],'yellow')
    plt.plot([leftPos,rightPos],[topPos,topPos],'yellow')
    plt.plot([leftPos,leftPos],[topPos,botPos],'yellow')
    plt.plot([rightPos,rightPos],[topPos,botPos],'yellow')

###############################################################################################################################################################################
colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000']

def plot_velocity():

    plt.close('all')  

    plt.figure(figsize=(12,12))

    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000','#e6194B', '#3cb44b', '#ffe119', '#4363d8','#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#473C8B', '#000000']
    plt.gcf().clear()
    plotWalls()
    plotVerletGrid()
    plotSurfaceBoundaries(augSurfaceMeshes)
    plt.scatter(nodesX,nodesY)
    Q1=plt.quiver(nodesX, nodesY, forceX, forceY,width=0.002)
    plt.title(r'velocity,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    plt.axis('equal')

    F=[(forceX[ii]**2+forceY[ii]**2)**0.5 for ii in range(len(forceX))]
    ArrowLabel="{:.5f}".format(max(F))
    xmin, xmax, ymin, ymax = plt.axis()
    plt.scatter(augNodesX,augNodesY)
    plt.scatter(nodesX,nodesY, c='r')
    plt.axis([xmin,xmax,ymin,ymax])
    plt.quiverkey(Q1, 0.079,0.015, max(F), ArrowLabel, coordinates='axes',color='r',labelpos='E')

    plt.savefig('pictures/velocity/velocity-'+str(timeStep).zfill(2)+'.png',dpi=300)

def plot_presssure():
    plt.figure()
    plt.gcf().clear()
    plt.plot([leftPos,rightPos],[botPos,botPos],'g')
    plt.plot([leftPos,rightPos],[topPos,topPos],'g')
    plt.plot([leftPos,leftPos],[topPos,botPos],'g')
    plt.plot([rightPos,rightPos],[topPos,botPos],'g')
    plt.tripcolor(nodesX, nodesY,elements,facecolors=-(CStressYY+CStressXX)/2.0,edgecolors='grey',cmap='jet')
    plt.title(r'pressure,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    plt.colorbar()
    plt.axis('equal')
    plt.savefig('pictures/pressure/pressure-'+str(timeStep).zfill(2)+'.png',dpi=300)

def plot_shear():
    plt.gcf().clear()
    plt.tripcolor(nodesX, nodesY,elements,facecolors=-(CStressYY-CStressXX)/2.0,edgecolors='grey',cmap='jet')
    plt.title(r'shear,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    plt.colorbar()
    plt.axis('equal')
    plt.savefig('pictures/shear/shear-'+str(timeStep).zfill(2)+'.png',dpi=80)

def plot_areaRatio():  
    plt.figure()
    plt.gcf().clear()
    plt.plot([leftPos,rightPos],[botPos,botPos],'g')
    plt.plot([leftPos,rightPos],[topPos,topPos],'g')
    plt.plot([leftPos,leftPos],[topPos,botPos],'g')
    plt.plot([rightPos,rightPos],[topPos,botPos],'g')
    plt.tripcolor(nodesX, nodesY,elements,facecolors=areaRatio,edgecolors='grey',cmap='jet')
    plt.title(r'shear,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    plt.colorbar()
    plt.axis('equal')
    plt.savefig('pictures/areaRatio/areaRatio-'+str(timeStep).zfill(2)+'.png',dpi=300)

    plt.close('all')  

def plot_surfaceParticles():

    plt.figure()
    plt.gcf().clear()

    j = 0
    for p in surfaceMeshes:
        xs = []
        ys = []
        for i in p:    
            plt.scatter(nodesX[i],nodesY[i], c=colors[j])
            xs.append(nodesX[i])
            ys.append(nodesY[i])
            # plt.text(nodesX[i],nodesY[i],str(i))
        # plt.text(np.array(xs).mean(),np.array(ys).mean(),str(j))
        j+=1
    plt.title(r'surfaceNodes,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    plt.axis('equal')
    plt.savefig('pictures/surfaces/surfaces-'+str(timeStep).zfill(2)+'.png',dpi=80)

    plt.close('all') 

def plot_force_energy_consistency():

    plt.figure(figsize=(12,12))
    plt.gcf().clear()
    # ghost nodes
    plt.scatter(augNodesX,augNodesY, c='blue')
    # actual nodes
    plt.scatter(nodesX,nodesY, c='red')
    plotWalls()
    plotVerletGrid()
    plotSurfaceBoundaries(augSurfaceMeshes)


    for i in range(numNodes):
        # plt.scatter(nodesX[i],nodesY[i])
        plt.text(nodesX[i],nodesY[i],"{:.5f}".format(errForceX[i]))
    # Q1=plt.quiver(nodesX, nodesY, forceX, forceY,width=0.002)
    # F=[(forceX[ii]**2+forceY[ii]**2)**0.5 for ii in range(len(forceX))]
    # ArrowLabel="{:.5f}".format(max(F))
    plt.title(r'$(F_x  \delta x) /\delta E$,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    # plt.colorbar()
    plt.axis('equal')
    plt.savefig('pictures/consistency/consistencyX-'+str(timeStep).zfill(2)+'.png',dpi=280)
    plt.close('all') 


    plt.figure(figsize=(10,10))
    plt.gcf().clear()
    plotWalls()
    plotVerletGrid()
    plotSurfaceBoundaries(augSurfaceMeshes)

    plt.scatter(augNodesX,augNodesY, c='blue')
    plt.scatter(nodesX,nodesY, c='red')
    for i in range(numNodes):
        # plt.scatter(nodesX[i],nodesY[i])
        plt.text(nodesX[i],nodesY[i],"{:.5f}".format(errForceY[i]))
    # Q1=plt.quiver(nodesX, nodesY, forceX, forceY,width=0.002)
    # F=[(forceX[ii]**2+forceY[ii]**2)**0.5 for ii in range(len(forceX))]
    # ArrowLabel="{:.5f}".format(max(F))
    plt.title(r'$(F_y  \delta y) /\delta E$,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
    # plt.colorbar()
    plt.axis('equal')
    plt.savefig('pictures/consistency/consistencyY-'+str(timeStep).zfill(2)+'.png',dpi=280)
    plt.close('all') 
plot_force_energy_consistency()




# ############# XY ###################

# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(CStressXY)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'CS_XY,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/CSXY/CSXY-'+str(timeStep).zfill(2)+'.png',dpi=300)


# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(FXY)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'F_XY,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/FXY/FXY-'+str(timeStep).zfill(2)+'.png',dpi=300)

############# SYX ###################

# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(CStressYX)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'CS_YX,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/CSYX/CSYX-'+str(timeStep).zfill(2)+'.png',dpi=300)

# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(FYX)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'F_YX,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/FYX/FYX-'+str(timeStep).zfill(2)+'.png',dpi=300)


############# YX-XY ###################

# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(CStressYX-CStressXY)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'CS_XY-C_YX,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/CSXY-CSYX/CSXY-CSYX-'+str(timeStep).zfill(2)+'.png',dpi=300)

# plt.figure()
# plt.gcf().clear()
# plt.plot([leftPos,rightPos],[botPos,botPos],'g')
# plt.plot([leftPos,rightPos],[topPos,topPos],'g')
# plt.plot([leftPos,leftPos],[topPos,botPos],'g')
# plt.plot([rightPos,rightPos],[topPos,botPos],'g')
# plt.tripcolor(nodesX, nodesY,elements,facecolors=-(FYX-FXY)/2.0,edgecolors='grey',cmap='jet')
# plt.title(r'F_XY-F_YX,  $1/N\Omega = $'+str(kTOverOmega)+r', $\phi=$'+strPhi)
# plt.colorbar()
# plt.axis('equal')
# plt.savefig('pictures/FXY-FYX/FXY-FYX-'+str(timeStep).zfill(2)+'.png',dpi=300)
