import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import pickle
import time
import matplotlib.patches as mpatches
import sys

fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)


OutputDir = sys.argv[1]
def animate(i):
    Risidual = pd.read_csv(OutputDir+"/R.dat", delimiter='\n')
    dt = pd.read_csv(OutputDir+"/dt.dat", delimiter='\n')
    ax1.clear()
    ax1.plot(Risidual,'-bo',markevery=[len(Risidual)-1],markeredgecolor='red',markerfacecolor='red')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$R_{max}$    ($NKT/R_{DSP}$)')
    ax1.set_xlabel(r'timstep')


    E = pd.read_csv(OutputDir+"/E.dat", delimiter='\n')
    ax2.clear()
    ax2.plot(E,'-bo',markevery=[len(E)-1],markeredgecolor='red',markerfacecolor='red')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$E$    ($NKT$)')
    ax2.set_xlabel(r'timstep')
    
    # ax2.clear()
    # ax2.plot(Risidual,'-bo',markevery=[len(Risidual)-1],markerfacecolor='red',markeredgecolor='red')
    # ax2.set_yscale('log')
    # ax2.set_xlim([len(Risidual)-300,len(Risidual)+10])

    
    # ax2.clear()
    # ax2.plot(dt,'-bo',markevery=[len(dt)-1],markeredgecolor='red',markerfacecolor='red')




    # ax2.clear()
    
    # numpyTris=pickle.load(open("output-FIREtest/numpyTris.pickle","rb"))
    # nodeData=pickle.load(open("output-FIREtest/nodedata-pickle.xyz.pickle","rb"))
    # triData=pickle.load(open("output-FIREtest/tridata-pickle.xyz.pickle","rb"))
    # wallData=pickle.load(open("output-FIREtest/walldata-pickle.xyz.pickle","rb"))

    # ax2.plot([wallData[0],wallData[1]],[wallData[2],wallData[2]],'g')
    # ax2.plot([wallData[0],wallData[1]],[wallData[3],wallData[3]],'g')
    # ax2.plot([wallData[0],wallData[0]],[wallData[3],wallData[2]],'g')
    # ax2.plot([wallData[1],wallData[1]],[wallData[3],wallData[2]],'g')
    # phi = 875.0 /(nodeData[0].max()-nodeData[0].min())/(nodeData[1].max()-nodeData[1].min())
    # strPhi = "{:.2f}".format(phi)
    # ax2.quiver(nodeData[0], nodeData[1],nodeData[2],nodeData[3])

    # # F=[(nodeData[2][i]**2+nodeData[3][i]**2)**0.5 for i in range(len(nodeData[0]))]
    # # ArrowLabel="{:.5f}".format(max(F))
    # # xmin, xmax, ymin, ymax = plt.axis()
    # # rect = mpatches.Rectangle((xmin,ymin),4.7,0.7,linewidth=1,facecolor='w',edgecolor='k')
    # # # plt.ax2.add_patch(rect)
    # # ax2.quiverkey(Q1, 0.079,0.015, max(F), ArrowLabel, coordinates='axes',color='r',labelpos='E')
    # # # ax2.tripcolor(nodeData[0], nodeData[1],numpyTris,facecolors=(triData[12]-triData[9]),edgecolors='grey',cmap='jet')
    
    # ax2.axis('equal')

    plt.title(OutputDir)

    # ax2.title(r'shear stress,  $1/N\Omega = 133$, $\phi=$'+strPhi)

ani1 = animation.FuncAnimation(fig1, animate, interval=3000)
ani2 = animation.FuncAnimation(fig2, animate, interval=3000)
# ani2 = animation.FuncAnimation(fig2, animate, interval=1)
plt.show()