import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path






def process_geo(x, y, subs, nndist= 1/np.sqrt(3)):
    '''tidy edges, generate distance matrix, and classify sites from coordinates alone'''
    
    x1, y1, subs1, sysdist = tidy_geometry(x, y, subs, nndist=nndist) 

    inside = np.arange(len(x1))
    coords = np.column_stack((x1, y1))
    #sysdist = distance_matrix(coords, coords)

    sysneigh, neighlists =count_and_track_neighbours(inside, sysdist, nndist ) 
    sysedges=classify_sites(sysneigh, neighlists)

    return x1, y1, subs1, sysedges, sysdist


def tidy_geometry(x, y, subs, nndist= 1/np.sqrt(3)):
    '''removes dangling atoms from imported geometries'''
    inside = np.arange(len(x))
    coords = np.column_stack((x, y))
    sysdist = distance_matrix(coords, coords)
    
    #count neighbours and remove dangling atoms
    newlist=count_neighbours(inside, sysdist, nndist)
    newinside=inside
    while len(newinside[newlist<2]) > 0:
        newinside=newinside[newlist>1]
        newlist=count_neighbours(newinside, sysdist, nndist)
        
    return x[newinside], y[newinside], subs[newinside], sysdist[np.ix_(newinside, newinside)]


def count_neighbours(site_list, dmat, neigh_dist):
    '''counts the number of neighbours of each site'''
    neighs = [np.isclose(dd,neigh_dist).sum() for dd in dmat[:, site_list][site_list, :]]
    return np.array(neighs)


def count_and_track_neighbours(site_list, dmat, neigh_dist):
    '''counts and keeps track of the number of neighbours of each site'''
    outlist=[]
    neighs=[]
    for dd in dmat[np.ix_(site_list, site_list)]:
        inlist =np.isclose(dd,neigh_dist)
        outlist.append(list(np.array(np.where(inlist==True)).flatten()))
        neighs.append(inlist.sum())
    return np.array(neighs), outlist


def classify_sites(sysneigh, neighlists):
    '''determines whether sites are type 0 (bulk), 1 (zigzag), 2 (corner) or 3(armchair)'''
    classification = np.zeros_like(sysneigh)
    
    #for each site, keep track of how many neighbours its neighbours have
    for i, neigh in enumerate(sysneigh):
        if neigh == 2:
            sumneigh = sysneigh[neighlists[i]].sum()
            if(sumneigh==6):
                classification[i] =1
            elif (sumneigh==4):
                classification[i] =2
            elif (sumneigh==5):
                classification[i] =3
    return classification

def map_site_types (x1, y1, subs1, sysedges):
    '''creates a map of the different types of sites in a structure
    (similar to Fig 1(b) in the paper)'''
    
    # zigzag sites
    zzsame = np.where( (sysedges==1) & (subs1==1))[0]
    zzopp = np.where( (sysedges==1) & (subs1==-1))[0]

    bsame = np.where( (sysedges==0) & (subs1==1))[0]
    bopp = np.where( (sysedges==0) & (subs1==-1))[0]

    # corner sites
    corsame = np.where( (sysedges==2) & (subs1==1))[0]
    coropp = np.where( (sysedges==2) & (subs1==-1))[0]

    # armchair sites
    acsame = np.where( (sysedges==3) & (subs1==1))[0]
    acopp = np.where( (sysedges==3) & (subs1==-1))[0]

    fig, ax =plt.subplots(figsize=(10, 8))
    b11 = ax.scatter(x1[bsame], y1[bsame],  color='darkgrey', alpha=0.8, label='bulk', s=10)
    ax.scatter(x1[bopp], y1[bopp],  edgecolors='darkgrey', facecolor='w', alpha=0.8, linewidth=1.6, s=10)
    z11 = ax.scatter(x1[zzsame], y1[zzsame],  marker='^', color='blue', alpha=0.8, label='zigzag', s=13)
    ax.scatter(x1[zzopp], y1[zzopp],  marker='^', edgecolors='blue', facecolor='w', alpha=0.8, linewidth=1.2, s=13)
    c11 = ax.scatter(x1[corsame], y1[corsame],  marker='s', color='orange', alpha=0.8, label='corner',  s=13)
    ax.scatter(x1[coropp], y1[coropp],  marker='s', edgecolors='orange', facecolor='w', alpha=0.8, linewidth=1.2, s=13)
    a11 = ax.scatter(x1[acsame], y1[acsame],  marker='D', color='green', alpha=0.8, label='armchair',  s=13)
    ax.scatter(x1[acopp], y1[acopp], marker='D', edgecolors='green', facecolor='w', alpha=0.8, linewidth=1.2, s=13)

    leg1 = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 0.95),  borderaxespad=0.0, borderpad=0.25, 
    labelspacing=0.15, handletextpad=0.1, handlelength=1.3, frameon=False, markerscale=3.5, fontsize=18)


    ax.set_aspect('equal')
    ax.axis('off')
    #ax.set_xlim(x1.min()-1, x1.max()+1)
    
    return fig, ax


def TB_Hamiltonian(dist_mat, neigh_dist, hopping):
    '''creates a tight-binding Hamiltonian based on a distance matrix'''
    hops = np.isclose(dist_mat, neigh_dist)
    H = np.where(hops==True, hopping+0j, 0.0)
    return H


def initialise_moments_smart(subs, types):
    '''Generates an initial guess for moments of a system, based on site classification'''
    mag = np.zeros_like(subs*1.0)
    
    mag[np.where(types==0)] = subs[np.where(types==0)] * 0.02
    mag[np.where(types==1)] = subs[np.where(types==1)] * 0.35
    mag[np.where(types==2)] = subs[np.where(types==2)] * 0.1
    mag[np.where(types==3)] = subs[np.where(types==3)] * 0.02
             
    return mag


def find_moments(ham, init, prec=0.0001, hubU=1.33, alpha=1.0, verbose=False):
    '''Finds the moments of a system self-consistently'''
    new_prec=1.0
    moments = init
    
    up_ham = np.copy(ham)
    down_ham = np.copy(ham)
    filling=len(init)
    accs=[]
    norms=[]

    
    while (new_prec > prec):
        new_moms = moments
        np.fill_diagonal(up_ham, -hubU*new_moms/2)
        np.fill_diagonal(down_ham, hubU*new_moms/2)
        upv, upvex = sorted_eigen(up_ham)
        downv, downvex = sorted_eigen(down_ham)
                  
        allv = np.append(upv, downv)        
        idx=allv.argsort()
        idx=idx[0:filling]
        idxup=idx[idx<filling]
        idxdown=idx[idx>filling-1] -filling 

        upoccs  =(abs(upvex[:,idxup])**2).sum(axis=1)
        downoccs=(abs(downvex[:,idxdown])**2).sum(axis=1)
        momsa = upoccs - downoccs
        dmom = momsa - new_moms
        moments = new_moms + alpha*dmom
        norm=np.linalg.norm(dmom)/len(dmom)

        new_prec = abs(np.max(dmom))
        if(verbose==True):
            print("Current precision: "+str(new_prec))

        accs.append(new_prec)
        norms.append(norm)

        
        
    return moments , accs, norms

def sorted_eigen(ham):
    '''returns the sorted eigenvalues and eigenvectors of a Hamiltonian'''
    val, vec = np.linalg.eig(ham)
    vv=np.real(val)
    index = vv.argsort()
    vals, vecs = vv[index], vec[:, index]
    return vals, vecs


def generate_allmaxcon_system_file (file, 
                          numzz=11, numcor=3, numac=5, 
                          max_dist=15.0):
    '''generates allmaxcon type descriptor for every site in a list of systems'''
    
    size_desc = 2*(numzz + numcor + numac)+1
    red_size = int(size_desc*(size_desc+1)/2) -size_desc 
    full_descs = np.zeros((0, red_size + 3))
    
    #full_descs = np.zeros((0,2*(numzz + numcor + numac) + 3))
    

    x, y, subs, types, moms= np.loadtxt(file, delimiter=',', unpack=True)    
    coords = np.column_stack((x, y))
    sysdist = distance_matrix(coords, coords)
    graph = np.isclose(sysdist, 1/np.sqrt(3))*1.0
    sysdist = shortest_path(graph, unweighted=True)
    syssize=len(x)
    site_desc = np.ones((syssize, red_size + 3))

    for site in np.arange(syssize):

        zzsame = np.where( (types==1) & (subs==subs[site]))[0]
        zzs_dist = sysdist[site] [zzsame]
        zzs_sort = zzs_dist.argsort()[0:numzz]
        zzs_list = zzsame[zzs_sort]

        zzopp = np.where( (types==1) & (subs==-subs[site]))[0]
        zzo_dist = sysdist[site] [zzopp]
        zzo_sort = zzo_dist.argsort()[0:numzz]
        zzo_list = zzopp[zzo_sort]

        corsame = np.where( (types==2) & (subs==subs[site]))[0]
        cors_dist = sysdist[site] [corsame]
        cors_sort = cors_dist.argsort()[0:numcor]
        cors_list = corsame[cors_sort]

        coropp = np.where( (types==2) & (subs==-subs[site]))[0]
        coro_dist = sysdist[site] [coropp]
        coro_sort = coro_dist.argsort()[0:numcor]
        coro_list = coropp[coro_sort]

        acsame = np.where( (types==3) & (subs==subs[site]))[0]
        acs_dist = sysdist[site] [acsame]
        acs_sort = acs_dist.argsort()[0:numac]
        acs_list = acsame[acs_sort]

        acopp = np.where( (types==3) & (subs==-subs[site]))[0]
        aco_dist = sysdist[site] [acopp]
        aco_sort = aco_dist.argsort()[0:numac]
        aco_list = acopp[aco_sort]

        full_list=np.hstack(([site], zzs_list, zzo_list, cors_list, coro_list, acs_list, aco_list))

        #which elements appear in the descriptor
        temp_index=np.zeros((2*(numzz + numcor + numac))+1, dtype=int)
        temp_index[0:len(zzs_list)+1]=1
        temp_index[numzz+1:numzz+len(zzo_list)+1]=1
        temp_index[2*(numzz)+1:2*numzz+len(cors_list)+1]=1
        temp_index[2*(numzz)+numcor+1:2*numzz+numcor+len(coro_list)+1]=1
        temp_index[2*(numzz+numcor)+1:2*numzz+2*numcor+len(acs_list)+1]=1
        temp_index[2*(numzz+numcor)+numac+1:2*numzz+2*numcor+numac+len(aco_list)+1]=1

        temp_desc = np.ones((2*(numzz+numcor+numac)+1, 2*(numzz+numcor+numac)+1))
        ti2 = np.where(temp_index==1)[0]
        temp_desc[np.ix_(ti2, ti2)] = sysdist[np.ix_(full_list, full_list)] / max_dist

        site_desc[site, 0:red_size ] = temp_desc[np.triu_indices(2*(numzz + numcor + numac)+1, 1)]

        temp_indices = np.where(site_desc[site]>1)
        site_desc[site][temp_indices] = 1

        site_desc[site, red_size] = subs[site]
        site_desc[site, red_size+1] = 0
        site_desc[site,red_size+2] = moms[site]

    full_descs = np.vstack((full_descs, site_desc))

    
    return full_descs   

# required information for the descriptor / model used in the paper
con_all_desc = 11, 3, 5, 'allmaxcon', "new_set.allmaxcon_11_3_5_15.0", "[1000,1000,1000,1000,1000,200].adam_mae_e1000_b2048.vdo_0.12_lr_0.0001_fac_0.5_v0", generate_allmaxcon_system_file


def compare_SC_ML (x, y, subs, moms, mom_preds, dotsize=40, figsize=12):
    '''Compare self-consistent and ML results'''

    renorm = dotsize/max(abs(mom_preds))
    fig=plt.figure(figsize=(figsize,figsize))
    ax = fig.add_subplot(221)
    ax.set_title('Actual moments', fontsize=19)
    ax.scatter(x[subs==1], y[subs==1], s=renorm*abs(moms)[subs==1], label='spin up', color='black')
    ax.scatter(x[subs==-1], y[subs==-1], s=renorm*abs(moms)[subs==-1], label='spin down', edgecolors='red', facecolors='mistyrose')
    ax.axis('off')
    fig.legend(fontsize = 14, loc='upper center', bbox_to_anchor=(0.5, 0.8))
    ax.set_aspect('equal')

    ax = fig.add_subplot(222)
    ax.set_title('Predicted moments', fontsize=19)
    ax.scatter(x[subs==1], y[subs==1], s=renorm*abs(mom_preds)[subs==1], label='spin up', color='black')
    ax.scatter(x[subs==-1], y[subs==-1], s=renorm*abs(mom_preds)[subs==-1], label='spin down', edgecolors='red', facecolors='mistyrose')
    ax.axis('off')
    ax.set_aspect('equal')

    ax = fig.add_subplot(223)
    ax.set_title('Error map', fontsize=19)
    sc=ax.scatter(x, y, c=abs(subs*mom_preds)-abs(moms), s=dotsize/4, cmap=cm.coolwarm, vmin=-0.05, vmax=0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    cax = fig.add_axes([0.26, 0.1, 0.1, 0.01])
    fig.colorbar(sc, cax=cax, orientation='horizontal', label='Error (magnitude)')

    ax = fig.add_subplot(224)
    ax.set_title('Errors', fontsize=19)
    ax.plot(abs(moms), mom_preds, 'o', alpha=1.0, ms=2)
    ax.plot([0,1], [0,1], '--')
    ax.set_ylim(0, 0.4)
    ax.set_ylabel('predicted moments', fontsize=15)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel('actual moments', fontsize=15)
    ax.set_aspect('equal')
    
    return fig, ax


def compare_levels (x, y, subs, moms, mom_preds, emin, emax):
    figlevels=plt.figure(figsize=(4, 6))
    axbands=figlevels.add_subplot(1,1,1)
    coords = np.column_stack((x, y))
    sysdist = distance_matrix(coords, coords)
    nospin_ham = TB_Hamiltonian(sysdist, 1/np.sqrt(3), -1.0)
    hubU=1.33
    up_ham=np.copy(nospin_ham)
    down_ham=np.copy(nospin_ham)
    ml_up_ham=np.copy(nospin_ham)
    ml_down_ham=np.copy(nospin_ham)

    np.fill_diagonal(up_ham, -hubU*moms/2)
    np.fill_diagonal(down_ham, hubU*moms/2)

    np.fill_diagonal(ml_up_ham, -hubU*subs*mom_preds/2)
    np.fill_diagonal(ml_down_ham, hubU*subs*mom_preds/2)

    nov, novex = sorted_eigen(nospin_ham)
    upv, upvex = sorted_eigen(up_ham)
    downv, downvex = sorted_eigen(down_ham)
    ml_upv, ml_upvex = sorted_eigen(ml_up_ham)
    ml_downv, ml_downvex = sorted_eigen(ml_down_ham)
    
    for v in nov:
        axbands.plot([0,1],[v,v], 'gray')
    for v in upv:
        axbands.plot([2,3],[v,v], 'k')
    for v in ml_upv:
        axbands.plot([3,4],[v,v], 'k--')
    for v in downv:
        axbands.plot([5,6],[v,v], 'r', linestyle='-')
    for v in ml_downv:
        axbands.plot([6,7],[v,v], 'r', linestyle='--')   

    axbands.set_ylim( emin, emax)
    axbands.set_xticklabels([])

    noindex = np.argwhere(nov<0).max()

    upindex = np.argwhere(upv<0).max()
    ml_upindex = np.argwhere(ml_upv<0).max()

    downindex = np.argwhere(downv<0).max()
    ml_downindex = np.argwhere(ml_downv<0).max()

    axbands.set_xticks([0.5, 2.5, 3.5, 5.5, 6.5])
    axbands.set_xticklabels(['NM', r'$\uparrow$', r'$(ML)$', r'$\downarrow$', r'$(ML)$'], fontsize=15)

    axbands.xaxis.tick_top()
    axbands.set_ylabel('Energy (|t|)', fontsize=20)
    
    # Hide the right and top spines
    axbands.spines['right'].set_visible(False)
    axbands.spines['top'].set_visible(False)
    axbands.spines['bottom'].set_visible(False)


    # Only show ticks on the left and bottom spines
    axbands.yaxis.set_ticks_position('left')
    axbands.xaxis.set_ticks_position('top')
    axbands.tick_params(axis='x', length=0)
    axbands.tick_params(axis='y',  labelsize=13)

      
    return figlevels, axbands


