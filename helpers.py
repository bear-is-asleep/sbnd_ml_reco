import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import plotters
import matplotlib.lines as mlines
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import yaml
from mlreco.main_funcs import process_config, prepare

#Constants needed 
particle_pid_index=5
clust_pid_index=5
energy_label_index = -1
input_planecharge_indeces = (6,7,8)
simenergydeposit_energy_index = -2

def get_max_matches(reco_xyz_noghost,true_xyz,distance=1e10):
  """
  Get max number of true voxels that are matched
  """
  tree = cKDTree(true_xyz) #Get tree to perform query
  #Return closest distance to each reco point, and indeces of that point in the truth array
  distances, indeces = tree.query(reco_xyz_noghost,k=1) 
  #indeces that match distance constraint in true and reco arrays
  reco_indeces = []
  true_indeces = []
  for i,d in enumerate(distances):
    if d<=distance: #Ignore elements that don't satisfy the criteria
      reco_indeces.append(i)
      true_indeces.append(indeces[i])
  #reco_tagged_xyz = reco_xyz_noghost[np.unique(reco_indeces)]
  reco_tagged_xyz = reco_xyz_noghost[np.unique(reco_indeces)]
  eff = len(np.unique(true_indeces))/len(true_xyz)
  pur = len(reco_tagged_xyz)/len(reco_xyz_noghost)
  return len(np.unique(true_indeces))
  
  

def purity_efficiency(reco_xyz_noghost,true_xyz,distance=0,max_matches=None):
    """
        Calculates purtiy and efficiency by matching true voxel locations to reco voxel locations
    reco_xyz_noghost : xyz coords for nonghost
    true_xyz : xyz for true parts
    distance: distance between voxels. if distance is 0, it has to be an exact match

    eff = true tagged voxel count / true voxel count
    pur = true tagged voxel count / reco voxels (noghost)
    return eff,pur,true_tagged_arr
    """
    tree = cKDTree(true_xyz) #Get tree to perform query
    #Return closest distance to each reco point, and indeces of that point in the truth array
    distances, indeces = tree.query(reco_xyz_noghost,k=1) 
    #indeces that match distance constraint in true and reco arrays
    reco_indeces = []
    true_indeces = []
    for i,d in enumerate(distances):
      if d<=distance: #Ignore elements that don't satisfy the criteria
        reco_indeces.append(i)
        true_indeces.append(indeces[i])
    #reco_tagged_xyz = reco_xyz_noghost[np.unique(reco_indeces)]
    reco_tagged_xyz = reco_xyz_noghost[np.unique(reco_indeces)]
    if max_matches is None:
      eff = len(np.unique(true_indeces))/len(true_xyz)
    else:
      eff = len(np.unique(true_indeces))/max_matches
    pur = len(reco_tagged_xyz)/len(reco_xyz_noghost)
    return pur,eff,reco_tagged_xyz,true_indeces,reco_indeces

def get_unmatched_voxels(reco,true,distance=0):
  """
  Return unmatched reco and true voxels
  """
  max_matches = get_max_matches(reco,true)
  _,_,_,max_matches_true_indeces,max_matches_reco_indeces = purity_efficiency(reco,true,distance=1e10,max_matches=max_matches)
  _,_,_,true_indeces,reco_indeces = purity_efficiency(reco,true,distance=distance,max_matches=max_matches)
  #Find unmatched indeces
  true_unmatched = []
  for i,ind in enumerate(np.unique(max_matches_true_indeces)):
    if ind not in true_indeces:
        true_unmatched.append(ind)
  reco_unmatched = []
  for i,ind in enumerate(np.unique(max_matches_reco_indeces)):
    if ind not in reco_indeces:
        reco_unmatched.append(ind)
  return true[true_unmatched],reco[reco_unmatched],true_unmatched,reco_unmatched
        
def get_voxel_distances(reco,true):
  """
  Return array of voxel distances between reco and true in x,y,z,total
  Dimensionality of (R,4)
  """
  distances = np.zeros((len(reco),4))
  for i,rec in enumerate(reco):
    min_ind = np.argmin(cdist(rec.reshape(1,3),true))
    distances[i,0] = rec[0]-true[min_ind,0]
    distances[i,1] = rec[1]-true[min_ind,1]
    distances[i,2] = rec[2]-true[min_ind,2]
    distances[i,3] = np.min(cdist(rec.reshape(1,3),true))
    #print(distances[i],min_ind)
  return distances


def bbox_voxels(xyz,xmin,xmax,ymin,ymax,zmin,zmax):
    """
    Returns coordinates inside bounding box region
    """
    ret = []
    for i,coord in enumerate(xyz):
        xbound = coord[0] > xmin and coord[0] < xmax
        ybound = coord[1] > ymin and coord[1] < ymax
        zbound = coord[2] > zmin and coord[2] < zmax
        #if xbound:
        #    print('x:',xmin,coord[0],xmax)
        #    print(coord)
        #if xbound and ybound:
        #    print('y:',ymin,coord[1],ymax)    
        #if xbound and ybound and zbound:
        #    print(xmin,coord[0],xmax,ymin,coord[1],ymax,zmin,coord[2],zmax)
        if (coord[0] > xmin and coord[0] < xmax and 
            coord[1] > ymin and coord[1] < ymax and
            coord[2] > zmin and coord[2] < zmax):
            ret.append(coord)
    return np.array(ret)
  
def plot_matching_dict(matching_dict):
  """
  make plots of metrics vs purity,etc.
  """
  fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(18, 6))
  fig_d, axs_d = plt.subplots(nrows=1,ncols=2,figsize=(16, 6)) #voxel distance
  fig_ad, axs_ad = plt.subplots(nrows=1,ncols=2,figsize=(16, 6)) #absolute voxel distance
  #fig_tw, ax_tw = plt.subplots(figsize=(6, 6))
  pur_label = mlines.Line2D([],[],color='red',label='Pur',marker='o',linestyle='')
  eff_label = mlines.Line2D([],[],color='blue',label='Eff',marker='o',linestyle='')
  ticks = [[],[],[]]
  ticks_d = ticks.copy()
  #distances to probe, must be in dictionary
  distances = [0,1,1e10]
  best_effs = np.zeros(len(distances))
  best_purs = np.zeros(len(distances))
  bw_a = 1/20
  bw_ad = 1/25
  #cnt distance=1 bar plots for distance shift in left column, distance=2 shifts in right column, next 2 columns are for absolute distance
  cnters = [0,0,0,0] 
  #Pretty colors and markers for types
  for key,contents in matching_dict.items():
    #unpack values
    ne = contents[0]
    tw = contents[1]
    vds = contents[2]
    dist = contents[3]
    pur = contents[4]
    eff = contents[5]
    distance_dict = contents[6]

    for i,distance in enumerate(distances):
      if dist == distance:
        if pur > best_purs[i]:
          best_purs[i] = pur
        if eff > best_effs[i]:
          best_effs[i] = eff
        ticks[i].append(key[:-3])
        axs[i].scatter(key,pur,color='r')
        axs[i].scatter(key,eff,color='b')

    for key_d,contents_d in distance_dict.items():
      label = key_d[5:]
      if dist == 0 or dist == 3:continue #skip uninteresting events
      index = dist-1 #index by distance for counter and axis
      if dist == 1e10: index = 1
      if label == 'none':
        cnt = cnters[index+2]
      else:
        cnt = cnters[index]
      #Make histograms for distance seperations
      heights = contents_d[1] #Get bin heights
      if label == 'none':
        bins = contents_d[0]+bw_ad*cnt-0.17+bw_ad #Offset bin so they're not overlapping, and center it
        axs_ad[index].bar(bins,heights,width=bw_ad,align='edge',label=f'{key[:-3]}')
        #Iterate counter corresponding to correct plot 
        cnt+=1
        cnters[index+2] = cnt
      else:
        if label == 'x':
          alpha = 1
        else:
          alpha = 0.3
        bins = contents_d[0]+bw_a*cnt-0.5+bw_a #Offset bin so they're not overlapping, and center it
        axs_d[index].bar(bins,heights,width=bw_a,align='edge',label=f'{key[:-3]} ({label}-axis)',alpha=alpha)
        #Iterate counter corresponding to correct plot 
        cnt+=1
        cnters[index] = cnt

      


        
  axs[0].set_ylabel('Pur/Eff')
  axs[0].set_title('Match Distance = 0')
  axs[1].set_title('Match Distance = 1')
  axs[2].set_title(r'Match Distance = $\infty$')
  print(cnt)
  #axs[3].set_title(r'Match Distance = $\sqrt{3}$')
  for i,ax in enumerate(axs): 
    ax.set_xticklabels(ticks[i],rotation=40,fontsize=10)
    ax.legend(handles=[pur_label,eff_label],fontsize=16)
    ax.axhline(best_purs[i],color='r',ls='--')
    ax.axhline(best_effs[i],color='b',ls='--')
    ax.set_ylim([0,1])
    plotters.set_style(ax)
  axs_d[0].set_title('Match Distance = 1')
  axs_d[1].set_title(r'Match Distance = $\infty$')
  axs_d[1].legend(bbox_to_anchor=(1.04, 1))
  for i,ax in enumerate(axs_d):
    ax.set_xlabel(r'True-Reco distance from matched voxel')
    #ax.axvline(-0.5,ls='--',color='red')
    #ax.axvline(0.5,ls='--',color='red')
    #ax.axvline(-1.5,ls='--',color='red')
    #ax.axvline(1.5,ls='--',color='red')
    plotters.set_style(ax)
  axs_ad[0].set_title('Match Distance = 1')
  axs_ad[1].set_title(r'Match Distance = $\infty$')
  axs_ad[1].legend(bbox_to_anchor=(1.04, 1))
  for i,ax in enumerate(axs_ad):
    ax.set_xlabel('True-Reco modulus distance from matched voxel')
    #ax.axvline(-0.5,ls='--',color='red')
    #ax.axvline(0.5,ls='--',color='red')
    #ax.axvline(-1.5,ls='--',color='red')
    #ax.axvline(1.5,ls='--',color='red')
    plotters.set_style(ax)

  plotters.save_plot(f'pur_eff_corr_inf',fig=fig)
  plotters.save_plot(f'dist_offset_inf',fig=fig_d)
  plotters.save_plot(f'mod_dist_offset_inf',fig=fig_ad)

def plot_pur_eff_d(purs,effs,ds,**kwargs):
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(ds,purs,color='red',label='Purity', **kwargs)
  ax.plot(ds,effs,color='blue',label='Efficiency',**kwargs)
  ax.plot(ds,purs*effs,color='green',label=r'$P\times E$',**kwargs)
  ax.set_xlabel(r'$Max(|\vec{x}_{true}-\vec{x}_{reco}|)^2$ (voxels$^2$)')
  ax.set_ylabel('Pur/Eff')
  ax.legend()
  
  return fig,ax

def plot_pur_eff_hyperparams(purs,effs,nes,tws,vds,pat=None,**kwargs):
  """
  Plot pur, eff vs hyperparams
  """
  x = np.arange(len(purs)) #Get x values
  tick_labels = [''] #Store tick labels
  print(len(nes),len(purs))
  for i,ne in enumerate(nes):
    if pat is None:
      tick = fr'{ne} $N_e$ , {tws[i]} TW , {vds[i]} vds'
    else: 
      tick = fr'{ne} $N_e$ , {tws[i]} TW , {vds[i]} vds, {pat[i]/1e3:.3f} pa'
    tick_labels.append(tick)
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x,purs,color='red',label='Purity', **kwargs)
  ax.plot(x,effs,color='blue',label='Efficiency',**kwargs)
  ax.plot(x,effs*purs,color='green',label=r'P$\times$E',**kwargs)
  ax.axvline(np.argmax(effs*purs),ls='--',color='black')
  ax.set_xticks(x)
  ax.set_xticklabels(tick_labels[1:],rotation=90)
  ax.legend()
  ax.set_ylabel('Pur/Eff')
  
  return fig,ax
    
def make_cfgs(fnames,cfg):
  """
  Make configuration strings from config file
  """
  cfgs = []
  for _,fname in enumerate(fnames):
    cfg_copy = cfg #Make dumby copy
    cfgs.append(cfg_copy.replace('LARCVFILE',fname))
  return cfgs

def load_cfg(cfg):
  """
  return dataset just from cfg
  """
  cfg=yaml.load(cfg,Loader=yaml.Loader)
  # pre-process configuration (checks + certain non-specified default settings)
  process_config(cfg)
  # prepare function configures necessary "handlers"
  hs=prepare(cfg)
  data = next(hs.data_io_iter)
  return data

def load_data_fname(fname,cfg):
  """
  return dataset just from fname
  """
  cfg = make_cfgs([fname],cfg)[0]
  data = load_cfg(cfg)
  return data
  

def make_matching_dict(fnames,distances,cfg,entry=0):
  """
  Get metrics from all loaded fnames. Also input distances for matching threshold
  """
  cfgs = make_cfgs(fnames,cfg)
  purs = np.zeros((len(fnames),12))
  effs = purs.copy()
  axes = [None,0,1,2]
  ax_str = ['none','x','y','z']
  matching_dict = {}
  for i,fname in enumerate(fnames):
    hyperparams = fname[6:-5].split("_")
    #print(hyperparams)
    ne = int(hyperparams[0]) #number of electrons
    tw = int(hyperparams[1]) #tick window
    vds = int(hyperparams[2]) #voxel distance seperation
    pa = int(hyperparams[3]) #voxel distance seperation
    
    #Get data
    data = load_cfg(cfgs[i])
    clust_label = data['cluster_label'][data['cluster_label'][:, 0] == entry]
    input_data = data['input_data'][data['input_data'][:, 0] == entry]
    segment_label = data['segment_label'][data['segment_label'][:, 0] == entry,-1]
    particles_label = data['particles_label'][data['particles_label'][:, 0] == entry]
    simenergydeposit_label = data['simenergydeposits'][data['simenergydeposits'][:, 0] == entry]
    
    reco = input_data[segment_label<5,1:4]
    true = simenergydeposit_label[:,1:4]
    max_matches = get_max_matches(reco,true)
    
    #Iterate through axes and distances to calculate separations along various axes
    for _,dist in enumerate(distances):
      key = f'ne{ne}_tw{tw}_vds{vds}_pa{pa}_d{dist}'
      pur,eff,tagged,true_indeces,reco_indeces = purity_efficiency(reco,true,distance=np.sqrt(dist),
                                                                          max_matches=max_matches)
      #matching_dict[key] = [pur,eff]
      distance_dict = {}
      for j,axis in enumerate(axes):
        if axis is None:
          separations,counts = np.unique(np.linalg.norm(true[true_indeces]-tagged,axis=1),return_counts=True)
        else:
          separations,counts = np.unique(true[true_indeces,axis]-tagged[:,axis],return_counts=True)
        distance_dict[f'axis_{ax_str[j]}'] = [separations,counts]
      matching_dict[key] = [ne,tw,vds,pa,dist,pur,eff,distance_dict]
  return matching_dict
    
    
    

def project_points_onto_line(points, line_start, line_end):
  #https://en.wikipedia.org/wiki/Vector_projection
  # Calculate the direction vector of the line
  line_direction = (line_end - line_start)/np.linalg.norm(line_end - line_start)

  # Calculate the projection of each point onto the line
  projections = np.dot(points - line_start, line_direction)
  projections = np.expand_dims(projections, axis=1)

  # Calculate the new position of each point
  closest_points = line_start + projections * line_direction

  return closest_points
  

def gap_length_calc(clust_label,particles_label,pid,norm_to_track_length=True,
                    method=0):
  """
  return gap length using clust label and matching to the pid
  """
  mask = clust_label[:,clust_pid_index] == pid
  mask_particle = particles_label[:,particle_pid_index] == pid
  line = particles_label[mask_particle,1:4]
  direction = (line[0]-line[1])/np.linalg.norm(line[0] - line[1])
  points = clust_label[mask,1:4]
  gap = 0
  gamma = 1/np.sqrt(np.max(abs(direction)))
  #We use gamma since we expect the difference in the distance on average to be equal to the direction along the track direction
  if method == 0: #Gap sum method
    for i,coord in enumerate(points):
      if i == len(points)-1: break #skip last and 2nd to last point to avoid out of bounds
      di = np.linalg.norm(coord-points[i+1])
      #print(di,gamma,coord,points[i+1])
      if di == 0: continue #skip points on top of each other
      gap+=di-gamma
  elif method == 1: #Projection method
    projected_points = project_points_onto_line(points,line[0],line[1])
    di = np.diff(projected_points,axis=0)
    gap = np.sum(np.linalg.norm(di,axis=1)-gamma)
    
  #print(gap/len(points)-2,np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)),gamma*(len(points)-2))
  if norm_to_track_length:
    gap = gap/track_length_calc(line[0],line[1])
  return gap

def order_clusters(points, eps=1.1, min_samples=1):
  """
  Order clusters along the principal axis of the track.

  Args:
  points (numpy.ndarray): array of 3D points

  Returns:
  numpy.ndarray: array of ordered cluster indices
  """
  #DBSCAN with chebyshev clustering
  clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='chebyshev').fit(points)
  labels = clustering.labels_
  unique_labels = np.unique(labels)
  centroids = np.array([points[labels == label].mean(axis=0) for label in unique_labels if label != -1])

  #PCA
  pca = PCA(n_components=2)
  pca.fit(points)
  projected_centroids = pca.transform(centroids)
  sorted_indices = np.argsort(projected_centroids[:, 0])

  return unique_labels[sorted_indices],labels

def inter_cluster_distance(points, labels, cluster_indices):
  """
  Compute inter-cluster distance between consecutive clusters.

  Args:
  points (numpy.ndarray): array of 3D points
  labels (numpy.ndarray): array of cluster labels
  cluster_indices (numpy.ndarray): array of ordered cluster indices

  Returns:
  List[float]: list of inter-cluster distances
  """
  distances = []

  for i in range(1, len(cluster_indices)):
    curr_cluster = points[labels == cluster_indices[i]]
    prev_cluster = points[labels == cluster_indices[i - 1]]
    #min_dist = np.min([np.linalg.norm(curr - prev, ord=np.inf) for curr in curr_cluster for prev in prev_cluster])
    min_dist = np.min(cdist(curr_cluster,prev_cluster))
    distances.append(min_dist)

  return distances

def gap_length_calc_cheb(clust_label,particles_label,pid,norm_to_track_length=True,
                         eps=1.1, min_samples=1):
  """
  return gap length using clust label and matching to the pid
  """
  #Get particle points
  mask = clust_label[:,clust_pid_index] == pid
  mask_particle = particles_label[:,particle_pid_index] == pid
  line = particles_label[mask_particle,1:4]
  points = clust_label[mask,1:4]
  
  #Get direction information
  direction = (line[0]-line[1])/np.linalg.norm(line[0] - line[1])
  gamma = 1/np.max(abs(direction))
  
  #Get cluster distances
  cluster_indices,labels = order_clusters(points)
  distances = inter_cluster_distance(points,labels,cluster_indices)
  g = np.sum(distances-gamma)
  if norm_to_track_length:
    g /= track_length_calc(line[0],line[1])
  return g
  
  

def track_length_calc(line_start,line_end):
  #print(line_start,line_end)
  return np.linalg.norm(line_start-line_end)
def match_input_cluster(clust_label,input_data,drop_duplicates=False,distance_threshold=0):
  """
  match cluster_label to input data using voxel coordinates
  distance_threshold matches voxels within threshold
  return input_data that matches clust_label
  """
  tree = cKDTree(input_data[:,1:4]) #Get tree to perform query
  #Return indeces that are on top of the point
  distances, indeces = tree.query(clust_label[:,1:4],k=1)
  ret_ind = []
  for d,i in zip(distances,indeces):
    if d <= distance_threshold:
      ret_ind.append(i)
  input_data = input_data[ret_ind]
  if drop_duplicates: #--- to implement
    #Match voxel coordinates
    #arr = np.sort(input_data[:,1:4])
    arr = input_data[:,1:4]
    # Get the unique rows and their indices in the original array
    unique_rows, indices = np.unique(arr, axis=0, return_inverse=True)

    # Find the indices of the rows that contain the same elements
    drop_ind = np.nonzero(np.bincount(indices) > 1)[0]

    #Drop data
    #print(len(input_data))
    input_data = np.delete(input_data,drop_ind,axis=0)
    #print(len(input_data))
  return input_data



def dE_dx_calc(clust_label,particles_label,energy_label,pid,
               eng_label_index=energy_label_index,distance_threshold=0,
               drop_duplicates=False):
  """
  Calculates dedx by matching energy deposited to cluster label by truth pid, then
  returning the total energy divided by the track length

  match to distance_threshold
  drop_duplicates drops voxels that fall on the same point
  """
  mask = clust_label[:,clust_pid_index] == pid
  mask_particle = particles_label[:,particle_pid_index] == pid
  energy_label_mask = match_input_cluster(clust_label[mask],
                                          energy_label,
                                          distance_threshold=distance_threshold,
                                          drop_duplicates=drop_duplicates)
  line = particles_label[mask_particle,1:4]
  track_length = track_length_calc(line[0],line[1])
  dE = np.sum(energy_label_mask[:,eng_label_index])
  #print(dE,track_length)
  return dE/track_length

def dQ_dx_calc(clust_label,particles_label,input_data,pid,distance_threshold=0,
               drop_duplicates=False):
  """
  Calculates dqdx by matching charge deposited to cluster label by truth pid, then
  returning the total energy divided by the track length
  """
  mask = clust_label[:,clust_pid_index] == pid
  mask_particle = particles_label[:,particle_pid_index] == pid
  input_data_mask = match_input_cluster(clust_label[mask],input_data,distance_threshold=distance_threshold,
                                          drop_duplicates=drop_duplicates)
  line = particles_label[mask_particle,1:4]
  track_length = track_length_calc(line[0],line[1])
  dQ = np.sum(input_data_mask[:,input_planecharge_indeces])/3 #divide by three to take average charge dep.
  
  return dQ/track_length
  
def dN_dx_calc(clust_label,particles_label,input_data,pid,distance_threshold=0,
               drop_duplicates=False):
  """
  Calculates dedx by matching energy deposited to cluster label by truth pid, then
  returning the total energy divided by the track length
  """
  mask = clust_label[:,6] == pid
  mask_particle = particles_label[:,5] == pid
  input_data_mask = match_input_cluster(clust_label[mask],input_data,distance_threshold=distance_threshold,
                                          drop_duplicates=drop_duplicates)
  line = particles_label[mask_particle,1:4]
  track_length = track_length_calc(line[0],line[1])
  dN = len(input_data_mask)
  
  return dN/track_length

