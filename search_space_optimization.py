import concatenated_point_cloud
import importlib
import pickle 
from sklearn.mixture import GaussianMixture 


importlib.reload(concatenated_point_cloud)



def provide_location_data(labels, prompts, filename):
    file_handle = open(filename, 'rb')
    data = pickle.load(file_handle)
    
    cam_intr = data["cam_intr"]
    cam_pose = data["cam_pose"]
    depth_img = data["depth"]
    image = data["image"]
    
    relevancy_pts_list = concatenated_point_cloud.return_relevancy_pts_list(image, labels, prompts)
    cam_pts = concatenated_point_cloud.return_cam_pts(depth_img,cam_intr, cam_pose)

    assert len(relevancy_pts_list) == len(labels) and len(labels) == 1
    
    relevancy_pts = relevancy_pts_list[0]
    obj = labels[0]
    
    loc = concatenated_point_cloud.return_highest_relevancy_pt(cam_pts, relevancy_pts)
    
    #reversing transformations due to RoboTHOR left hand coordinate system
    
    tmp = loc[2]
    loc[2] = loc[1]
    loc[1] = tmp
    loc[1] = -loc[1]
    loc[1] += 1.2
    
    

    return loc




def determine_optimal_GMM(location_data, max_components):
    GMMs = []
    BICs = []
    AICs = []
    
    

    for n_components in range(1,max_components):
        gm = GaussianMixture(n_components).fit(location_data)
        GMMs.append(gm)
        AICs.append(gm.aic(location_data))
        BICs.append(gm.bic(location_data))
    
    return GMMs
    

        
    
    
    
    
    














