import concatenated_point_cloud
import search_space_optimization
import pickle
import importlib
import numpy as np

importlib.reload(concatenated_point_cloud)
importlib.reload(search_space_optimization)

MAX_NUM_TRAINING_EXAMPLES = 500
NUM_EXAMPLES = [10,20,50,100,200,400,500]

location_data = []

for i in range(MAX_NUM_TRAINING_EXAMPLES):
    labels = ["cellphone"]
    prompts = ["{}"]
    
    filename = 'data/cellphone_search_space/data_testing_cellphone'+str(i)+'.pkl'
    
    obj_location = search_space_optimization.provide_location_data(labels, prompts, filename)
    
    location_data.append(obj_location)
    
location_data = np.array(location_data)    

location_data = np.random.shuffle(location_data)


training_data = {}

for examples in NUM_EXAMPLES:
    min_location = location[0:NUM_EXAMPLES]
    training_data[str(NUM_EXAMPLES)] = min_location

output = open('training_data/location_data.pkl','wb')
pickle.dump(data, output)
output.close()