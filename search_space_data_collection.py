from ai2thor.controller import Controller
from pprint import pprint
from PIL import Image 
import numpy as np
import pickle
from transforms3d import euler, affines
import threading

fov_w = 80.0
width = 224 * 4
height = 224 * 4
obj_name = "Tomato_5f4bf236"

def create_training_example(i):
        controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan30",

            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,

            # image modalities
            renderDepthImage=True,
            renderInstanceSegmentation=False,

            # camera properties
            width=width,
            height=height,
            fieldOfView=fov_w
        )

        event= None

        elements  = [0,1,2]
        probabilities = [1/3,1/3,1/3]


        placement = np.random.choice(elements, p=probabilities)
        epsilon_noise_x = np.random.normal(loc=0.0,scale=0.05)
        epsilon_noise_y = np.random.normal(loc=0.0,scale=0.05)
        epsilon_noise_z = np.random.normal(loc=0.0,scale=0.05)

        if(placement==0):
            event= controller.step(
                        action='SetObjectPoses',
                        objectPoses=[
                        {
                            "objectName": obj_name,
                            "rotation": {
                                "x": 0,
                                "y": 0,
                                "z": 0
                            },
                            "position": {
                                "x": 1.2+epsilon_noise_x,
                                "y": 1.87+epsilon_noise_y,
                                "z": -1.4+epsilon_noise_z
                            }
                        }])
        elif(placement==1):
            event = controller.step(
                        action='SetObjectPoses',
                        objectPoses=[
                        {
                            "objectName": obj_name,
                            "rotation": {
                                "x": 0,
                                "y": 0,
                                "z": 0
                            },
                            "position": {
                                "x": 0.8+epsilon_noise_x,
                                "y": 1.87+epsilon_noise_y,
                                "z": -1.4+epsilon_noise_z
                            }
                        }])
        else:
            event = controller.step(
                        action='SetObjectPoses',
                        objectPoses=[
                        {
                            "objectName": obj_name,
                            "rotation": {
                                "x": 0,
                                "y": 0,
                                "z": 0
                            },
                            "position": {
                                "x": 0.3+epsilon_noise_x,
                                "y": 1.67+epsilon_noise_y,
                                "z": -1.7+epsilon_noise_z
                            }
                        }
                        ]
                )

        event= controller.step("RotateRight")
        event= controller.step("MoveLeft")
        event= controller.step("MoveLeft")
        event = controller.step(action="OpenObject",
                                        objectId="Cabinet|+01.40|+01.87|-01.26", 
                                        openness=1,
                                        forceAction=False)

        event = controller.step(action="OpenObject",
                                        objectId="Cabinet|+00.62|+01.87|-01.26", 
                                        openness=1,
                                        forceAction=False)

        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+00.14|+01.67|-01.56", 
                                openness=1,
                                forceAction=False)

        focal_length = (width / 2) / np.tan((np.pi * fov_w / 180) / 2)
                
        cam_intr = np.array(
            [[focal_length, 0, height / 2], [0, focal_length, width / 2], [0, 0, 1]]
        )

        cam_pose = affines.compose(
            T=list(event.metadata["agent"]["position"].values()),
            R=euler.euler2mat(
                list(event.metadata["agent"]["rotation"].values())[2] * np.pi / 180,
                list(event.metadata["agent"]["rotation"].values())[1] * np.pi / 180,
                list(event.metadata["agent"]["rotation"].values())[0] * np.pi / 180,
            ),
            Z=np.ones(3),
        )

        img1 = Image.fromarray(event.frame)    


        data = {"depth": event.depth_frame,
                "cam_intr":cam_intr,
                "cam_pose":cam_pose,
                "image":img1}

        output = open('assets/cellphone_search_space_optimization/data_testing_cellphone'+str(i)+'.pkl','wb')
        pickle.dump(data,output)
        output.close()
        controller.stop()


if __name__ == "__main__":
    threads = []
    for j in range(0,4):
        for i in range(10):
            t = threading.Thread(target=create_training_example,args=(i+j*10,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

    print("Done!")
