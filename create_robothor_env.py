from ai2thor.controller import Controller
from pprint import pprint
from PIL import Image 
import numpy as np
import pickle
from transforms3d import euler, affines

fov_w = 80.0
width = 224 * 4
height = 224 * 4

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
while True :
    controller.step("Done")
    choice= input()

    for obj in controller.last_event.metadata["objects"]:
        print(obj["name"])



    if choice == "w":
        event= controller.step("MoveAhead")
    if choice == "s":
        event= controller.step("MoveBack")
    if choice == "a":
        event= controller.step("MoveLeft")
    if choice == "d":
        event= controller.step("MoveRight")
    if choice == "r":
        event= controller.step("RotateRight")
    if choice == "e":
        event= controller.step("RotateLeft")

    if choice == "t":
        event= controller.step(    
            action="LookUp",
            degrees=30
            )
    if choice == "g":
        event= controller.step(    
            action="LookDown",
            degrees=30
            )

    if choice == "o":
        event= controller.step(
            action='SetObjectPoses',
            objectPoses=[
            {
                "objectName": "Potato_8a72b89b",
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": 0
                },
                "position": {
                    "x": 1.2,
                    "y": 1.87,
                    "z": -1.4
                }
            },
            {
                "objectName": "Fork_29878d10",
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": 0
                },
                "position": {
                    "x": 0.8,
                    "y": 1.87,
                    "z": -1.4
                }
            },
            {
                "objectName": "DishSponge_a3f8f753",
                "rotation": {
                    "y": 0,
                    "x": 0,
                    "z": 0
                },
                "position": {
                    "x": 0.3,
                    "y": 1.67,
                    "z": -1.7
                }
            }
            ]
        )


    if choice == "camera":
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
        
        output = open('assets/data.pkl','wb')
        pickle.dump(data,output)
        output.close()

    if choice == "close cabinets":
        #cabinet 1
        controller.step(action="CloseObject",
                                objectId="Cabinet|+01.40|+01.87|-01.26", 
                                forceAction=False)
        
        #cabinet 8
        event = controller.step(action="CloseObject",
                                objectId="Cabinet|+00.62|+01.87|-01.26", 
                                forceAction=False)
        #cabinet 9
        event = controller.step(action="CloseObject",
                                objectId="Cabinet|+00.14|+01.67|-01.56", 
                                forceAction=False)
    if choice == "open cabinets":
        #cabinet 1
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+01.40|+01.87|-01.26", 
                                openness=1,
                                forceAction=False)
        
        #cabinet 8
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+00.62|+01.87|-01.26", 
                                openness=1,
                                forceAction=False)
        #cabinet 9
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+00.14|+01.67|-01.56", 
                                openness=1,
                                forceAction=False)


    if choice == "1":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+01.40|+01.87|-01.26", 
                                openness=1,
                                forceAction=False)
    if choice == "2":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|-00.20|+01.96|-01.33", 
                                openness=1,
                                forceAction=False)
    if choice == "3":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+02.82|+01.77|-01.85", 
                                openness=1,
                                forceAction=False)
    if choice == "4":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+02.85|+00.42|+00.41", 
                                openness=1,
                                forceAction=False)
    if choice == "5":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+02.85|+00.42|-00.61", 
                                openness=1,
                                forceAction=False)
    if choice == "6":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+03.07|+01.67|-00.71", 
                                openness=1,
                                forceAction=False)
    if choice == "7":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+02.82|+01.77|-01.05", 
                                openness=1,
                                forceAction=False)
    if choice == "8":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+00.62|+01.87|-01.26", 
                                openness=1,
                                forceAction=False)
    if choice == "9":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|+00.14|+01.67|-01.56", 
                                openness=1,
                                forceAction=False)
    if choice == "10":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|-00.19|+01.67|-01.34", 
                                openness=1,
                                forceAction=False)
    if choice == "11":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|-00.71|+01.96|-00.82", 
                                openness=1,
                                forceAction=False)
    if choice == "12":
        event = controller.step(action="OpenObject",
                                objectId="Cabinet|-00.92|+01.67|-00.62", 
                                openness=1,
                                forceAction=False)
