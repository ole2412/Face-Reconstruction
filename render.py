import pyrender
import trimesh
import numpy as np
import cv2
import sys


def render_mesh(input_img_path, mesh_path, output_path):
    # load the input_img
    img = cv2.imread(input_img_path)
    
    # take x y coordinates of mesh
    mesh = trimesh.load(mesh_path)
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    
    # write x y coordinates on top of the input_img in color red
    for i in range(len(x)):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 3, (0, 0, 255), 10)
    
    # store the img to output_path
    cv2.imwrite(output_path, img)
    
    return

if __name__ == '__main__':
    
    # example usage:
    # python3 render.py datasets/Ole1_with_landmarks.jpg output/Ole1.obj output/Ole1_rendered_landmarks.jpg

    input_img_path = sys.argv[1]
    mesh_path = sys.argv[2]
    output_path = sys.argv[3]

    render_mesh(input_img_path=input_img_path, 
                    mesh_path=mesh_path, 
                    output_path=output_path
                    ) 

