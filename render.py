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
    color = mesh.visual.vertex_colors
    
    # write x y coordinates on top of the input_img in the given color, note that the given color is 4D
    for i in range(len(x)):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (int(color[i][0]*255), int(color[i][1]*255), int(color[i][2]*255)), -1)
    
    # store the img to output_path
    cv2.imwrite(output_path, img)
    
    return

if __name__ == '__main__':
    
    # example usage:
    # python3 render.py Ole1
    identifier = sys.argv[1]

    input_img_path = f'datasets/{identifier}_with_landmarks.jpg' 
    mesh_path = f'output/{identifier}.obj'
    output_path = f'output/{identifier}_rendered_landmarks.jpg'

    render_mesh(input_img_path=input_img_path, 
                    mesh_path=mesh_path, 
                    output_path=output_path
                    ) 

