from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import os
import json
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

def read_json(file):
    skeletonDict = json.load(open(file,'r'))
    bodySkeleton = numpy.array(skeletonDict['Body']).squeeze()
    if len(bodySkeleton.shape)>2:
        bodySkeleton = bodySkeleton[0]
    return bodySkeleton[:,:2]

def drawFunc():
    skeleton_folder = '/Users/liweijie/SLR/scripts/01April_2010_Thursday_heute_default-0'
    file_list = os.listdir(skeleton_folder)
    file_list.sort()
    global mat
    mat = []
    for i,file in enumerate(file_list):
        file_path = os.path.join(skeleton_folder,file)
        skeleton = read_json(file_path)
        t = 100*i*numpy.ones([skeleton.shape[0],1])
        skeleton_t = numpy.concatenate([skeleton,t],1)
        mat.append(skeleton_t)
    mat = numpy.concatenate(mat,0)
    mat = mat.astype(numpy.int)
    glClearColor(0.0, 0.0, 0.0,0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
 
    # 设置点大小
    glPointSize(5)
    # 只绘制端点
    glBegin(GL_POINTS)
    for i,skeleton in enumerate(mat):
        x = skeleton[0]/300
        y = 1-skeleton[1]/300
        z = skeleton[2]/20000
        t = i/len(mat)
        glColor3f(0.0,0.0,t)
        glVertex3f(x,y,z)
    glEnd()
 
    glFlush()

 

glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
glutInitWindowSize(400, 400)
glutCreateWindow(b"First")
glutDisplayFunc(drawFunc)
glutMainLoop()
