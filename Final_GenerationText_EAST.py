#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:52:18 2017

@author: cooperjack
"""
#import tensorflow as tf
#tf.
import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from numpy import *

#import pygame
#from pygame.locals import * 

import codecs

import math
#from PIL import imageOps

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import os 
import csv
import glob
import re
import string
#from zhon.hanzo import punctuation



pathToImage = ''
data_dir = './Model/data/'
#pathToFontList = '/Users/cooperjack/Downloads/SynthText_Chinese_version-master/data/fonts/ubuntu/Ubuntu-Bold.ttf'
pathToFontList = os.path.join(data_dir, 'fonts/fontlist.txt')
fontsTable = [os.path.join(data_dir,'fonts',f.strip()) for f in open(pathToFontList)]

def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
        os.path.join(path, '*.{}'.format(ext))))
    return files

def save_annoataionCH(finalPos,text,textPath,textName,times):
    finalPath = textPath
    finalPath += '/'
    finalPath += (textName)
    finalPath += ('.txt')
    
    text_polys = []
    text_tags = []
#    if not os.path.exists(finalPath):
#        os.mknod(finalPath)
      
    with open(finalPath, 'wb') as f:
        for index in range(times):
            for item in range(len(finalPos[index])):
                f.write(str(finalPos[index][item]))
                if item <= len(finalPos[index])-2:
                    f.write(',')
            f.write(',')         
            f.write((text[index]).encode('utf8'))
            f.write('\n')
def save_annoataionEN(finalPos,text,textPath,textName,times):
    finalPath = textPath
    #finalPath += '/'
    finalPath += (textName)
    finalPath += ('.txt')
    
    text_polys = []
    text_tags = []

    with open(finalPath, 'w') as f:
        for index in range(times):
            b = len(finalPos[index])
            for item in range(b):
                f.write(str(int(finalPos[index][item])))
                if item <= b-2:
                    f.write(',')
            f.write(',')         
            f.write(text[index])
            f.write('\n')
def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    with open(p, 'r') as f:
        reader = csv.reader(f) 
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def GenerateFontPosByOpenCV(img):
    im, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("num of contours: {}".format(len(contours)))
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations = 1)  # dilate
    _,contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    mult = 1.2   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
    plt.imshow(img_box)
    plt.show()
    
def GenerateRotatedPnt(origin,angle,pnt):
    tW = pnt[0]-origin[0]
    tH = pnt[1]-origin[1]
    temp = []
    anglePi = angle * math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    temp.append(int((tW*cosA-tH*sinA)+origin[0]) ) 
    temp.append(int((tW*sinA+tH*cosA)+origin[1]) ) 
    pnt[0]=temp[0]
    pnt[1]=temp[1]

def GenerateRotationFontPos(img,size):
    #print "size",size
    tempImg = img;
    input = array(tempImg)

    dilated = input.copy()
    eleSize = max(int(size),1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(eleSize,eleSize))
    for index in range(10):
         dilated = cv2.dilate(array(input), kernel)
       
    _,contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


    if(len(np.nonzero(dilated))==0):
        print 'Dilated image is empty'
    flattened_list =[]
    mult = 1.2   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = cv2.cvtColor(dilated.copy(), cv2.COLOR_GRAY2BGR)
    rect = []
    if len(contours)==0:
        rows,cols = np.nonzero(input)
        minY = min(rows)
        maxY = max(rows)
        minX = min(cols)
        maxX = max(cols)
        
        rect = [minX,minY,maxX-minX,maxY-minY]
        
        flattened_list = [minX, minY, maxX,minY,minX,maxY,maxX,maxY]
    else:
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cen= rect[0]; wh =rect[1];angle= rect[2]
            
            bottomRight=[cen[0]+ 0.5*wh[0],cen[1]+ 0.5*wh[1]];
            rightTop=[cen[0]+ 0.5*wh[0],cen[1]- 0.5*wh[1]];
            leftTop=[cen[0]- 0.5*wh[0],cen[1]- 0.5*wh[1]];
            bottomLeft=[cen[0]- 0.5*wh[0],cen[1]+ 0.5*wh[1]];
            
            GenerateRotatedPnt(cen,angle,bottomRight)
            GenerateRotatedPnt(cen,angle,rightTop)
            GenerateRotatedPnt(cen,angle,leftTop)
            GenerateRotatedPnt(cen,angle,bottomLeft)
            
            rows,cols = np.nonzero(dilated)
            minY = min(rows)
            maxY = max(rows)
            minX = min(cols)
            maxX = max(cols)
            
            rect = [minX,minY,maxX-minX,maxY-minY]
        
            vertice =[]
            vertice.append(leftTop)
            vertice.append(rightTop)
            vertice.append(bottomRight )
            vertice.append(bottomLeft)
            
            flattened_list = [y for x in vertice for y in x]

#    cv2.drawContours(dilated, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
#    cv2.imshow("dilated", dilated)
#    cv2.waitKey(0)
    return flattened_list, rect

def GenerateFontPos(img):
    #print'nonzero', len(np.nonzero(img))
    temp = np.nonzero(img)
    rows = temp[0]
    cols = temp[1]
    #rows,cols,_ = np.nonzero(img)
    minY = min(rows)
    maxY = max(rows)
    minX = min(cols)
    maxX = max(cols)
    
    rect = [minX,minY,maxX-minX,maxY-minY]
    coord = [minX,minY,maxX,minY,maxX,maxY,minX,maxY]    
    return coord,rect

def GenerateRandomGrayValue(image):
    meanGray = np.mean(np.asarray(image))
    tR = 250
    
    grayH = 100; grayL = 10;
    if meanGray > 200 and meanGray<255 :
        grayL = 10; grayH = 100;
    elif meanGray >50 and meanGray <=200: 
        grayL = 220; grayH = 250;
    elif  meanGray <=50: 
        grayL = 200; grayH = 250;
        
    tR = np.random.randint(grayL,grayH)
    return tR
    
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.
def draw_rotated_text(image, angle, text,leftTopPnt,rightBtmPnt, fontSize,byLine, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:
    :param image: Image to write text into
    :param angle: Angle to write text at
    """

    grayL = 50; grayH = 100;
    tR = np.random.randint(grayL,grayH)
    tG = np.random.randint(grayL,grayH)
    tB = np.random.randint(grayL,grayH)
    fontColor = (int(tR),int(tG),int(tB))

    # get the size of our image
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)
    #print'mask info:',np.nonzero(array(mask))[0],np.nonzero(array(mask))[1]

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), unicode(text), 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
                                  mask_size, resample=Image.LANCZOS)
    # crop the mask to match image
    mask_xy = (int(max_dim - leftTopPnt[0]), int(max_dim - leftTopPnt[1]))
    b_box = mask_xy + (int(mask_xy[0] + width), int(mask_xy[1] + height) )
    mask = rotated_mask.crop(b_box)
    #mask.show()
    #print 'b_Box',b_box
    #print 'mask',(mask.size)		
    #print 'mask size', len(np.nonzero(mask))
    rows,cols = np.nonzero(mask)
	
    #print 'mask rows',len(rows)
    #print 'mask cols',len(cols)
    if len(rows)==0 or len(cols)==0:
        print 'nonzero(mask) is not exist:'
        return [],[];

    #eleSize = max(fontSize*0.5,1)#for english
    #if(byLine == 0):
    eleSize = max(fontSize*10,2)
    #cor,rect = GenerateRotationFontPos(mask,int(eleSize))
    cor,rect = GenerateFontPos(mask)
    #else:
        #mask.show()
        #cor,rect = GenerateFontPos(mask)		

    color_image = Image.new('RGBA', image.size, fontColor)
    image.paste(color_image, mask)
    
    #print'text_polys',[[cor[0], cor[1]], [cor[2], cor[3]], [cor[4], cor[5]], [cor[6], cor[7]]]
#    print x1,y1,x2,y2,x3,y3,x4,y4
    text_polys=[]
    polyLine = [(cor[0],cor[1]),(cor[2],cor[3]),(cor[4],cor[5]),(cor[6],cor[7])]
    
    text_polys.append([[cor[0], cor[1]], [cor[2], cor[3]], [cor[4], cor[5]], [cor[6], cor[7]]])
    
    p_area=  polygon_area((polyLine))
    if abs(p_area) < 1:
        print poly
        print('invalid poly')
    dr = ImageDraw.Draw(mask)
#    dr.rectangle(rect,outline='green')
    dr.polygon(polyLine,outline='green')
#    mask.show()

    return cor,rect
def pntIsInRect(pnt,finalTextRect):
    flag = 0
    for item in range(len(finalTextRect)):
        rect = finalTextRect[item]
        if pnt[0]>rect[0]-5 and pnt[0]<rect[0]+rect[2]+10\
            and  pnt[1]>rect[1]-5 and pnt[1]<=rect[1]+rect[3]+10:
            flag =  1
    return flag 
    
def GenerateDrawPos(img,finalTextRect,first,newTextSize,angle):
    [imgW,imgH]= img.size

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0])
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
        
    origin=[]
    origin.append((blkTLX))
    origin.append((blkTLY))
    
    newTextIsOut = 0
    iteration = 0;

    while(newTextIsOut==0 and first == 0):

        if imgW>blkW+80:
            blkTLX = np.random.randint(20,imgW-blkW-50)
        else:
            blkTLX = np.random.randint(10,imgW-blkW)
        if imgH>blkH+80:
            blkTLY = np.random.randint(20,imgH-blkH-50)
        else:
            blkTLY = np.random.randint(10,imgH-blkH)
        origin=[]
        origin.append((blkTLX))
        origin.append((blkTLY))

        bottomRight=[origin[0]+ wh[0],origin[1]+ wh[1]];
        rightTop=[origin[0]+ wh[0],origin[1]];
        leftTop=[origin[0],origin[1]];
        bottomLeft=[origin[0],origin[1]+ wh[1]];
        center = [origin[0]+0.5* wh[0],origin[1]+0.5* wh[1]]
        
        GenerateRotatedPnt(origin,angle,bottomRight)
        GenerateRotatedPnt(origin,angle,rightTop)
        GenerateRotatedPnt(origin,angle,leftTop)
        GenerateRotatedPnt(origin,angle,bottomLeft)
        GenerateRotatedPnt(origin,angle,center)
        
        con1 =  pntIsInRect(bottomRight,finalTextRect)
        con2 =  pntIsInRect(rightTop,finalTextRect)
        con3 =  pntIsInRect(leftTop,finalTextRect)
        con4 =  pntIsInRect(bottomLeft,finalTextRect)
        con5 =  pntIsInRect(center,finalTextRect)
        
        if con1==0 and con2==0 and con3 == 0 and con4 == 0 and con5 == 0:
            newTextIsOut = 1
           # print'pntIsOutRect'
            return origin
        else:
            newTextIsOut = 0
        iteration += 1
        if iteration >100:
            break;

    return origin
def pntIsInRectNew(pnt,finalTextRect):
    flag = 0
    for item in range(len(finalTextRect)):
        rect = finalTextRect[item]
        dist = (pnt[0]-rect[0]-0.5*rect[2])*(pnt[0]-rect[0]-0.5*rect[2])\
                    +(pnt[1]-rect[1]-0.5*rect[3])*(pnt[1]-rect[1]-0.5*rect[3])
        dist = np.sqrt(dist)
        thresh = min(rect[2],rect[3])*0.5
        thresh = max(thresh-5,5)
        if dist < thresh:
            flag =  1
    return flag 
def GenerateDrawPosNew(img,finalTextRect,first,newTextSize,angle):
    [imgW,imgH]= img.size

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0])
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
        
    origin=[]
    origin.append((blkTLX))
    origin.append((blkTLY))
    
    newTextIsOut = 0
    iteration = 0;

    while(newTextIsOut==0 and first == 0):

        if imgW>blkW+80:
            blkTLX = np.random.randint(20,imgW-blkW-50)
        else:
            blkTLX = np.random.randint(10,imgW-blkW)
        if imgH>blkH+80:
            blkTLY = np.random.randint(20,imgH-blkH-50)
        else:
            blkTLY = np.random.randint(10,imgH-blkH)
        origin=[]
        origin.append((blkTLX))
        origin.append((blkTLY))
        
        topX = list(np.linspace(origin[0],(origin[0]+wh[0]),3))
        topY = list(np.linspace(origin[1],(origin[1]),3))
        
        leftX = list(np.linspace(origin[0],(origin[0]),3))
        leftY = list(np.linspace(origin[1],(origin[1]+wh[1]),3))
        
        rightX = list(np.linspace((origin[0]+wh[0]),(origin[0]+wh[0]),3))
        rightY = list(np.linspace(origin[1],(origin[1]+wh[1]),3))
        
        bottomX = list(np.linspace((origin[0]),(origin[0]+wh[0]),3))
        bottomY = list(np.linspace((origin[1]+wh[1]),(origin[1]+wh[1]),3))
        
        coordX = []; coordY = [];
        coordX.append(topX); coordX.append(leftX); coordX.append(rightX);coordX.append(bottomX);
        coordY.append(topY); coordY.append(leftY); coordY.append(rightY);coordY.append(bottomY);
        
        flattened_listX = [y for x in coordX for y in x]
        flattened_listY = [y for x in coordY for y in x]
        
        rightVote = 0
        for pntX in range(len(flattened_listX)):
            for pntY in range(len(flattened_listY)):
                pnt=[]
                pnt.append(flattened_listX[pntX])
                pnt.append(flattened_listY[pntY])
                GenerateRotatedPnt(origin,angle,pnt)
                
                if(pntIsInRectNew(pnt,finalTextRect) == 0):
                    rightVote+=1
                if rightVote == 36:
                    return origin
        iteration += 1
        if iteration >100:
            break;
    return origin
def GenerateDrawPosOnText(img,txt,pathToFont,row,col,numR,numC):
    
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/numC))
    blockH = int(ceil(imgH/numR))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    img_fraction = 0.8
#    print 'font,getSize',font.getsize(txt)
#    print 'threshold',img_fraction*blockW,img_fraction*blockH
#    print 'Font.get',font.getmask(txt).getbbox()
    while (font.getsize(txt)[0] < img_fraction*blockW-10 and font.getsize(txt)[1] < img_fraction*blockH-10 ):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
#              print 'font,getSize',font.getsize(txt)
#              print 'threshold',img_fraction*blockW,img_fraction*blockH
#              print 'Font.get',font.getmask(txt).getbbox()
              
              
    fontSize -= 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)
    
    origin=[]
    origin.append(int(cenX-font.getsize(txt)[0]*0.5+5))
    origin.append(int(cenY-font.getsize(txt)[1]*0.5+5))
    
    #print'fontSize', fontSize
    return origin,fontSize;

def GenerateDrawPosByBlocks(img,txt,pathToFont,row,col,splitNum):
    
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/splitNum))
    blockH = int(ceil(imgH/splitNum))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    img_fraction = 0.8
#    print 'font,getSize',font.getsize(txt)
#    print 'threshold',img_fraction*blockW,img_fraction*blockH
#    print 'Font.get',font.getmask(txt).getbbox()
    while (font.getsize(txt)[0] < img_fraction*blockW-10 and font.getsize(txt)[1] < img_fraction*blockH-10 ):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
#              print 'font,getSize',font.getsize(txt)
#              print 'threshold',img_fraction*blockW,img_fraction*blockH
#              print 'Font.get',font.getmask(txt).getbbox()
              
              
    fontSize -= 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)
    
    origin=[]
    origin.append(int(cenX-font.getsize(txt)[0]*0.5+5))
    origin.append(int(cenY-font.getsize(txt)[1]*0.5+5))
    
    #print'fontSize', fontSize
    return origin,fontSize;

def GenerateDrawPosOnNextLine(img,finalTextRect,first,newTextSize,angle):
    
    [imgW,imgH]= img.size
    origin=[]

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0])
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,int(0.5*imgW))
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,int(0.5*imgH))
        
    origin=[]
    
    if first == 1:
        origin.append((blkTLX))
        origin.append((blkTLY))
    else:
        origin.append((finalTextRect[0]))
        origin.append((finalTextRect[1]+finalTextRect[3]*1.1+8))

    return origin

def GenerateDrawPosbyLines(img,stringLen,rect, first):
    [imgW,imgH]= img.size
    
    ## generate the text region 
    ratioMax = 0.1;
    ratioMin = 0.01;
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(imgW,imgH)
    fontSize = int(curRatio*minSize)
    
    blkW = fontSize*stringLen;
    blkH = int(fontSize*1.1)
    
    blkTLX = 30
    blkTLY = 30
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
    
    # Generate the text four coordinate
    origin=[]
    if first == 1:
        origin.append(blkTLX)
        origin.append(blkTLY)
    else:
        origin.append(rect[0])
        origin.append(rect[5])
    return origin,fontSize

## Generate the string to render
def GenerateStringCH(filename,stringNum):
    f = codecs.open(filename, 'rb', encoding="utf8")
    data = f.read()##.decode("gbk").encode("utf-8")

    callNumber = np.arange(len(data))
    np.random.shuffle(callNumber)
    pune = '!"#$%&\'()*+,  -./:;<=>?@[\\]^_`{|}~"'
    pune = pune.decode("utf-8")
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／： ；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc = punc.decode("utf-8")
    chooseString = u''
    index = 0
    strLen = 0
    while strLen <stringNum:
        strLen = len(chooseString)
        character = data[callNumber[index]]
        d=re.findall(u'[\u4e00-\u9fa5_a-zA-Z0-9]',character)
        if d!='\s'and d!='\n'and d!= '0':
            chooseString += ''.join(d)
            index += 1
    #print chooseString
    return chooseString

def GenerateStringEN(path,stringNum):
    
    try:
        fileOpen = open(path,'r')
    except IOError:
        print 'Open text failed!'
        
    r='^[A-Za-z0-9]+$'
    chooseString = '';
    allContent = fileOpen.readlines()
    
    srcLen = len(allContent)
    test  =len(set(chooseString))
    index = 0
    while index <stringNum-1:
            s = np.random.randint(0,srcLen-1)
            tempLen = len(allContent[s])
            if tempLen == 1:
                continue;
            tempIndex = np.random.randint(0,tempLen-1)
            d=re.findall(u'[\u4e00-\u9fa5_a-zA-Z0-9]',allContent[s][tempIndex])
            if d !='\t'and d != '\n':
                chooseString +=''.join(d)
                index += 1
    #print 'length', len(set(chooseString))
    xx = chooseString.replace('\n','')
    yy = xx.replace('\\','')
    yy = yy.replace('','')
    
    return yy


def GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,byLine,splitNum):
    files = get_images(dataSetPath)
    ## for circle 
    index = 1+startIndex;
    viz = 0
    tempS = ['']
    maxNum = min(len(files),dataSetNum)
    print 'maxNum',maxNum

    indexlist   = range(len(files))
    np.random.shuffle(indexlist)
    
    #splitNum = 3
    writeTimes = splitNum*splitNum
    for item in (indexlist):
        img=Image.open(files[item])
        [imgW,imgH]= img.size
        if imgW<=0 or imgH <=0:
            print'invalid image input!', files[item]
            continue;
        
        tR = np.random.randint(200,250)
        emptyImg = Image.new('L', img.size, int(tR))
        finalCoord = []
        finalText = []
        newTextSize=[];
        curAngle = 0;
        textRect = [(0,0)]
        finalTextRect= []
        
        maxAngle = 30
        curAngle = np.random.randint(-maxAngle,maxAngle)

        first = 1
        stringNum = np.random.randint(8,50)
        for times in range(writeTimes):
            
            row= int(floor(times/splitNum))
            col = int(times%splitNum)
            #print row,col
            
            if english == 1:
                chooseString = GenerateStringEN(novelPath,stringNum)
            else:
                chooseString = GenerateStringCH(novelPath,stringNum)
            
            pathToFont = fontsTable[np.random.randint(0,len(fontsTable)-1)]
            fontSize=10
            font = ImageFont.truetype(pathToFont, fontSize)
            #for row in range(splitNum):
                #for col in range(splitNum):
            #origin,fontSize = GenerateDrawPosByBlocks(emptyImg,chooseString,pathToFont,row,col,splitNum)
            origin,fontSize = GenerateDrawPosOnText(emptyImg,chooseString,pathToFont,times,0,writeTimes,1)
    
            font = ImageFont.truetype(pathToFont, fontSize)
            
            newTextSize = font.getsize(chooseString)
            #print 'newTextSize',newTextSize
            
            if (byLine) !=1:
                curAngle = np.random.randint(-1,1)

            pnt=[]
            pnt.append(origin[0]+newTextSize[0])
            pnt.append(origin[1]+newTextSize[1])
            
            GenerateRotatedPnt(origin,curAngle,pnt)
            #cor,textRect = draw_rotated_text(emptyImg, curAngle, chooseString, origin,pnt,fontSize, font=font)
            cor,textRect = draw_rotated_text(emptyImg, curAngle, chooseString, origin,pnt,byLine,fontSize, font=font)
            if cor == [] or textRect ==[]:
                continue;

            
            finalTextRect.append(textRect);
            finalCoord.append(cor)
            finalText.append(chooseString)
            
            if first == 1:
                first = 0
            
        textPath = outputPath
        textName = 'img_'+str(index)

        print "Num", index
        if english == 1:
            save_annoataionEN(finalCoord,finalText,textPath,textName,writeTimes)
        else:
            save_annoataionCH(finalCoord,finalText,textPath,textName,writeTimes)
        if viz==1:
            emptyImg.show()
        
        ## Generate the Position, size, angle of the string
        outputImgPath = outputPath 
        outputImgPath += 'img_'+str(index)
        outputImgPath += '.jpg'
        emptyImg.save(outputImgPath)
        
        index = index+1
        if index-startIndex>maxNum+1:
            break;  
dataSetPath = './training_samples/'
outputPath = './test/'

pathToCHText = './chinese.txt'#Test
pathToENText = './AliceWonderland.txt'



novelPath = pathToENText
dataSetNum = 5
english = 1

splitNum = 2
startIndex = 5
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)

print"Session 1 finished!"

splitNum = 3
#startIndex = startIndex+dataSetNum
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
print"Session 2 finished!"

splitNum = 2
#startIndex = startIndex+dataSetNum
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,0,splitNum)
print"Session 3 finished!"

#startIndex = startIndex+dataSetNum
splitNum = 3
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,0,splitNum)
print"Session 4 finished!"


### Chinese
novelPath = pathToCHText
#startIndex = startIndex+dataSetNum
english = 0

#startIndex = startIndex+dataSetNum
splitNum = 2
GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
print"Session 5 finished!"

startIndex = startIndex+dataSetNum
splitNum = 3
GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
print"Session 6 finished!"

startIndex = startIndex+dataSetNum
splitNum = 2
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,0,splitNum)

print"Session 7 finished!"
startIndex = startIndex+dataSetNum
splitNum = 3
GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,0,splitNum)
print"Session 8 finished!"








