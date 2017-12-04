
#coding=utf-8

import random
import math
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import pdb
import codecs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import glob
from hyperparams import Hyperparams as hp
import os
import re
font_file_list = glob.glob("fonts/*.*")
#font_file_list = ["fonts/simsun.ttc"]


content_list = []
text_file = codecs.open("output.txt","r","utf-8")
text_content = text_file.readlines()
for content in text_content:
  content_list.append(content.strip())

one_hot_len = len(hp.string_list)
max_len_scale = hp.max_len


def gene_line(draw,width,height):
  begin = (random.randint(0, width), random.randint(0, height))
  end = (random.randint(0, width), random.randint(0, height))
  draw.line([begin, end], fill = hp.linecolor)

#[0] is for both the end of sequence signal and the padding signal
def gene_code():
  height, width = hp.size
  image = Image.new('RGBA',(width,height), hp.bgcolor)

#######################################################################################
# random sampling code
#######################################################################################
  rand_index = random.randint(0, len(content_list)-1)
  con_text = content_list[rand_index]#[0:2]
  con_list = list(con_text)
  #print con_list[0]

  rand_font_size = random.randint(18, 25)
  rand_left = random.randint(10, 400-int(rand_font_size*len(con_list)/2) )
  rand_top = random.randint(3, 12-int(rand_font_size/3))

  rand_font_index = random.randint(0, len(font_file_list)-1)
  font_path = font_file_list[rand_font_index]
  #print font_path
#######################################################################################


  font = ImageFont.truetype(font_path, rand_font_size)
  draw = ImageDraw.Draw(image)

  #con_text = "".join(con_list)
  #print con_text
  mask_len = len(con_list) + 1
  padding_len = hp.max_len - mask_len
  # mask = [1]*mask_len + [0]*padding_len

  decoder_input_str = [hp.start_token] + con_list + [hp.pad_token]*padding_len
  decoder_target_str = con_list + [hp.end_token] + [hp.pad_token]*padding_len
  #print"decoder_input_str:",''.join(decoder_input_str)
  decoder_input_index = map(lambda x: hp.string_list.index(x), decoder_input_str)
  decoder_target_index = map(lambda x: hp.string_list.index(x), decoder_target_str)
  #print"decoder_input_index:",(decoder_input_index)
  decoder_length = mask_len
  #index_list += [hp.end_token]  # this is for the end of sequence signal
  #padding = max_len - len(con_list)
  #index_list += # this is for the padding signal

  font_width, font_height = font.getsize(con_text)
  draw.text((rand_left, rand_top), con_text, font= font, fill=hp.fontcolor)
  if hp.draw_line:
    gene_line(draw,width,height)
  image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
  #return text,np.array(image)
  return np.array(image.convert("L"))/128.0-1.0, decoder_length,  decoder_input_index,decoder_target_index

def myget_images(path,num):
    files = []
    imgList = []
    encoder_length = []
    target_batch =[]
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
        os.path.join(path, '*.{}'.format(ext))))

    endIndex = len(files);
    if num!=0:
        endIndex = num
    for i in range(len(files)):
        if i > endIndex-1:
            break;
        image =  Image.open(files[i])
        imgList.append(np.array(image.convert("L"))/128.0-1.0)
        encoder_length.append(150)
        target_batch.append( files[i][:-4])
    return imgList,encoder_length,target_batch
def generateChIndexCode(inputStr):
    index = [];
    character = '0'; curIndex = 0;
    for i in range(len(inputStr)):
        #print'inputStr:',inputStr[i]
        if inputStr[i] in hp.string_list:
            curIndex = hp.string_list.index(inputStr[i])
        else:
            curIndex = hp.string_list.index(hp.unknown_token)

        index.append(curIndex)
    return index
def recusiveSubFolder(path):
    imgFileList = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith((".jpg",'.jpg'))]
    return imgFileList
def myBatchGetImages(path,batch_size):
    files = []
    imgList = []
    i_len_batch = []
    target_batch =[]
    label_batch = []
    l_len_batch = []

    files = recusiveSubFolder(path)
    #print("Num:{}".format(len(files)))
    callNumber = np.arange(len(files))
    np.random.shuffle(callNumber)

    string = ""; stringNum = 0;
    index = 0
    i = 0
    #for i in range(batch_size):
    while(index<batch_size):
        #print"callNumber:",(files[callNumber[i]])
        txtPath = files[callNumber[i]][:-4]+'.txt'
        #print"txtPath:",(txtPath)
        i+=1

        if os.path.exists(txtPath)==False:
            continue;
        index+=1
        with codecs.open(txtPath,'r','utf-8') as f:
            string = f.read()
            stringNum = len(string)+1;
        #print"string:",(string)

        image =  Image.open(files[callNumber[i]])
        imgList.append(np.array(image.convert("L"))/128.0-1.0)

        if stringNum >= hp.max_len-1:
            stringNum = hp.max_len-1

        temp = [ch for ch in string[0:stringNum] ]

        conlist = list(temp)
        mask_len = len(conlist) + 1
        padding_len = hp.max_len - mask_len

        #print"batch_size:{}mask_len:{}".format(i,mask_len)

        decoder_input_str = [hp.start_token] + conlist + [hp.pad_token]*padding_len
        decoder_target_str = conlist + [hp.end_token] + [hp.pad_token]*padding_len

        decoder_input_index = generateChIndexCode(decoder_input_str)
        decoder_target_index = generateChIndexCode(decoder_target_str)

        #print "decoder_input_index:", decoder_input_index

        label_batch.append(decoder_input_index)#input string

        i_len_batch.append(150)#
        target_batch.append(decoder_target_index)#decoder_target
        l_len_batch.append(mask_len)#decoder_length input string's length +1
    yield imgList,label_batch,target_batch,i_len_batch,l_len_batch


def myBatchGetImagesForInfer(path,batch_size):
    files = []
    imgList = []
    i_len_batch = []
    target_batch =[]
    label_batch = []
    l_len_batch = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
        os.path.join(path, '*.{}'.format(ext))))
    #for i in range(len(files)):
    print("Num:{}".format(len(files)))
    callNumber = np.arange(len(files))
    #np.random.shuffle(callNumber)

    string = "123"; stringNum = 0;
    for i in range(batch_size):

        image =  Image.open(files[callNumber[i]])
        imgList.append(np.array(image.convert("L"))/128.0-1.0)

        string = files[callNumber[i]][:-4]
        temp = [ch for ch in string ]
        conlist = list(temp)
        mask_len = len(conlist) + 1
        padding_len = hp.max_len - mask_len


        decoder_input_str = [hp.start_token] + conlist + [hp.pad_token]*padding_len
        decoder_target_str = conlist + [hp.end_token] + [hp.pad_token]*padding_len

        decoder_input_index = generateChIndexCode(decoder_input_str)
        decoder_target_index = generateChIndexCode(decoder_target_str)
        #print "decoder_input_str:", decoder_input_index
        label_batch.append(decoder_input_index)#input string

        i_len_batch.append(150)#
        target_batch.append(decoder_target_index)#decoder_target
        l_len_batch.append(mask_len)#decoder_length input string's length +1
    yield imgList,label_batch,target_batch,i_len_batch,l_len_batch
def gen_batch(batch_size):
  image = []
  decoder_input = []
  decoder_target = []
  decoder_length = []
  encoder_length = []

  for i in range(batch_size):
    a, b, c, d = gene_code()
    image.append(a)
    decoder_length.append(b)#decoder_length
    decoder_input.append(c)#输入的字符串
    decoder_target.append(d)
    encoder_length.append(hp.input_length)
  #return np.array(x, dtype=np.int32), np.array(y), np.array(mask)
  yield image, decoder_input, decoder_target, encoder_length, decoder_length

if __name__ == '__main__':
    imgPath = './test171124/'
    files = recusiveSubFolder(imgPath)
    print(len(files))
    #test = gen_batch(2000)
    #input_batch, label_batch, target_batch, i_len_batch, l_len_batch = test.next()
    #for index in range(2000):
        #imgPath = './TrainImage/'+'Sim_'+str(index)+'.jpg'
        #io.imsave(imgPath, input_batch[index])
        #txtPath = imgPath[:-4]+'.txt'
        #with codecs.open(txtPath,'w','utf-8') as f:
            #f.write(target_batch[index])
        #pdb.set_trace()
