
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




one_hot_len = len(hp.string_list)
max_len_scale = hp.max_len


def gene_line(draw,width,height):
  begin = (random.randint(0, width), random.randint(0, height))
  end = (random.randint(0, width), random.randint(0, height))
  draw.line([begin, end], fill = hp.linecolor)

def draw_rotated_text(image, angle, text,leftTopPnt,rightBtmPnt, fontSize,byLine, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:
    :param image: Image to write text into
    :param angle: Angle to write text at
    """

    grayL = 10; grayH = 50;
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

    color_image = Image.new('RGBA', image.size, fontColor)
    image.paste(color_image, mask)
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
def GenerateDrawPosByBlocks(img,txt,pathToFont,row,col,splitNum,angle):
    [imgW,imgH]= img.size
    blockW = int(np.ceil(imgW/splitNum))
    blockH = int(np.ceil(imgH/splitNum))

    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)

    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)+1
    font = ImageFont.truetype(pathToFont, fontSize)

    img_fraction = 0.7

    longestLength = 1
    hwRatio = font.getsize(txt)[1]/(font.getsize(txt)[0])
    formalAngle = np.tanh(hwRatio)*180.0/math.pi;
    anglePi = (formalAngle+np.abs(angle))* math.pi/180.0
    sinA = math.sin(anglePi)
    maxLength = 40 /(sinA+0.00001)
    while (font.getsize(txt)[0] < img_fraction*blockW and font.getsize(txt)[1] < img_fraction*blockH \
            and longestLength < maxLength*img_fraction):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
              longestLength = np.sqrt(font.getsize(txt)[0]*font.getsize(txt)[0] + font.getsize(txt)[1]*font.getsize(txt)[1])

    fontSize -= 1
    font = ImageFont.truetype(pathToFont, fontSize)

    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)

    cenX = np.random.randint( int(font.getsize(txt)[0]*0.5),(int(blockW-font.getsize(txt)[0]*0.5)))
    origin=[]
    origin.append(int(cenX-font.getsize(txt)[0]*0.5))
    origin.append(int(cenY-font.getsize(txt)[1]*0.5))
    cen=[]
    cen.append(int(cenX))
    cen.append(int(cenY))
    GenerateRotatedPnt(cen,-angle,origin)
    #print'fontSize', fontSize
    return origin,fontSize;

def GenerateTextLineImg(chooseString):

    imgW = 800;
    imgH = 40;

    tR = np.random.randint(200,250)
    emptyImg = Image.new('L', (imgW,imgH), (tR))

    newTextSize=[];
    curAngle = 0;

    maxAngle = 2
    curAngle = np.random.randint(-maxAngle,maxAngle)

    row= 0
    col = 0
    font_file_list = glob.glob("fonts/*.*")
    rand_font_index = random.randint(0, len(font_file_list)-1)
    pathToFont = font_file_list[rand_font_index]
    fontSize=10
    font = ImageFont.truetype(pathToFont, fontSize)

    origin,fontSize = GenerateDrawPosByBlocks(emptyImg,chooseString,pathToFont,row,col,1,curAngle)
    font = ImageFont.truetype(pathToFont, fontSize)

    newTextSize = font.getsize(chooseString)

    pnt=[]
    pnt.append(origin[0]+newTextSize[0])
    pnt.append(origin[1]+newTextSize[1])

    GenerateRotatedPnt(origin,curAngle,pnt)
    byLine = 1
    draw_rotated_text(emptyImg, curAngle, chooseString, origin,pnt,byLine,fontSize, font=font)
    return emptyImg
#[0] is for both the end of sequence signal and the padding signal
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
def gene_code(con_text):

    con_list = list(con_text)
    #print con_list[0]

    #print font_path
    #print con_text
    mask_len = len(con_list) + 1
    padding_len = hp.max_len - mask_len
    # mask = [1]*mask_len + [0]*padding_len

    decoder_input_str = [hp.start_token] + con_list + [hp.pad_token]*padding_len
    decoder_target_str = con_list + [hp.end_token] + [hp.pad_token]*padding_len
    #print"decoder_input_str:",''.join(decoder_input_str)
    #decoder_input_index = map(lambda x: hp.string_list.index(x), decoder_input_str)
    #decoder_target_index = map(lambda x: hp.string_list.index(x), decoder_target_str)
    decoder_input_index = generateChIndexCode(decoder_input_str)
    decoder_target_index = generateChIndexCode(decoder_target_str)
    #print"decoder_input_index:",(decoder_input_index)
    decoder_length = mask_len

    #image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    image = GenerateTextLineImg(con_text)
    #return text,np.array(image)
    return np.array(image.convert("L")), decoder_length,  decoder_input_index, decoder_target_index

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
def myBatchGetImages(path,batch_size):
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
    #print("Num:{}".format((files)))
    callNumber = np.arange(len(files))
    np.random.shuffle(callNumber)

    string = ""; stringNum = 0;
    for i in range(batch_size):
        #print"callNumber:",(files[callNumber[i]])
        txtPath = files[callNumber[i]][:-4]+'.txt'
        #print"txtPath:",(txtPath)
        with codecs.open(txtPath,'r','utf-8') as f:
            string = f.read()
            stringNum = len(string)+1;
        #print"string:",(string)

        image =  Image.open(files[callNumber[i]])
        imgList.append(np.array(image.convert("L"))/128.0-1.0)

        temp = [ch for ch in string ]

        conlist = list(temp)
        mask_len = len(conlist) + 1
        padding_len = hp.max_len - mask_len

        decoder_input_str = [hp.start_token] + conlist + [hp.pad_token]*padding_len
        decoder_target_str = conlist + [hp.end_token] + [hp.pad_token]*padding_len

        decoder_input_index = generateChIndexCode(decoder_input_str)
        decoder_target_index = generateChIndexCode(decoder_target_str)

        #print "decoder_input_index:", decoder_input_index

        label_batch.append(decoder_input_index)#input string

        i_len_batch.append(150)#
        target_batch.append(decoder_target_index)#decoder_target
        l_len_batch.append(mask_len)#decoder_length input string's length +1
    return imgList,label_batch,target_batch,i_len_batch,l_len_batch


def gen_batch(batch_size,txtFolder,savePath):
    image = []
    decoder_input = []
    decoder_target = []
    decoder_length = []
    encoder_length = []

    content_list = []
    txtName = []
    files = []
    files.extend(glob.glob(os.path.join(txtFolder, '*.{}'.format('txt'))))
    txtName = files[np.random.randint(len(files))]
    text_file = codecs.open(txtName,"r",'utf-8')
    text_content = text_file.readlines()
    for content in text_content:
        #print'content', content
        #if content != '\r\n' and content != '\n' and content != '\t\r\n': # for chinese_ocr
        #if content != '\r\n' and content != '\n' and content != '\t\r\n':
        content_list.append(content.strip(' '))


    index = 0
    while(index <batch_size):
        if len(content_list)<=1:
            index += 1
            continue

        rand_index = random.randint(0, len(content_list)-1)
        con_text = content_list[rand_index]#[0:2]
        con_text = ''.join(con_text)
        if len(con_text) > 50:
            #listLength = np.random.randint(8,30) # for chinese_ocr
            listLength = np.random.randint(10,50) # for chinese_ocr
            con_text = con_text[0:listLength-1]
        if len(con_text) < 15: # 10 # for chinese
            continue
        index += 1
        a, b, c, d = gene_code(con_text)
        image.append(a)
        decoder_input.append(con_text)
    yield image, decoder_input, decoder_target, encoder_length, decoder_length

def recusiveSubFolder(path):
    imgFileList = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith((".jpg"))]
    return imgFileList

if __name__ == '__main__':
     txtFolder = './foreign novel/chinese/'
     savePath = './syn_CH_171204V2/'

     batchNum = 40000; batchSize = 16;
     for i in range(batchNum):
         test = gen_batch(batchSize,txtFolder,savePath)
         input_batch, label_batch, target_batch, i_len_batch, l_len_batch = test.next()
         if len(input_batch) <=1:
             continue;
         for index in range(batchSize):
             imgPath = savePath +'Syn_'+str(index+i*batchSize)+'.jpg'
             io.imsave(imgPath, input_batch[index])
             textPath = savePath +'Syn_'+str(index+i*batchSize)+'.txt'
             with codecs.open(textPath,'w','utf-8') as f:
                 f.write(label_batch[index])
             print 'Num:{} is finished'.format(index+i*batchSize)

#    txtPath= '/media/veilytech/Model/chinese_ocr/TestImage/jin_000_0.txt'
#
#    string ='wefsdf'
#    with codecs.open(txtPath,'r','utf-8') as f:
#        string = f.read()
#        stringNum = len(string)+1;
#
#    usample=string
#    print"string befor repr:",(usample)
#    #conlist = re.findall(u'[\u4e00-\u9fff]+', usample)#replace
#    #print"string after repr:",(conlist)
#
#    conlist = [ch for ch in usample]
#    mask_len = len(conlist) + 1
#    padding_len = hp.max_len - mask_len
#
#    print'curString', conlist
#
#    decoder_input_str = [hp.start_token] + conlist + [hp.pad_token]*padding_len
#    decoder_target_str = conlist + [hp.end_token] + [hp.pad_token]*padding_len
#
#    decoder_input_index = generateChIndexCode(decoder_input_str)
#    decoder_target_index = generateChIndexCode(decoder_target_str)
#
#    print "decoder_input_str:", decoder_input_index
#    print "decoder_target_index:", decoder_target_index
