# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:50:45 2020

@author: james
"""

import argparse
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
import model.model as module_arch
from parse_config import ConfigParser
import os.path


def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(net, img):
    trf = T.Compose([T.Resize(256),
               T.CenterCrop(224),
               T.ToTensor(),
               T.Normalize(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = net(inp)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    return rgb

class VideoRecorder:
    def __init__(self, model):
        self.cap = cv2.VideoCapture(0)
        
        self.list = []
        
        #tkinter app
        self.root = tk.Tk()
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        #create frame to contain video
        self.app = tk.Frame(self.root, bg="white")
        self.app.grid()
        
        #create label in the frame
        self.lmain = tk.Label(self.app)
        self.lmain.grid(row=0, column=0)
        
        self.lmain2 = tk.Label(self.app)
        self.lmain2.grid(row=0, column=1)
        
        #button to save video
        self.but = tk.Button(text="Save Session As Video And Exit", command = self.saveVideo)
        self.but.grid(row = 1, column = 0)
        
        #dimensions of frame
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        self.out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
        
        
        #model
        self.model = model.eval()
        #pretrained 
        #self.model = models.segmentation.fcn_resnet101(pretrained=1).eval()
        
        self.video_stream()
        self.root.mainloop()
        

    
    def saveVideo(self):
        print("Saving Video")
        #fcn = models.segmentation.fcn_resnet101(pretrained=1).eval()
        for i in self.list:
            """
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            
            i = Image.fromarray(np.uint8(i))
            i = segment(fcn, i)
            """
            i = cv2.resize(i, (self.frame_width, self.frame_height))
            
            self.out.write(i)
        
        self.out.release()
        print("out")
        self.cap.release()
        print("cap")
        self.root.destroy()
        
    #note: might not need transforms variable
    #draw rectangle
    def modifyFrame(self, img, transforms):
        start_pos = (200, 100)
        end_pos = (500, 400)
        color = (255, 0, 0)
        line_width = 3
        cv2.rectangle(img, start_pos, end_pos, color, line_width)
        return img
    
    # function for video streaming
    #note: might not need transforms variable
    def video_stream(self, transforms = 0):
        #reads frame from webcam
        _, frame = self.cap.read()
        #webcam image is BGR, convert to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #draw rectangle
        cv2image = self.modifyFrame(cv2image, transforms)
        
        #list for saving video
        #self.list.append(cv2image)
        
        #since cv2 is np array, convert to PIL so that tkinter can display
        img = Image.fromarray(np.uint8(cv2image))
        
        #segments the image using fcn
        segment_img = segment(self.model, img)
        print(type(self.model))
        
        
        
        #must resize the img so that we can save video
        segment_img = cv2.resize(segment_img, (self.frame_width, self.frame_height))
        #display images side by side, cv2image is real time image, and segment_img is semantic segementation
        stacked = np.hstack((cv2image, segment_img))
        
        #convert to PIL
        stacked = Image.fromarray(np.uint8(stacked))
        imgtk = ImageTk.PhotoImage(image=stacked)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        
        #pause is the amount of time before next webcam capture is display, change to 1 for smoothest
        pause = 5
        self.lmain.after(pause, self.video_stream) 


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    logger = config.get_logger('test')
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    
    checkpoint = torch.load(os.path.basename("C:/Users/james/Downloads/CMPT-414-R/CMPT-414-Rotoscoping/checkpoint-epoch1.pth"))
    
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    video = VideoRecorder(model)