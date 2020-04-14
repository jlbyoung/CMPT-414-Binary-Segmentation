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

from model.model import ENet



class VideoRecorder:
    def __init__(self, model, device):
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
        
        #button to save video
        self.but = tk.Button(text="Save Session As Video And Exit", command = self.saveVideo)
        self.but.grid(row = 1, column = 0)
        
        #dimensions of frame
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi',fourcc, 20, (self.frame_width*2,self.frame_height))
        
        self.device = device
        #model
        self.model = model.eval()
        
        self.video_stream()
        self.root.mainloop()
        
    def decode_segmap(self, image, nc=2):
      
      image = image * 255
        
      rgb = np.stack([image, image, image], axis=2)
      return rgb
    
    def segment(self, net, img):
        trf = T.Compose([
            
                   T.ToTensor(),
                   T.Normalize(mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        inp = inp.to(self.device)
        out = net(inp)

        out = torch.sigmoid(out.squeeze())
        out = (out > 0.6).to(torch.uint8).cpu().numpy()
        rgb = self.decode_segmap(out)

        return rgb
    
    def saveVideo(self):
        print("Saving Video")
        for i in self.list:
            
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            
            self.out.write(i)
        
        self.out.release()
        print("out")
        self.cap.release()
        print("cap")
        self.root.destroy()
          
    # function for video streaming
    #note: might not need transforms variable
    def video_stream(self, transforms = 0):
        #reads frame from webcam
        _, frame = self.cap.read()
        #webcam image is BGR, convert to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(type(cv2image))
        print(cv2image.shape)
        

        
        #since cv2 is np array, convert to PIL so that tkinter can display
        img = Image.fromarray(np.uint8(cv2image))
        
        #segments the image using fcn
        segment_img = self.segment(self.model, img)
        
        
        
        #must resize the img so that we can save video
        segment_img = cv2.resize(segment_img, (self.frame_width,self.frame_height))
        segment_img = Image.fromarray(segment_img, 'RGB')
        
        #display images side by side, cv2image is real time image, and segment_img is semantic segementation

        stacked = np.hstack((cv2image, segment_img))

        #list for saving video
        self.list.append(stacked)        

        #convert to PIL
        stacked = Image.fromarray(np.uint8(stacked))
        imgtk = ImageTk.PhotoImage(image=stacked)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        
        #pause is the amount of time before next webcam capture is display, change to 1 for smoothest
        pause = 1
        self.lmain.after(pause, self.video_stream) 


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Preview model using your webcam')
    args.add_argument('-m', '--model', default='model_best.pth', type=str,
                      help='path to model file (default: model_best.pth)')
    
    parsed = args.parse_args()
    model = ENet(num_classes=1)
    print('Loading model: {} ...'.format(parsed.model))
    
    state_dict = torch.load(parsed.model)['state_dict']

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(state_dict)
        model = model.to(device)
        video = VideoRecorder(model, device)
    else:
        print('NVIDIA GPU not found, aborting script.')