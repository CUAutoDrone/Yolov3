import xml.etree.ElementTree as ET

import os
import pandas as pd


avg_info = []
avg_all_info = []

'''
directory = '/Users/pavithrak/CUAD/Pavithra_Repo/yolov3/src/mast_data/barton'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if ".xml" in f:
        tree = ET.parse(f)
        
        root = tree.getroot()

        file_n = f

        text_name = f[:-3] + "txt"

        
        for input in range(4):
                #print(root[6][4][input].text)
            if input == 0:
                xmin = root[6][4][input].text
            elif input == 1:
                ymin = root[6][4][input].text
            elif input == 2:
                xmax = root[6][4][input].text
            elif input == 3:
                ymax = root[6][4][input].text

        for input in range(2):
            if input == 0:
                width = root[4][input].text
            elif input == 1:
                height = root[4][input].text


        width_of_box = (int(xmax) - int(xmin))/ int(width)
        height_of_box = (int(ymax) - int(ymin))/ int(height)

        x_center = ((int(xmin) + int(xmax))/2)/ int(width)
        y_center = ((int(ymin) + int(ymax))/2)/ int(height) 


        jpg_name = f[:-3] + 'jpg'
        

            
        avg_info = [x_center, y_center, width_of_box, height_of_box, jpg_name]
        avg_all_info.append(avg_info)


avg_xmlToDf = pd.DataFrame(avg_all_info, columns = ['x_center','y_center','width', 'height', 'filename'])
avg_xmlToDf.to_csv('avg_test_mast_data.csv')

'''

directory = "/Users/pavithrak/CUAD/Pavithra_Repo/yolov3/src/MastDataTest/MastData/masts/test"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if ".xml" in f:
        tree = ET.parse(f)
        
        root = tree.getroot()

        file_n = f

        text_name = f[:-3] + "txt"

        
        for input in range(4):
                #print(root[6][4][input].text)
            if input == 0:
                xmin = root[6][4][input].text
            elif input == 1:
                ymin = root[6][4][input].text
            elif input == 2:
                xmax = root[6][4][input].text
            elif input == 3:
                ymax = root[6][4][input].text

        for input in range(2):
            if input == 0:
                width = root[4][input].text
            elif input == 1:
                height = root[4][input].text


        width_of_box = (int(xmax) - int(xmin))/ int(width)
        height_of_box = (int(ymax) - int(ymin))/ int(height)

        x_center = ((int(xmin) + int(xmax))/2)/ int(width)
        y_center = ((int(ymin) + int(ymax))/2)/ int(height) 


        jpg_name = f[:-3] + 'jpg'
        

            
        avg_info = [x_center, y_center, width_of_box, height_of_box, jpg_name]
        avg_all_info.append(avg_info)


avg_xmlToDf = pd.DataFrame(avg_all_info, columns = ['x_center','y_center','width', 'height', 'filename'])
avg_xmlToDf.to_csv('new_test_data.csv')