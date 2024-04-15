#extracting bounding box data from the xml files

import xml.etree.ElementTree as ET

import os
import pandas as pd
'''
folder_list = ['barton', 'barton1', 'green', 'green1', 'parking', 'parking1']
for i in range(len(folder_list)):
    directory = '/Users/pavithrak/CUAD/Pavithra_Repo/yolov3/src/mast_data/' + folder_list[i]
    dict1 = {}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if ".xml" in f:
            tree = ET.parse(f)
            #parser = ET.XMLParser()
            root = tree.getroot()
           

            coord = {}
            for input in range(4):
                #print(root[6][4][input].text)
                if input == 0:
                    coord['xmin'] = root[6][4][input].text
                elif input == 1:
                    coord['ymin'] = root[6][4][input].text
                elif input == 2:
                    coord['xmax'] = root[6][4][input].text
                elif input == 3:
                    coord['ymax'] = root[6][4][input].text

            for input in range(2):
                if input == 0:
                    coord['width'] = root[4][input].text
                elif input == 1:
                    coord['height'] = root[4][input].text
        
                #bound_info.append(root[6][4][input].text)
            dict1[f[:-3] + 'jpg'] = coord
            coord = {}
            #print()

print(dict1)

'''
store_info = []
all_info = []

avg_info = []
avg_all_info = []

folder_list = ['barton', 'barton1', 'green', 'green1', 'parking', 'parking1']
for i in range(len(folder_list)):
    directory = '/Users/pavithrak/CUAD/Pavithra_Repo/yolov3/src/mast_data/' + folder_list[i]

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if ".xml" in f:
            tree = ET.parse(f)
            #parser = ET.XMLParser()
            root = tree.getroot()

            file_n = f

            text_name = f[:-3] + "txt"

            #d_file = open(text_name, 'w')
        
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


            #data_str = "0" + " " + str(x_center) + " " + str(y_center) + " " + str(width_of_box) + " " + str(height_of_box) 

            #d_file.write(data_str)
            #d_file.close()

            jpg_name = f[:-3] + 'jpg'
            #file_info = [jpg_name, text_name]
            #data_info.append(file_info)

            
            avg_info = [x_center, y_center, width_of_box, height_of_box, jpg_name]
            avg_all_info.append(avg_info)

            '''
            store_info = [jpg_name, xmin, ymin, xmax, ymax, width, height]
            all_info.append(store_info)
            '''



#xmlToDf = pd.DataFrame(data_info, columns = ['imagename', 'labelname'])

#xmlToDf.to_csv('mast_data.csv')

avg_xmlToDf = pd.DataFrame(avg_all_info, columns = ['x_center','y_center','width', 'height', 'filename'])
avg_xmlToDf.to_csv('avg_mast_data.csv')




        
        
                
    