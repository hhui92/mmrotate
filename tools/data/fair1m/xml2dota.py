import os
import os.path as osp
import xml.etree.ElementTree as ET
from tqdm import tqdm

'''
    来源
    分辨率
    x1, y1, x2, y2, x3, y3, x4, y4, 类别, 难度
'''

gsd_dict = {'GF2/GF3': 0.8}

labels = []


def fair1m2dota(data_folder):
    print('-- start fair1m2dota --')
    xml_folder = osp.join(data_folder, 'labelXml')
    txt_folder = osp.join(data_folder, 'labelTxt')  # 保存
    if not osp.exists(txt_folder):
        os.mkdir(txt_folder)
    xml_names = os.listdir(xml_folder)
    for xml_name in tqdm(xml_names):
        xml_path = osp.join(xml_folder, xml_name)
        txt_path = osp.join(txt_folder, xml_name.replace('.xml', '.txt'))
        # 写txt
        with open(txt_path, 'w') as f:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            imagesource = root.find('source').find('origin').text
            f.write('imagesource:' + imagesource + '\n')  # 写来源
            f.write('gsd:' + str(gsd_dict[imagesource]) + '\n')  # 写分辨率
            for obj in root.find('objects').findall('object'):
                clas = obj.find('possibleresult').find('name').text.replace(' ', '-')
                if clas not in labels:
                    labels.append(clas)
                points = obj.find('points')
                x_list = []
                y_list = []
                for point in points.findall('point'):
                    xy = point.text.split(',')
                    x = float(xy[0])
                    y = float(xy[1])
                    x_list.append(x)
                    y_list.append(y)
                point_str = points_str(x_list, y_list)
                f.write(point_str)  # 写坐标
                f.write(clas + ' 0\n')  # 写类别
    return labels


def points_str(x_list, y_list):
    tmp = 0
    for i in range(5):
        for j in range(i, 5, 1):
            if x_list[i] == x_list[j] and y_list[i] == y_list[j]:
               tmp = j
               break
    del x_list[tmp]
    del y_list[tmp]
    point_str = ''
    for x, y in zip(x_list, y_list):
        point_str += str(x)
        point_str += ' '
        point_str += str(y)
        point_str += ' '

    return point_str


if __name__ == '__main__':
    data_path = r"D:\WorkSpace\NerualNet\dataset\FAIR1M2.0\validation"
    fair1m2dota(data_path)
    print(labels)
