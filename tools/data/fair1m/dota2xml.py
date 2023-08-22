import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import ast

# 读取文件内容并解析为字典
def parse_file(filename, class_name):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            items = line.strip().split(" ")
            data.append({
                'cls': class_name,
                'filename': items[0] + '.xml',
                'coordinates': items[1:]
            })
    return data


# 创建XML文档
def create_xml(data):
    for k, v in data.items():
        xml_path = os.path.join(xml_dir, k)

        root = ET.Element('annotation')
        # 添加source子元素
        source = ET.SubElement(root, 'source')
        filename = ET.SubElement(source, 'filename')
        filename.text = k.replace("xml", "tif")
        origin = ET.SubElement(source, 'origin')
        origin.text = 'GF2/GF3'
        # 添加research子元素
        research = ET.SubElement(root, 'research')
        version = ET.SubElement(research, 'version')
        version.text = '1.0'
        provider = ET.SubElement(research, 'provider')
        provider.text = 'Company/School of team'
        author = ET.SubElement(research, 'author')
        author.text = 'Team name'
        pluginname = ET.SubElement(research, 'pluginname')
        pluginname.text = 'FAIR1M'
        pluginclass = ET.SubElement(research, 'pluginclass')
        pluginclass.text = 'object detection'
        time = ET.SubElement(research, 'time')
        time.text = '2021-03'
        # 添加objects子元素
        objects = ET.SubElement(root, 'objects')
        for cls, coords in v.items():
            for coord in coords:
                coord = ast.literal_eval(coord)
                # 添加第一个object子元素
                object = ET.SubElement(objects, 'object')
                coordinate = ET.SubElement(object, 'coordinate')
                coordinate.text = 'pixel'
                type = ET.SubElement(object, 'type')
                type.text = 'rectangle'
                description = ET.SubElement(object, 'description')
                description.text = 'None'
                possibleresult = ET.SubElement(object, 'possibleresult')
                name = ET.SubElement(possibleresult, 'name')
                name.text = cls
                probability = ET.SubElement(possibleresult, 'probability')
                probability.text = coord[0]
                points = ET.SubElement(object, 'points')
                if len(coord) < 8:
                    print(f'{k},{cls},\n{coord}有误')
                # 添加坐标点
                for i in range(1, len(coord) - 1, 2):
                    point = ET.SubElement(points, 'point')
                    point.text = f'{coord[i]},{coord[i + 1]}'
                # 添加第五个坐标点（与第1个相同）
                point = ET.SubElement(points, 'point')
                point.text = f'{coord[1]},{coord[2]}'

        # 创建XML树
        tree = ET.ElementTree(root)
        # 将XML保存到文件
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        root.clear()


def extract_class_name(f_name):
    f = f_name.replace("Task1_", "")
    class_name = f.split(".")[0]
    class_name = class_name.replace("-", " ")
    return class_name


def trans(datas):
    """将类别--内容 转换为文件名--类别"""
    dct = {}
    for tmp_lst in datas:
        for tmp in tmp_lst:
            if tmp['filename'] in dct:
                if tmp['cls'] in dct[tmp['filename']]:
                    dct[tmp['filename']][tmp['cls']].append(str(tmp['coordinates']))
                else:
                    c = [str(tmp['coordinates'])]
                    dct[tmp['filename']][tmp['cls']] = c
            else:
                c = [str(tmp['coordinates'])]
                dct[tmp['filename']] = {tmp['cls']: c}
    return dct


txt_dir = "D:\\WorkSpace\\NerualNet\\dataset\\FAIR1M2.0\\predict\\"
xml_dir = "D:\\WorkSpace\\NerualNet\\dataset\\FAIR1M2.0\\test\\"
# 主程序
if __name__ == '__main__':
    file_lst = os.listdir(txt_dir)
    lst = []
    for f in file_lst:
        file_path = txt_dir + f
        class_name = extract_class_name(f)
        file_data = parse_file(file_path, class_name)
        lst.append(file_data)
    data_lst = trans(lst)
    xml_tree = create_xml(data_lst)
