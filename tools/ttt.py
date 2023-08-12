import os


if __name__ == '__main__':
    # imgs = os.listdir('D:\\WorkSpace\\NerualNet\\dataset\\DOTA-v1.0\\train\\annfiles\\')
    # s = set()
    # for img in imgs:
    #    s.add(img)
    # print(len(s))
    # imgs = os.listdir('D:\\WorkSpace\\NerualNet\\dataset\\DOTA-v1.0\\val\\annfiles\\')
    # for img in imgs:
    #     s.add(img)
    # print(len(s))
    total_lit = []
    with open("C:\\Users\\hhui9\\Desktop\\ff.txt", 'r') as f:
        lines = f.readlines()
        lst = [line.strip() for line in lines]
        total_lit.append(lst)
    with open("C:\\Users\\hhui9\\Desktop\\f.txt", 'r') as f:
        lines = f.readlines()
        lst = [line.strip() for line in lines]
        total_lit.append(lst)

    for t in total_lit[1]:
        if t in total_lit[0]:
            total_lit[0].remove(t)
            total_lit[1].remove(t)

    print(len(total_lit[0]))
    print(len(total_lit[1]))
    with open("C:\\Users\\hhui9\\Desktop\\ff_remove.txt", 'w') as f:
        for tt in total_lit[0]:
            f.write(tt + "\n")
    with open("C:\\Users\\hhui9\\Desktop\\f_remove.txt", 'w') as f:
        for t in total_lit[1]:
            f.write(t + "\n")