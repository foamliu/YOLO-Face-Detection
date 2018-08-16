import zipfile

from config import train_annot_file, valid_annot_file


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def parse_annot(usage):
    if usage == 'train':
        annot_file = train_annot_file
    else:
        annot_file = valid_annot_file

    with open(annot_file, 'r') as file:
        lines = file.readlines()

    annots = []
    i = 0
    while True:
        filename = lines[i].strip()
        num_bbox = int(lines[i + 1].strip())
        bboxes = []
        for j in range(num_bbox):
            tokens = lines[i + j + 2].strip().split()
            x1 = int(tokens[0])
            y1 = int(tokens[1])
            w = int(tokens[2])
            h = int(tokens[3])
            bboxes.append((x1, y1, w, h))
        annots.append({'filename': filename, 'bboxes': bboxes})
        i = i + num_bbox + 2
        if len(lines) - i < 3:
            break

    print('parsed {} images'.format(len(annots)))
    return annots


if __name__ == '__main__':
    extract('WIDER_train')
    extract('WIDER_val')
    extract('WIDER_test')

    extract('wider_face_split')
    parse_annot('train')
    parse_annot('valid')
