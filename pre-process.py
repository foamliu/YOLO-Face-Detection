import zipfile


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    extract('WIDER_train')
    extract('WIDER_val')
    extract('WIDER_test')
    extract('wider_face_split')
