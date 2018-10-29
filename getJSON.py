import glob

filename = 'dataFiles.json'
try:
    with open(filename) as f:
        root_file = eval(f.read())
except SyntaxError:
    print('Unable to open the {} file. Terminating...'.format(filename))
except IOError:
    print('Unable to find the {} file. Terminating...'.format(filename))

def can_get(attr):
    return bool(glob.glob(get(attr) + '*'))
    
def get(attr, root=root_file):
    node = root
    for part in attr.split('.'):
        node = node[part]
    return node