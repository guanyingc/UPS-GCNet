import os

def make_file(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def make_files(f_list):
    for f in f_list:
        make_file(f)

def dict_to_string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def check_in_list(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists
