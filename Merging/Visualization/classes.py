import numpy as np

CLASSES = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
REL_CLASSES = ['above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
valid_class = [3, 5, 8, 9, 10, 14, 16, 17, 18, 23, 27, 29, 31, 33, 34, 35, 37, 70, 92, 95, 96, 100, 102, 106, 109, 111, 115, 122, 125, 130, 131, 138, 139, 144]
valid_rel_class = [0, 2, 6, 19, 21, 28, 30, 42, 49]

obj_colors = np.array([[224, 208, 184], [216, 224, 184], [192, 224, 184], [184, 224, 200], [184, 224, 224], [184, 200, 224], [192, 184, 224], [216, 184, 224], [224, 184, 208], [224, 128, 108], [224, 198, 108], [180, 224, 108], [110, 224, 108], [108, 224, 175], [108, 203, 224], [108, 134, 224], [151, 108, 224], [221, 108, 224], [223, 159, 171], [223, 190, 159], [213, 223, 159], [171, 223, 159], [159, 223, 190], [159, 213, 223], [159, 171, 223], [190, 159, 223], [221, 60, 168], [221, 60, 61], [221, 166, 60], [168, 221, 60], [ 61, 221, 60], [ 60, 221, 166], [ 60, 168, 221], [ 60, 61, 221]], dtype=np.uint8) / 255.
rel_colors = np.array([[238, 43, 43], [238, 173, 43], [173, 238, 43], [ 43, 238, 43], [ 43, 238, 173], [ 43, 173, 238], [ 43, 43, 238], [173, 43, 238], [238, 43, 173]], dtype=np.uint8) / 255.