from app import app
from PIL import Image
from flask import render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')


 
def get_model():
	checkpoint_path='app/checkpoint.pth'
	checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	model = models.densenet121(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False
	model.classifier = nn.Sequential(nn.Linear(1024, 512),
	                                 nn.ReLU(),
	                                 nn.Dropout(0.25),
	                                 nn.Linear(512, 102),
	                                 nn.LogSoftmax(dim=1))
	model.load_state_dict(checkpoint['model_state_dict'],strict=False)
	
	return model
model=get_model()	

def process_image(image_bytes):
	test_transforms = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
	img=Image.open(io.BytesIO(image_bytes))
	if img.size[0] > img.size[1]:
		img.thumbnail((10000, 256))
	else:
		img.thumbnail((256, 10000))
	left_margin = (img.width-224)/2
	bottom_margin = (img.height-224)/2
	right_margin = left_margin + 224
	top_margin = bottom_margin + 224
	img = img.crop((left_margin, bottom_margin, right_margin,   
	                  top_margin))
	# Normalize
	img = np.array(img)/255
	mean = np.array([0.485, 0.456, 0.406]) #provided mean
	std = np.array([0.229, 0.224, 0.225]) #provided std
	img = (img - mean)/std
	img = img.transpose((2, 0, 1))
	return img
class_to_idx={'1': 0,
 '10': 1,
 '100': 2,
 '101': 3,
 '102': 4,
 '11': 5,
 '12': 6,
 '13': 7,
 '14': 8,
 '15': 9,
 '16': 10,
 '17': 11,
 '18': 12,
 '19': 13,
 '2': 14,
 '20': 15,
 '21': 16,
 '22': 17,
 '23': 18,
 '24': 19,
 '25': 20,
 '26': 21,
 '27': 22,
 '28': 23,
 '29': 24,
 '3': 25,
 '30': 26,
 '31': 27,
 '32': 28,
 '33': 29,
 '34': 30,
 '35': 31,
 '36': 32,
 '37': 33,
 '38': 34,
 '39': 35,
 '4': 36,
 '40': 37,
 '41': 38,
 '42': 39,
 '43': 40,
 '44': 41,
 '45': 42,
 '46': 43,
 '47': 44,
 '48': 45,
 '49': 46,
 '5': 47,
 '50': 48,
 '51': 49,
 '52': 50,
 '53': 51,
 '54': 52,
 '55': 53,
 '56': 54,
 '57': 55,
 '58': 56,
 '59': 57,
 '6': 58,
 '60': 59,
 '61': 60,
 '62': 61,
 '63': 62,
 '64': 63,
 '65': 64,
 '66': 65,
 '67': 66,
 '68': 67,
 '69': 68,
 '7': 69,
 '70': 70,
 '71': 71,
 '72': 72,
 '73': 73,
 '74': 74,
 '75': 75,
 '76': 76,
 '77': 77,
 '78': 78,
 '79': 79,
 '8': 80,
 '80': 81,
 '81': 82,
 '82': 83,
 '83': 84,
 '84': 85,
 '85': 86,
 '86': 87,
 '87': 88,
 '88': 89,
 '89': 90,
 '9': 91,
 '90': 92,
 '91': 93,
 '92': 94,
 '93': 95,
 '94': 96,
 '95': 97,
 '96': 98,
 '97': 99,
 '98': 100,
 '99': 101}

def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    image_tensor.to('cpu')
    model_input.to('cpu')
    model.to('cpu')
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers




cat_to_name={'1': 'pink primrose',
 '10': 'globe thistle',
 '100': 'blanket flower',
 '101': 'trumpet creeper',
 '102': 'blackberry lily',
 '11': 'snapdragon',
 '12': "colt's foot",
 '13': 'king protea',
 '14': 'spear thistle',
 '15': 'yellow iris',
 '16': 'globe-flower',
 '17': 'purple coneflower',
 '18': 'peruvian lily',
 '19': 'balloon flower',
 '2': 'hard-leaved pocket orchid',
 '20': 'giant white arum lily',
 '21': 'fire lily',
 '22': 'pincushion flower',
 '23': 'fritillary',
 '24': 'red ginger',
 '25': 'grape hyacinth',
 '26': 'corn poppy',
 '27': 'prince of wales feathers',
 '28': 'stemless gentian',
 '29': 'artichoke',
 '3': 'canterbury bells',
 '30': 'sweet william',
 '31': 'carnation',
 '32': 'garden phlox',
 '33': 'love in the mist',
 '34': 'mexican aster',
 '35': 'alpine sea holly',
 '36': 'ruby-lipped cattleya',
 '37': 'cape flower',
 '38': 'great masterwort',
 '39': 'siam tulip',
 '4': 'sweet pea',
 '40': 'lenten rose',
 '41': 'barbeton daisy',
 '42': 'daffodil',
 '43': 'sword lily',
 '44': 'poinsettia',
 '45': 'bolero deep blue',
 '46': 'wallflower',
 '47': 'marigold',
 '48': 'buttercup',
 '49': 'oxeye daisy',
 '5': 'english marigold',
 '50': 'common dandelion',
 '51': 'petunia',
 '52': 'wild pansy',
 '53': 'primula',
 '54': 'sunflower',
 '55': 'pelargonium',
 '56': 'bishop of llandaff',
 '57': 'gaura',
 '58': 'geranium',
 '59': 'orange dahlia',
 '6': 'tiger lily',
 '60': 'pink-yellow dahlia',
 '61': 'cautleya spicata',
 '62': 'japanese anemone',
 '63': 'black-eyed susan',
 '64': 'silverbush',
 '65': 'californian poppy',
 '66': 'osteospermum',
 '67': 'spring crocus',
 '68': 'bearded iris',
 '69': 'windflower',
 '7': 'moon orchid',
 '70': 'tree poppy',
 '71': 'gazania',
 '72': 'azalea',
 '73': 'water lily',
 '74': 'rose',
 '75': 'thorn apple',
 '76': 'morning glory',
 '77': 'passion flower',
 '78': 'lotus lotus',
 '79': 'toad lily',
 '8': 'bird of paradise',
 '80': 'anthurium',
 '81': 'frangipani',
 '82': 'clematis',
 '83': 'hibiscus',
 '84': 'columbine',
 '85': 'desert-rose',
 '86': 'tree mallow',
 '87': 'magnolia',
 '88': 'cyclamen',
 '89': 'watercress',
 '9': 'monkshood',
 '90': 'canna lily',
 '91': 'hippeastrum',
 '92': 'bee balm',
 '93': 'ball moss',
 '94': 'foxglove',
 '95': 'bougainvillea',
 '96': 'camellia',
 '97': 'mallow',
 '98': 'mexican petunia',
 '99': 'bromelia'}

# def return_name(inf):
# 	a=[cat[i] for i in inf]
# 	return a

idx_to_class={0: '1', 1: '10', 2: '100', 3: '101', 4: '102', 5: '11', 6: '12', 7: '13', 8: '14', 9: '15', 10: '16', 11: '17', 12: '18', 13: '19', 14: '2', 15: '20', 16: '21', 17: '22', 18: '23', 19: '24', 20: '25', 21: '26', 22: '27', 23: '28', 24: '29', 25: '3', 26: '30', 27: '31', 28: '32', 29: '33', 30: '34', 31: '35', 32: '36', 33: '37', 34: '38', 35: '39', 36: '4', 37: '40', 38: '41', 39: '42', 40: '43', 41: '44', 42: '45', 43: '46', 44: '47', 45: '48', 46: '49', 47: '5', 48: '50', 49: '51', 50: '52', 51: '53', 52: '54', 53: '55', 54: '56', 55: '57', 56: '58', 57: '59', 58: '6', 59: '60', 60: '61', 61: '62', 62: '63', 63: '64', 64: '65', 65: '66', 66: '67', 67: '68', 68: '69', 69: '7', 70: '70', 71: '71', 72: '72', 73: '73', 74: '74', 75: '75', 76: '76', 77: '77', 78: '78', 79: '79', 80: '8', 81: '80', 82: '81', 83: '82', 84: '83', 85: '84', 86: '85', 87: '86', 88: '87', 89: '88', 90: '89', 91: '9', 92: '90', 93: '91', 94: '92', 95: '93', 96: '94', 97: '95', 98: '96', 99: '97', 100: '98', 101: '99'}


@app.route('/',methods=['GET','POST'])
def index():
	    if request.method == 'POST':
	        # check if the post request has the file part
	        if 'file' not in request.files:
	            print('No file part')
	            return redirect(request.url)
	        file = request.files['file']
	        image_recieved=file.read()    
	        if file.filename == '':
	            flash('No selected file')
	            return redirect(request.url)
	        prob,top_labels,names=predict(image_recieved,model)
	        print()
	        return render_template('after.html',n1=names[0].capitalize(),n2=names[1].capitalize(),n3=names[2].capitalize(),n4=names[3].capitalize(),n5=names[4].capitalize(),p1=prob[0],p2=prob[1],p3=prob[2],p4=prob[3],p5=prob[4])
	    return render_template('index.html')	      


