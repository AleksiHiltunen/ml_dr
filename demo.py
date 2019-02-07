from i2a import Image2arr
from ml_dr import NN_Model


def init():
	model = NN_Model()

	i2a = Image2arr()
	images = []
	images.append(i2a.image2arr("test_data/aaa.png"))
	images.append(i2a.image2arr("test_data/bbb.png"))
	images.append(i2a.image2arr("test_data/ccc.png"))
	images.append(i2a.image2arr("test_data/ddd.png"))
	images.append(i2a.image2arr("test_data/eee.png"))
	images.append(i2a.image2arr("test_data/fff.png"))
	images.append(i2a.image2arr("test_data/ggg.png"))

	return model, images