from PIL import Image
import numpy as np

class Image2arr:
	def __init__(self):
		pass
		
	def image2arr(self, path):
		img = Image.open(path).convert("L")
		#img.thumbnail((28,28), Image.ANTIALIAS)
		x, y = img.size
		data = list(img.getdata())
		res = []
		for i in range(28):
			row = []
			for j in range(28):
				row.append(data[i*28+j])
			res.append(row)
		a = np.array(res, dtype="uint8", ndmin=3)
		return a
	
	def print_arr(self, res):
		num_vals = 0
		row_str = ""
		for val in res:
			if val > 200:
					row_str += "  "
			elif val <= 200 and val > 150:
					row_str += "--"
			elif val <= 150:
					row_str += "++"
			num_vals += 1
			if num_vals >= 20:
				print(row_str)
				row_str = ""
				num_vals = 0
			