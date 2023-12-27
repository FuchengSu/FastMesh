import numpy as np
import cv2
import imageio
import os
writer = imageio.get_writer(os.path.join("/workspace/nvrender/results/scan110", 'novel.gif'), mode='I')
for i in range(30):
    img = imageio.imread("/workspace/nvrender/results/scan110/novel_mask%05d.png"%i)
    writer.append_data((img).astype(np.uint8))
