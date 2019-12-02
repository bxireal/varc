from PIL import Image
import matplotlib.pyplot as plt



# path = '/data/JXR/video_data/SDR_540p/SDR_540p_output/97265060/97265060_47.png'
path = '/data/JXR/video_data/SDR_540p/SDR_540p_output/89627570/89627570_57.png'
path1 = '/data/JXR/video_data/train_data/89627570/89627570_57.png'
path2 = '/data/JXR/video_data/train_sr/89627570/89627570_57.png'




img = Image.open(path)
img1 = Image.open(path2)
img2 = Image.open(path3)
# img = Image.open(path)

plt.imshow(img)
plt.show()
plt.imshow(img2)
plt.show()
