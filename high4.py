#
# 画像処理
from PIL import Image
img=Image.open('bird.jpg')
#
# 縦横比を固定したままリサイズ
img.thumbnail((150,150))
img.save('bird1.jpg')
#
# ぼかし
from PIL import Image, ImageFilter
img=Image.open('bird.jpg')
img=img.filter(ImageFilter.BLUR)
img.save('bird2.jpg')
#
# エッジを強調
img=Image.open('bird.jpg')
img=img.filter(ImageFilter.EDGE_ENHANCE_MORE)
img.save('cbird3.jpg')
#
# エンボス版画
img=Image.open('bird.jpg')
img=img.filter(ImageFilter.EMBOSS)
img.save('bird4.jpg') 
#
# 最大値でフィルター
img=Image.open('bird.jpg')
img=img.filter(ImageFilter.MaxFilter(size=9))
img.save('bird5.jpg')
