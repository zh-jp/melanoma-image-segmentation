from PIL import Image
import uuid


# def replace_colors(image):
#     # 将图像转换为RGBA模式
#     image = image.convert("RGBA")
#
#     # 获取图像的像素数据
#     data = image.getdata()
#
#     # 创建一个新的像素列表，用于存储替换颜色后的像素数据
#     new_data = []
#
#     for item in data:
#         # 判断当前像素的颜色是否为黑色
#         if item[:3] == (0, 0, 0):
#             # 替换为完全透明
#             new_data.append((0, 0, 0, 0))
#         # 判断当前像素的颜色是否为白色
#         elif item[:3] == (255, 255, 255):
#             # 替换为半透明的绿色（这里设置为RGBA中的(0, 255, 0, 128)）
#             new_data.append((0, 255, 0, 128))
#         else:
#             # 其他颜色保持不变
#             new_data.append(item)
#
#     # 更新图像的像素数据
#     image.putdata(new_data)
#
#     return image
def replace_colors(image):
    # 将图像转换为RGBA模式
    image = image.convert("RGBA")
    black_color = (0, 0, 0)
    white_color = (255, 255, 255)
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
            if (r, g, b) == black_color:
                image.putpixel((x, y), (r, g, b, 0))
            elif (r, g, b) == white_color:
                image.putpixel((x, y), (255, 0, 0, 100))
    return image


def mask_in_img(img_dir: str, mask_dir: str, alpha: float = .5, save_dir: str = "./") -> str:
    # alpha表示不透明度，0.0表示完全透明，1.0表示完全不透明
    image1 = Image.open(img_dir).convert("RGBA")  # 替换为原始图像的文件名或路径
    image2 = Image.open(mask_dir).convert("RGBA")  # 替换为掩膜图像的文件名或路径

    image2 = replace_colors(image2)

    # 调整第二张图片的大小，使其与第一张图片的大小相同（如果需要）
    image2 = image2.resize(image1.size)
    # 简单图像叠加
    result = Image.alpha_composite(image1, image2)

    # 使用blend方法混合两张图片
    # result = Image.blend(image1, image2, alpha)

    # 保存结果图像
    filename = save_dir + str(uuid.uuid1()) + '.png'
    result.save(filename)
    return filename
