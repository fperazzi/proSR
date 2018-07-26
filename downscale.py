def downscale_by_ratio(img, ratio, method=Image.BICUBIC, magic_crop=False):
    if ratio == 1: return img

    w, h = img.size
    if magic_crop:
        img = img.crop((0, 0, w - w % ratio, h - h % ratio))
        w, h = img.size

    w, h = floor(w / ratio), floor(h / ratio)
    return img.resize((w, h), method)

    # ret_data['input_img']= downscale_by_ratio(input_img, self.scale,
    #                                method=Image.BICUBIC,magic_crop=True)
