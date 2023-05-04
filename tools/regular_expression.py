# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/12 下午 03:53

class Regularexpression:

    '''构建正则表达式'''
    def pattern(self):
        patterns = \
            [
                r"CHA2DS2-VASc+[ :：,，]+[\d]+[分级]",
                r"CHA2DS2-VASc+[\d]+[分级]",
                r"STAF+[ ，：:,]+[\d]+[分级]",
                r"STAF+[\d]+[分级]",
                r"洼田吞咽功能障碍评价+[   ,:，：]+[\d]+[分级]",
                r"洼田吞咽功能障碍评价+[\d]+[分级]",
                r"mRS+[ ,:  ，：]+[\d零一二三四五六七八九十]+[分级]",
                r"mRS+[\d零一二三四五六七八九十]+[分级]",
                r"NIHSS+[:\d：，  ,]+\d+[分级]",
                r"NIHSS+[\d]+[分级]",
                r"洼田饮水试验+[：  ,，:]+[\d]+[分级]",
                r"洼田饮水试验+[\d]+[分级]",
                r"Essen+[：，, :]+[\d]+[分级]",
                r"Essen+[\d]+[分级]",
                r"HAS-BLED+[  ：，:,]+[\d]+[分级]",
                r"HAS-BLED+[\d]+[分级]",
                r"GUSS+[  ，：,:]+[\d]+[分級]",
                r"GUSS+[\d]+[分級]",
                r"Gugging+[：: ,，]+[\d]+[分級]",
                r"Gugging+[\d]+[分級]",
                r"mTICI+[：, ，:]+[\d\w]+[分级]",
                r"mTICI+[\d\w]+[分级]",
                r"神经功能缺损评分+[ ，:：,]+[\d]+[分级]",
                r"神经功能缺损评分+[\d\d]+[分级]",

            ]

        return patterns