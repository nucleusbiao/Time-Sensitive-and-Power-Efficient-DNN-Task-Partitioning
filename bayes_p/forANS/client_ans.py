# 添加前端延时预测，添加moblienet 和 alexnet 23.10.15，（连接不断开）
# 添加try except 10.21



import argparse
import subprocess
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
import json
import os
import psutil

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
print("client_ans parent_dir  ",parent_dir)
sys.path.append(parent_dir)
parent_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_parent_dir)

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from models.resnet import resnet50
from models.alexnet import alexnet
from models.mobilenetv2 import mobilenetv2
#from keyFrameDetection import KeyFrameDetection
from communicationdeep import clientCommunication
from muLinUCB_addfront import muLinUCB
from yolo_utils import load_class_names, get_boxes, plot_boxes_cv2

WINDOW_NAME = 'CameraDemo'

vgg_info_de = { # action No. : [layer type num{1: conv, 2: fc, 3: act}, total mac{1: conv, 2: fc, 3: act}, mid_data_size, partition point]
                0: [13, 3, 24, 15346630656, 123633664, 26208256, 4818272, 0],
                1: [12, 3, 23, 15259926528, 123633664, 22996992, 102761824, 1],
                2: [11, 3, 22, 13410238464, 123633664, 19785728, 102761824, 2],
                3: [11, 3, 21, 13410238464, 123633664, 16574464, 25691488, 3],
                4: [10, 3, 20, 12485394432, 123633664, 13363200, 51381600, 4],
                5: [9, 3, 19, 10635706368, 123633664, 10151936, 51381600, 5],
                6: [9, 3, 18, 10635706368, 123633664, 8546304, 12846432, 6],
                7: [8, 3, 17, 9710862336, 123633664, 6940672, 25691496, 7],
                8: [7, 3, 16, 7861174272, 123633664, 5335040, 25691496, 8],
                9: [6, 3, 15, 6011486208, 123633664, 4532224, 25691496, 9],
                10: [6, 3, 14, 6011486208, 123633664, 3729408, 6423912, 10],
                11: [5, 3, 13, 5086642176, 123633664, 2926592, 12846440, 11],
                12: [4, 3, 12, 3236954112, 123633664, 2123776, 12846440, 12],
                13: [3, 3, 11, 1387266048, 123633664, 1320960, 12846440, 13],
                14: [3, 3, 10, 1387266048, 123633664, 919552, 3212648, 14],
                15: [2, 3, 9, 924844032, 123633664, 518144, 3212648, 15],
                16: [1, 3, 8, 462422016, 123633664, 417792, 3212648, 16],
                17: [0, 3, 7, 0, 123633664, 317440, 3212648, 17],
                18: [0, 3, 6, 0, 123633664, 217088, 3212648, 18],
                19: [0, 3, 4, 0, 123633664, 16384, 804200, 19], #flatten and avpool
                20: [0, 2, 2, 0, 20873216, 12288, 804200, 20],
                21: [0, 1, 0, 0, 4096000, 0, 132416, 21],
                22: [0, 0, 0, 0, 0, 0, 0, 22]
                }

yolo_info_de = {
                0: [9, 0, 22, 3537437696, 0, 28640768, 16614800, 0],
                1: [8, 0, 22, 3462677504, 0, 28640768, 88606096, 1],
                2: [8, 0, 21, 3462677504, 0, 23102976, 88606096, 2],
                3: [8, 0, 20, 3462677504, 0, 17565184, 88606096, 3],
                4: [8, 0, 19, 3462677504, 0, 14796288, 22152576, 4],
                5: [7, 0, 19, 3263316992, 0, 14796288, 44303744, 5],
                6: [7, 0, 18, 3263316992, 0, 12027392, 44303744, 6],
                7: [7, 0, 17, 3263316992, 0, 9258496, 44303744, 7],
                8: [7, 0, 16, 3263316992, 0, 7874048, 11076992, 8],
                9: [6, 0, 16, 3063956480, 0, 7874048, 22152576, 9],
                10: [6, 0, 15, 3063956480, 0, 6489600, 22152576, 10],
                11: [6, 0, 14, 3063956480, 0, 5105152, 22152576, 11],
                12: [6, 0, 13, 3063956480, 0, 4412928, 5539200, 12],
                13: [5, 0, 13, 2864595968, 0, 4412928, 11076992, 13],
                14: [5, 0, 12, 2864595968, 0, 3720704, 11076992, 14],
                15: [5, 0, 11, 2864595968, 0, 3028480, 11076992, 15],
                16: [5, 0, 10, 2864595968, 0, 2682368, 2770304, 16],
                17: [4, 0, 10, 2665235456, 0, 2682368, 5539208, 17],
                18: [4, 0, 9, 2665235456, 0, 2336256, 5539208, 18],
                19: [4, 0, 8, 2665235456, 0, 1990144, 5539208, 19],
                20: [4, 0, 7, 2665235456, 0, 1817088, 1385864, 20],
                21: [3, 0, 7, 2465874944, 0, 1817088, 2770312, 21],
                22: [3, 0, 6, 2465874944, 0, 1644032, 2770312, 22],
                23: [3, 0, 5, 2465874944, 0, 1470976, 2770312, 23],
                24: [3, 0, 4, 2465874944, 0, 1384448, 2770312, 24],
                25: [2, 0, 4, 1668432896, 0, 1384448, 5539208, 25],
                26: [2, 0, 3, 1668432896, 0, 1038336, 5539208, 26],
                27: [2, 0, 2, 1668432896, 0, 692224, 5539208, 27],
                28: [1, 0, 2, 73548800, 0, 692224, 2770312, 28],
                29: [1, 0, 1, 73548800, 0, 346112, 2770312, 29],
                30: [1, 0, 0, 73548800, 0, 0, 2770312, 30],
                31: [0, 0, 0, 0, 0, 0, 0, 31]
                }
resnet_info_de={
            0: [53,1,121,8235158519,20480,49053872,4816896,0],
            1: [52,1,119,7999933431,20480,45842608,25690112,1],
            2: [52,1,118,7999933431,20480,44036272,6422528,2],
            3: [48,1,110,7539518455,20480,36810928,25690112,3],
            4: [45,1,103,7103990775,20480,31155216,25690112,4],
            5: [42,1, 96,6668463095,20480,25499504,25690112,5],
            6: [38,1, 88,5924754423,20480,20673608,12845056,6],
            7: [35,1, 81,5488624631,20480,17863752,12845056,7],
            8: [32,1, 74,5052494839,20480,15053896,12845056,8],
            9: [29,1, 67,4616365047,20480,12244040,12845056,9],
           10: [25,1, 59,3872004096,20480, 9835592, 6422528,10],
           11: [22,1, 52,3408573248,20480, 8430664, 6422528,11],
           12: [19,1, 45,2945142400,20480, 7025736, 6422528,12],
           13: [16,1, 38,2481711552,20480, 5620808, 6422528,13],
           14: [13,1, 31,2018280704,20480, 4215880, 6422528,14],
           15: [10,1, 24,1581849856,20480, 2810952, 6422528,15],
           16: [ 6,1, 16, 837162752,20480, 1506376, 3221264,16],
           17: [ 3,1,  9, 436581376,20480,  803912, 3221264,17],
           18: [ 0,1,  1,         0,20480,  104448, 3221264,18],
           19: [ 0,1,  0,         0,20480,       0,   65536,19],
           20: [ 0,0,  0,         0,    0,       0,       0,20]
            }
alexnet_info_de={
            0: [5,3,12,591024672,13641728,880736,4816896,0],
            1: [4,3,11,485609472,13641728,735536,4646400,1],
            2: [4,3,10,485609472,13641728,420608,1119744,2],
            3: [3,3, 9,261660672,13641728,327296,2985984,3],
            4: [3,3, 8,261660672,13641728,132608, 692224,4],
            5: [2,3, 7,186900480,13641728,100160,1038336,5],
            6: [1,3, 6, 74760192,13641728, 67712,1038336,6],
            7: [0,3, 5, 13641728,13641728, 46080, 692224,7],
            8: [0,3, 2,        0,13641728, 46080, 147456,8],
            9: [0,2, 1,        0, 4204544,  2080,  65536,9],
            10:[0,1, 0,        0,   10240,     0,  65536,10],
            11:[0,0, 0,        0,       0,     0,      0,11]
            }

mobilenet_info_de={
0	:	[	49	,	1	,	100	,	720271776	,	12800	,	19703488	,	4816896	,	0	]	,
1	:	[	48	,	1	,	98	,	698997152	,	12800	,	18499264	,	12845056	,	1	]	,
2	:	[	46	,	1	,	95	,	679528864	,	12800	,	16893632	,	6422528	,	2	]	,
3	:	[	45	,	1	,	90	,	622704544	,	12800	,	12227264	,	2408448	,	3	]	,
4	:	[	42	,	1	,	84	,	572202400	,	12800	,	9291968	,	2408448	,	4	]	,
5	:	[	39	,	1	,	79	,	541858464	,	12800	,	7548352	,	802816	,	5	]	,
6	:	[	36	,	1	,	73	,	522609696	,	12800	,	6588736	,	802816	,	6	]	,
7	:	[	33	,	1	,	67	,	501253536	,	12800	,	5629120	,	802816	,	7	]	,
8	:	[	30	,	1	,	62	,	485121952	,	12800	,	5039552	,	401408	,	8	]	,
9	:	[	27	,	1	,	56	,	460598432	,	12800	,	4550336	,	401408	,	9	]	,
10	:	[	24	,	1	,	50	,	436074912	,	12800	,	4061120	,	401408	,	10	]	,
11	:	[	21	,	1	,	44	,	411551392	,	12800	,	3571904	,	401408	,	11	]	,
12	:	[	18	,	1	,	39	,	375443488	,	12800	,	3082688	,	602112	,	12	]	,
13	:	[	15	,	1	,	33	,	309756832	,	12800	,	2348864	,	602112	,	13	]	,
14	:	[	12	,	1	,	27	,	239328544	,	12800	,	1615040	,	602112	,	14	]	,
15	:	[	9	,	1	,	22	,	200641280	,	12800	,	1176000	,	250880	,	15	]	,
16	:	[	6	,	1	,	16	,	154738080	,	12800	,	870240	,	250880	,	16	]	,
17	:	[	3	,	1	,	10	,	106294720	,	12800	,	564480	,	250880	,	17	]	,
18	:	[	1	,	1	,	5	,	40078080	,	12800	,	250880	,	501760	,	18	]	,
19	:	[	0	,	1	,	3	,	0	,	12800	,	62720	,	2007040	,	19	]	,
20	:	[	0	,	1	,	1	,	0	,	12800	,	0	,	40960	,	20	]	,
21	:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	,	21	]	,

    }


vgg_info_df={
    0	:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	]	,
    1	:	[	1	,	0	,	1	,	86704128	,	0	,	3211264	,	1	]	,
    2	:	[	2	,	0	,	2	,	1936392192	,	0	,	6422528	,	2	]	,
    3	:	[	2	,	0	,	3	,	1936392192	,	0	,	9633792	,	3	]	,
    4	:	[	3	,	0	,	4	,	2861236224	,	0	,	12845056	,	4	]	,
    5	:	[	4	,	0	,	5	,	4710924288	,	0	,	16056320	,	5	]	,
    6	:	[	4	,	0	,	6	,	4710924288	,	0	,	17661952	,	6	]	,
    7	:	[	5	,	0	,	7	,	5635768320	,	0	,	19267584	,	7	]	,
    8  	:	[	6	,	0	,	8	,	7485456384	,	0	,	20873216	,	8	]	,
    9	:	[	7	,	0	,	9	,	9335144448	,	0	,	21676032	,	9	]	,
    10	:	[	7	,	0	,	10	,	9335144448	,	0	,	22478848	,	10	]	,
    11	:	[	8	,	0	,	11	,	10259988480	,	0	,	23281664	,	11	]	,
    12	:	[	9	,	0	,	12	,	12109676544	,	0	,	24084480	,	12	]	,
    13	:	[	10	,	0	,	13	,	13959364608	,	0	,	24887296	,	13	]	,
    14	:	[	10	,	0	,	14	,	13959364608	,	0	,	25288704	,	14	]	,
    15	:	[	11	,	0	,	15	,	14421786624	,	0	,	25690112	,	15	]	,
    16	:	[	12	,	0	,	16	,	14884208640	,	0	,	25790464	,	16	]	,
    17	:	[	13	,	0	,	17	,	15346630656	,	0	,	25890816	,	17	]	,
    18	:	[	13	,	0	,	18	,	15346630656	,	0	,	25991168	,	18	]	,
    19	:	[	13	,	0	,	20	,	15346630656	,	0	,	26191872	,	19	]	,
    20	:	[	13	,	1	,	22	,	15346630656	,	102760448	,	26195968	,	20	]	,
    21	:	[	13	,	2	,	24	,	15346630656	,	119537664	,	26208256	,	21	]	,
    22	:	[	13	,	3	,	24	,	15346630656	,	123633664	,	26208256	,	22	]	,
    }

yolo_info_df={
0	:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	]	,
1	:	[	1	,	0	,	0	,	74760192	,	0	,	0	,	1	]	,
2	:	[	1	,	0	,	1	,	74760192	,	0	,	5537792	,	2	]	,
3	:	[	1	,	0	,	2	,	74760192	,	0	,	11075584	,	3	]	,
4	:	[	1	,	0	,	3	,	74760192	,	0	,	13844480	,	4	]	,
5	:	[	2	,	0	,	3	,	274120704	,	0	,	13844480	,	5	]	,
6	:	[	2	,	0	,	4	,	274120704	,	0	,	16613376	,	6	]	,
7	:	[	2	,	0	,	5	,	274120704	,	0	,	19382272	,	7	]	,
8	:	[	2	,	0	,	6	,	274120704	,	0	,	20766720	,	8	]	,
9	:	[	3	,	0	,	6	,	473481216	,	0	,	20766720	,	9	]	,
10	:	[	3	,	0	,	7	,	473481216	,	0	,	22151168	,	10	]	,
11	:	[	3	,	0	,	8	,	473481216	,	0	,	23535616	,	11	]	,
12	:	[	3	,	0	,	9	,	473481216	,	0	,	24227840	,	12	]	,
13	:	[	4	,	0	,	9	,	672841728	,	0	,	24227840	,	13	]	,
14	:	[	4	,	0	,	10	,	672841728	,	0	,	24920064	,	14	]	,
15	:	[	4	,	0	,	11	,	672841728	,	0	,	25612288	,	15	]	,
16	:	[	4	,	0	,	12	,	672841728	,	0	,	25958400	,	16	]	,
17	:	[	5	,	0	,	12	,	872202240	,	0	,	25958400	,	17	]	,
18	:	[	5	,	0	,	13	,	872202240	,	0	,	26304512	,	18	]	,
19	:	[	5	,	0	,	14	,	872202240	,	0	,	26650624	,	19	]	,
20	:	[	5	,	0	,	15	,	872202240	,	0	,	26823680	,	20	]	,
21	:	[	6	,	0	,	15	,	1071562752	,	0	,	26823680	,	21	]	,
22	:	[	6	,	0	,	16	,	1071562752	,	0	,	26996736	,	22	]	,
23	:	[	6	,	0	,	17	,	1071562752	,	0	,	27169792	,	23	]	,
24	:	[	6	,	0	,	18	,	1071562752	,	0	,	27256320	,	24	]	,
25	:	[	7	,	0	,	18	,	1869004800	,	0	,	27256320	,	25	]	,
26	:	[	7	,	0	,	19	,	1869004800	,	0	,	27602432	,	26	]	,
27	:	[	7	,	0	,	20	,	1869004800	,	0	,	27948544	,	27	]	,
28	:	[	8	,	0	,	20	,	3463888896	,	0	,	27948544	,	28	]	,
29	:	[	8	,	0	,	21	,	3463888896	,	0	,	28294656	,	29	]	,
30	:	[	8	,	0	,	22	,	3463888896	,	0	,	28640768	,	30	]	,
31	:	[	9	,	0	,	22	,	3537437696	,	0	,	28640768	,	31	]	

}
resnet_info_df={
    0	:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	,	]	,
    1	:	[	1	,	0	,	2	,	235225088	,	0	,	3211264	,	1		]	,
    2	:	[	1	,	0	,	3	,	235225088	,	0	,	5017600	,	2		]	,
    3	:	[	5	,	0	,	11	,	695640064	,	0	,	12242944	,	3		]	,
    4	:	[	8	,	0	,	18	,	1131167744	,	0	,	17898656	,	4		]	,
    5	:	[	11	,	0	,	25	,	1566695424	,	0	,	23554368	,	5		]	,
    6	:	[	15	,	0	,	33	,	2310404096	,	0	,	28380264	,	6		]	,
    7	:	[	18	,	0	,	40	,	2746533888	,	0	,	31190120	,	7		]	,
    8	:	[	21	,	0	,	47	,	3182663680	,	0	,	33999976	,	8		]	,
    9	:	[	24	,	0	,	54	,	3618793472	,	0	,	36809832	,	9		]	,
    10	:	[	28	,	0	,	62	,	4363154423	,	0	,	39218280	,	10		]	,
    11	:	[	31	,	0	,	69	,	4826585271	,	0	,	40623208	,	11		]	,
    12	:	[	34	,	0	,	76	,	5290016119	,	0	,	42028136	,	12		]	,
    13	:	[	37	,	0	,	83	,	5753446967	,	0	,	43433064	,	13		]	,
    14	:	[	40	,	0	,	90	,	6216877815	,	0	,	44837992	,	14		]	,
    15	:	[	43	,	0	,	97	,	6653308663	,	0	,	46242920	,	15		]	,
    16	:	[	47	,	0	,	105	,	7397995767	,	0	,	47547496	,	16		]	,
    17	:	[	50	,	0	,	112	,	7798577143	,	0	,	48249960	,	17		]	,
    18	:	[	53	,	0	,	120	,	8235158519	,	0	,	48949424	,	18		]	,
    19	:	[	53	,	0	,	121	,	8235158519	,	0	,	49053872	,	19		]	,
    20	:	[	53	,	1	,	121	,	8235158519	,	20480	,	49053872	,	20		]	,
}
alexnet_info_df={
    0	:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	]	,
    1	:	[	1	,	0	,	1	,	105415200	,	0	,	145200	,	170496	,	1	]	,
    2	:	[	1	,	0	,	2	,	105415200	,	0	,	460128	,	3697152	,	2	]	,
    3	:	[	2	,	0	,	3	,	329364000	,	0	,	553440	,	1830912	,	3	]	,
    4	:	[	2	,	0	,	4	,	329364000	,	0	,	748128	,	4124672	,	4	]	,
    5	:	[	3	,	0	,	5	,	404124192	,	0	,	780576	,	3778560	,	5	]	,
    6	:	[	4	,	0	,	6	,	516264480	,	0	,	813024	,	3778560	,	6	]	,
    7	:	[	5	,	0	,	7	,	577382944	,	0	,	834656	,	4124672	,	7	]	,
    8	:	[	5	,	0	,	10	,	591024672	,	0	,	834656	,	4669440	,	8	]	,
    9	:	[	5	,	1	,	11	,	591024672	,	9437184	,	878656	,	4751360	,	9	]	,
    10	:	[	5	,	2	,	12	,	591024672	,	13631488	,	880736	,	4751360	,	10	]	,
    11	:	[	5	,	3	,	12	,	591024672	,	13641728	,	880736	,	4816896	,	11	]	,
}

mobilenet_info_df={
    0:	[	0	,	0	,	0	,	0	,	0	,	0	,	0	],
    1:	[	1	,	0	,	2	,	21274624	,	0	,	1204224	,	1	],
    2:	[	3	,	0	,	5	,	40742912	,	0	,	2809856	,	2	],
    3:	[	4	,	0	,	10	,	97567232	,	0	,	7476224	,	3	],
    4:	[	7	,	0	,	16	,	148069376	,	0	,	10411520	,	4	],
    5:	[	10	,	0	,	21	,	178413312	,	0	,	12155136	,	5	],
    6:	[	13	,	0	,	27	,	197662080	,	0	,	13114752	,	6	],
    7:	[	16	,	0	,	33	,	219018240	,	0	,	14074368	,	7	],
    8:	[	19	,	0	,	38	,	235149824	,	0	,	14663936	,	8	],
    9:	[	22	,	0	,	44	,	259673344	,	0	,	15153152	,	9	],
    10:	[	25	,	0	,	50	,	284196864	,	0	,	15642368	,	10	],
    11:	[	28	,	0	,	56	,	308720384	,	0	,	16131584	,	11	],
    12:	[	31	,	0	,	61	,	344828288	,	0	,	16620800	,	12	],
    13:	[	34	,	0	,	67	,	410514944	,	0	,	17354624	,	13	],
    14:	[	37	,	0	,	73	,	480943232	,	0	,	18088448	,	14	],
    15:	[	40	,	0	,	78	,	519630496	,	0	,	18527488	,	15	],
    16:	[	43	,	0	,	84	,	565533696	,	0	,	18833248	,	16	],
    17:	[	46	,	0	,	90	,	613977056	,	0	,	19139008	,	17	],
    18:	[	48	,	0	,	95	,	680193696	,	0	,	19452608	,	18	],
    19:	[	49	,	0	,	97	,	720271776	,	0	,	19640768	,	19	],
    20:	[	49	,	0	,	99	,	720271776	,	0	,	19703488	,	20	],
    21:	[	49	,	1	,	100	,	720271776	,	12800	,	19703488	,	21	],
}


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('--rtsp', dest='use_rtsp',
    #                     help='use IP CAM (remember to also set --uri)',
    #                     action='store_true')
    # parser.add_argument('--uri', dest='rtsp_uri',
    #                     help='RTSP URI, e.g. rtsp://192.168.1.64:554',
    #                     default=None, type=str)
    # parser.add_argument('--latency', dest='rtsp_latency',
    #                     help='latency in ms for RTSP [200]',
    #                     default=200, type=int)
    # parser.add_argument('--usb', dest='use_usb',
    #                     help='use USB webcam (remember to also set --vid)',
    #                     action='store_true')
    # parser.add_argument('--vid', dest='video_dev',
    #                     help='device # of USB webcam (/dev/video?) [1]',
    #                     default=1, type=int)
    # parser.add_argument('--width', dest='image_width',
    #                     help='image width',
    #                     default=640, type=int)
    # parser.add_argument('--height', dest='image_height',
    #                     help='image height',
    #                     default=480, type=int)
    parser.add_argument('--dnn_model', dest='dnn_model',
                        help='vgg, yolo,resnet,alexnet,mobilenetv2',
                        default='vgg', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='192.168.31.33', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8889, type=int)
    # parser.add_argument('--anotherdnn', dest='another_dnn_model',
    #                     help='vgg, yolo,resnet',
    #                     default='vgg', type=str)
    parser.add_argument('--breaktime', dest='breaktime',
                        default=200, type=int)
    parser.add_argument('--netnumber',dest="netnumber",default=0,type=int)

                        

    args = parser.parse_args()
    return args

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)640, height=(int)480,'
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo 0')

def prepare_image_vgg(frame):
    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def prepare_image_yolo(frame):
    min_img_size = 416
    image = cv2.resize(frame, (min_img_size, min_img_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    img = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img

def prepare_image_resnet(frame):
    transform_pipeline = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def prepare_image_alexnet(frame):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img=data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def prepare_image_mobilenetv2(frame):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img=data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def show_preds(img, label, averageTime):
    x = 10
    y = 50

    font = cv2.FONT_HERSHEY_PLAIN

    pred = '{:20s}'.format(label[1])
    cv2.putText(img, pred, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)
    y += 30
    timeShow = 'AvgTime: {:.4f}'.format(averageTime)
    cv2.putText(img, timeShow, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)

    return img

def getVggLabelDic(class_file):
    with open(class_file, "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}
    return labels

def decodePrediction_vgg(res, labels):
    res = torch.autograd.Variable(res)
    label_index = torch.argmax(res).item()
    return labels[label_index]

def getActualDelay(action, model, preprocessed_image, totallayerNo, communication,device):
    if action == totallayerNo - 1: # local mobile process
        #w
        start_t=time.time()
        prediction = model(preprocessed_image.to(device))
        end_t=time.time()
        #print(str(0),action)
        #prediction = model(preprocessed_image.cuda())
        #return 0, prediction.item()
        return 0, prediction.data,end_t-start_t,0,0
    else:
        #w
        start_t=time.time()
        intermediate_output = model(preprocessed_image.to(device), server=False, partition=action)
        end_t=time.time()
        #intermediate_output = model((preprocessed_image.cuda()), server=False, partition=action)

    data_to_server = [action, intermediate_output.data]
    del intermediate_output

    start_time = time.time()
    #w
    #print(str(start_time),str(action))

    send_time=communication.send_msg(data_to_server)

    result= communication.receive_msg()

    #communication.close_channel()
    end_time = time.time()

    return end_time - start_time,  result, end_t-start_t, send_time, start_time

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    netnumber=args.netnumber

    #w
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device="cpu"
    print("using {} device.".format(device))
    #device="cuda:0"

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        #更改为w cpu/gpu延时
        labels = getVggLabelDic(str(parent_parent_dir)+'/models/imagenet_class_index.json')
        partitionInfo_e = vgg_info_de
        partitionInfo_f = vgg_info_df
        acttotal=22
    elif args.dnn_model == 'yolo':
        model = tinyYolo()
        model.eval()
        #frontEndDelay = load_obj('models/yoloFrontEndDelay')
        labels = load_class_names(str(parent_parent_dir)+'/models/voc.names')
        partitionInfo_e = yolo_info_de
        partitionInfo_f = yolo_info_df
        acttotal=31
    elif args.dnn_model == 'alexnet':
        model=alexnet(num_class=5)
        model.eval()
        with open(str(parent_parent_dir)+"/models/alexnetclass_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        #frontEndDelay = load_obj('models/alexnetFrontEndDelay')
        partitionInfo_e = alexnet_info_de
        partitionInfo_f = alexnet_info_df
        acttotal=11
    #elif args.dnn_model == '':
    elif args.dnn_model == 'mobilenetv2':
        model=mobilenetv2()
        model.eval()
        with open(str(parent_parent_dir)+"/models/mobile_class_indices.json", "r") as read_file:
            class_idx = json.load(read_file)
        #frontEndDelay=load_obj('models/mobilenetv2FrontEndDelay')
        partitionInfo_e = mobilenet_info_de
        partitionInfo_f = mobilenet_info_df
        acttotal=21
    elif args.dnn_model == 'resnet':
        # create model
        model = resnet50(num_classes=5)
        weights_path =str(parent_parent_dir)+ "/models/resNet50_e3_1_0.pth"
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        # read class_indict
        json_path = str(parent_parent_dir)+'/models/class_indices.json'
        #assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        #frontEndDelay = load_obj('models/resnet50FrontEndDelay_cpu')
        #labels = load_class_names('models/voc.names')
        partitionInfo_e = resnet_info_de
        partitionInfo_f = resnet_info_df
        acttotal=20
    else:
        print("please input the right net name")
        assert False
    
    #w
    #model.cuda()
    model.to(device)
    Action_num = len(partitionInfo_e)

    muLinUCB = muLinUCB(mu=0.25,layerInfof=partitionInfo_f, layerInfoe=partitionInfo_e)
    communication = clientCommunication(args.host, args.port)
    communication.connect()

    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN

    total_time = 0
    total_frame_num = 0
    currentFrameNum = 0
    keyflag = False
    #KeyFrame = KeyFrameDetection(threshold=0.8)

    #w,延时记录文件
    myfile=open(str(parent_dir)+"/latency/latency1.txt",'w')
    myfile.write(f'starttime: {str(time.time())}\n')
    myfile.write(f'actual_delay_all{"":9s}predict_delay_all{"":5s}actual_df{"":13s}predict_df{"":9s}tempf_1{"":15s}tempf_2{"":7s}\
                 actual_de{"":13s}predict_de{"":9s}tempe_1{"":15s}tempe_2{"":15s}force{"":1s}ppoint{"":11s}estimationtime{"":9s}updatetime{"":9s}totaltime{"":11s}timecounter')
    startime_all=time.time()
    totaltime_start=time.time()
    timecounter=0

    real_latency_offrame=0
    #lock=threading.Lock()
    oldtime_cpu=time.time()
    while True:
     try:
        totaltime=time.time()-totaltime_start
        totaltime_start=time.time()

        partitionPoint_nothing=communication.receive_msg()

        img=cv2.imread(str(parent_parent_dir)+"/models/tulip.jpg")

        if args.dnn_model == 'vgg':
            preprocessed_image = prepare_image_vgg(img)
        elif args.dnn_model == 'yolo':
            preprocessed_image = prepare_image_yolo(img)
        elif args.dnn_model == 'alexnet':
            preprocessed_image = prepare_image_alexnet(img)
        elif args.dnn_model == 'mobilenetv2':
            preprocessed_image = prepare_image_mobilenetv2(img)
        else:
            preprocessed_image = prepare_image_resnet(img)

        # doubling trick is here.
        currentFrameNum = currentFrameNum + 1
        if muLinUCB.updateDoublingTrickFrameNum(currentFrameNum):
            currentFrameNum = 0

        # key frame detection
        if total_frame_num == 0:
            keyflag = False
            old_frame = np.copy(img)
        else:
            #keyflag = KeyFrame.compare_images(old_frame, img)
            old_frame = np.copy(img)

        #predict_delay 预测总延时，

        estimatimestart=time.time()
        partitionPoint,predict_delay,force,predict_df,tempf_1,tempf_2,predict_de,tempe_1,tempe_2, = muLinUCB.getEstimationAction(keyflag, currentFrameNum,total_frame_num)
        #partitionPoint_0,predict_delay,force,predict_de,tempe_1,tempe_2 = muLinUCB.getEstimationAction(keyflag, currentFrameNum,total_frame_num)

        end2endtime_start = time.time()
        actual_de, res, actual_df ,send_time,start_time= getActualDelay(partitionPoint, model, preprocessed_image, Action_num, communication,device)
        end2endtime_end = time.time()

        #w 如果分割点为最后分割点,给服务器发送数据，服务器记录。
        if partitionPoint==acttotal:
            communication.send_msg([partitionPoint,0])
            #communication.close_channel()
        
        total_frame_num = total_frame_num + 1
        total_time = total_time + (end2endtime_end - end2endtime_start)
        average_time = total_time/total_frame_num

        # update A and b
        updatastart=time.time()
        if(total_frame_num!=1):
            muLinUCB.updateA_b(partitionPoint, actual_de,actual_df)
        else:
            print("已开始一帧")
        updataend=time.time()

        #w增加，记录延时
        myfile.write("\n")
        myfile.write(f'{str(end2endtime_end-end2endtime_start):<21}')
        myfile.write(f'{str(predict_delay.item()):<22}')
        myfile.write(f'{str(actual_df):<23}')
        myfile.write(f'{str(predict_df.item()):<25}')
        myfile.write(f'{str(tempf_1):<25}{str(tempf_2):<25}')
        myfile.write(f'{str(actual_de):<23}')
        myfile.write(f'{str(predict_de.item()):<23}')
        myfile.write(f'{str(tempe_1):<23}{str(tempe_2):<23}')
        myfile.write(f'{str(force):<7}')
        myfile.write(f'{str(partitionPoint):<5}')
        myfile.write(f'{str(end2endtime_start-estimatimestart):<23}')
        myfile.write(f'{str(updataend-updatastart):<23}')
        myfile.write(f'{str(totaltime):<23}')
        myfile.write(f'{str(timecounter):<23}')


        #cputotal=0
        cputotal=psutil.cpu_percent(interval=None)
        #增加计时
        cpu_timelenth=time.time()-oldtime_cpu        
        oldtime_cpu=time.time()
        cpuutotalsum=cputotal*cpu_timelenth

        #发送总延时数据，和cpu利用率数据。

        real_latency_offrame=end2endtime_end-end2endtime_start
        ladata=[real_latency_offrame,partitionPoint,cpuutotalsum,cpu_timelenth,cputotal]  ##一帧总延时，分割点，设备延时，传输延时，边缘延时，cpuu,cputime,cputotal,memaver
        communication.send_msg(ladata)  #

        ##定时 1min    50max
        if total_frame_num==1:
            print("已完成第一帧",netnumber)
        if(time.time()-startime_all>(timecounter+1)*60):
            timecounter+=1
            print(timecounter,'min  total_frame_num: ',total_frame_num,"netnumber: ",netnumber)
            if(args.breaktime==timecounter):
                print('break at ',timecounter,'min net ',netnumber," ",args.dnn_model)
                break

        ##定时 1min    50max
        # if total_frame_num==1:
        #     print("已完成第一帧")
        # if(time.time()-startime_all>(timecounter+1)*60):
        #     timecounter+=1
        #     print(timecounter,'min  total_frame_num: ',total_frame_num)
        #     if(args.breaktime==timecounter):
        #         print('break at ',timecounter,'min')
        #         break
        # elif(time.time()-startime_all>3000):
        #     print('break at 50min')   
        #     break

        #w
        # bandwidthlist=[40, 25, 20, 35, 50, 30, 15, 5, 45, 10]
        # if total_frame_num==1:
        #     print("已完成第一帧")
        #     res_ch=os.system("echo 6 | sudo -S /home/w/wondershaper/wondershaper -c -a wlo1 ")
        #     print(res_ch)
        # if(time.time()-startime_all>60*(timecounter+1)):
        #     if timecounter<10:
        #         bandwidth=1024*bandwidthlist[timecounter]
        #         timecounter+=1
        #         print(str(timecounter),'min  total_frame_num: ',total_frame_num)
        #         res_ch=os.system("echo 6 | sudo -S /home/w/wondershaper/wondershaper -c -a wlo1 ")  
        #         strwidth="echo 6 | sudo -S /home/w/wondershaper/wondershaper -a wlo1 -d "+str(bandwidth)+" -u "+str(bandwidth)   
        #         res_ch=os.system(strwidth)
        #         print("res_ch",res_ch,'  bandwidth ',bandwidth)
        #     else:
        #         timecounter+=1
        #         print(str(timecounter),'min  total_frame_num: ',total_frame_num)
        #         res_ch=os.system("echo 6 | sudo -S /home/w/wondershaper/wondershaper -c -a wlo1 ")
        #         break
        # k or u
        # bandwidthlist=[40, 25, 20, 35, 50, 30, 15, 5, 45, 10]
        # if total_frame_num==1:
        #     print("已完成第一帧")
        #     res_ch=os.system("echo 6 | sudo -S wondershaper clear wlan0 ")
        # if(time.time()-startime_all>60*(timecounter+1)):
        #     if timecounter<10:
        #         bandwidth=1024*bandwidthlist[timecounter]
        #         timecounter+=1
        #         print(str(timecounter),'min  total_frame_num: ',total_frame_num)
        #         res_ch=os.system("echo 6 | sudo -S wondershaper clear wlan0 ")  
        #         strwidth="echo 6 | sudo -S wondershaper wlan0 "+str(bandwidth)+" "+str(bandwidth)
        #         res_ch=os.system(strwidth)
        #         print("res_ch",res_ch,'  bandwidth ',bandwidth)
        #     else:
        #         timecounter+=1
        #         print(str(timecounter),'min  total_frame_num: ',total_frame_num)
        #         res_ch=os.system("echo 6 | sudo -S wondershaper clear wlan0 ")
        #         break


     except KeyboardInterrupt or TypeError or OSError:
        
            #发送结束
            communication.send_msg("close")
            communication.close_channel()
            myfile.close()
            print(netnumber," ",args.dnn_model,' average_time:',total_time/total_frame_num)
            # print('total_time ',total_time)
            # print("startime_all ",startime_all)
            # print("endtime_all",time.time())
            del model

    #w
    #print('average_time:',total_time/total_frame_num)
    communication.send_msg("close")
    communication.close_channel()
    myfile.close()
    print(netnumber," ",args.dnn_model,' average_time:',total_time/total_frame_num)
    # print('total_time ',total_time)
    # print("startime_all ",startime_all)
    # print("endtime_all",time.time())
    del model

    #cap.release()
    cv2.destroyAllWindows()
