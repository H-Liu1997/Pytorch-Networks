# --------------------------------------------------------------------------- #
# DenseNet, CVPR2017 bestpaper, https://arxiv.org/abs/1608.06993
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


# ------------------------------- background -------------------------------- #
#
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# main idea: Dense short cut in channel level, not elements level
# parameters: 35M dense264, 7M dense121, BN+ReLU
# sgd+momentum+Nesterov 1e-1 0.9 divide 10 * 2 
# batch size 256
# weight decay 1e-4
# input: resize and crop samll side to 256×256 then augment to 224
# output: linear 1000 + softmax
# TODO: Check details in small dense net
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------ dense121 model summary --------------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#        BatchNorm2d-5           [-1, 64, 56, 56]             128
#               ReLU-6           [-1, 64, 56, 56]               0
#             Conv2d-7          [-1, 128, 56, 56]           8,192
#        BatchNorm2d-8          [-1, 128, 56, 56]             256
#               ReLU-9          [-1, 128, 56, 56]               0
#            Conv2d-10           [-1, 32, 56, 56]          36,864
#        DenseLayer-11           [-1, 96, 56, 56]               0
#       BatchNorm2d-12           [-1, 96, 56, 56]             192
#              ReLU-13           [-1, 96, 56, 56]               0
#            Conv2d-14          [-1, 128, 56, 56]          12,288
#       BatchNorm2d-15          [-1, 128, 56, 56]             256
#              ReLU-16          [-1, 128, 56, 56]               0
#            Conv2d-17           [-1, 32, 56, 56]          36,864
#        DenseLayer-18          [-1, 128, 56, 56]               0
#       BatchNorm2d-19          [-1, 128, 56, 56]             256
#              ReLU-20          [-1, 128, 56, 56]               0
#            Conv2d-21          [-1, 128, 56, 56]          16,384
#       BatchNorm2d-22          [-1, 128, 56, 56]             256
#              ReLU-23          [-1, 128, 56, 56]               0
#            Conv2d-24           [-1, 32, 56, 56]          36,864
#        DenseLayer-25          [-1, 160, 56, 56]               0
#       BatchNorm2d-26          [-1, 160, 56, 56]             320
#              ReLU-27          [-1, 160, 56, 56]               0
#            Conv2d-28          [-1, 128, 56, 56]          20,480
#       BatchNorm2d-29          [-1, 128, 56, 56]             256
#              ReLU-30          [-1, 128, 56, 56]               0
#            Conv2d-31           [-1, 32, 56, 56]          36,864
#        DenseLayer-32          [-1, 192, 56, 56]               0
#       BatchNorm2d-33          [-1, 192, 56, 56]             384
#              ReLU-34          [-1, 192, 56, 56]               0
#            Conv2d-35          [-1, 128, 56, 56]          24,576
#       BatchNorm2d-36          [-1, 128, 56, 56]             256
#              ReLU-37          [-1, 128, 56, 56]               0
#            Conv2d-38           [-1, 32, 56, 56]          36,864
#        DenseLayer-39          [-1, 224, 56, 56]               0
#       BatchNorm2d-40          [-1, 224, 56, 56]             448
#              ReLU-41          [-1, 224, 56, 56]               0
#            Conv2d-42          [-1, 128, 56, 56]          28,672
#       BatchNorm2d-43          [-1, 128, 56, 56]             256
#              ReLU-44          [-1, 128, 56, 56]               0
#            Conv2d-45           [-1, 32, 56, 56]          36,864
#        DenseLayer-46          [-1, 256, 56, 56]               0
#        DenseBlock-47          [-1, 256, 56, 56]               0
#       BatchNorm2d-48          [-1, 256, 56, 56]             512
#              ReLU-49          [-1, 256, 56, 56]               0
#            Conv2d-50          [-1, 128, 56, 56]          32,768
#         AvgPool2d-51          [-1, 128, 28, 28]               0
#       BatchNorm2d-52          [-1, 128, 28, 28]             256
#              ReLU-53          [-1, 128, 28, 28]               0
#            Conv2d-54          [-1, 128, 28, 28]          16,384
#       BatchNorm2d-55          [-1, 128, 28, 28]             256
#              ReLU-56          [-1, 128, 28, 28]               0
#            Conv2d-57           [-1, 32, 28, 28]          36,864
#        DenseLayer-58          [-1, 160, 28, 28]               0
#       BatchNorm2d-59          [-1, 160, 28, 28]             320
#              ReLU-60          [-1, 160, 28, 28]               0
#            Conv2d-61          [-1, 128, 28, 28]          20,480
#       BatchNorm2d-62          [-1, 128, 28, 28]             256
#              ReLU-63          [-1, 128, 28, 28]               0
#            Conv2d-64           [-1, 32, 28, 28]          36,864
#        DenseLayer-65          [-1, 192, 28, 28]               0
#       BatchNorm2d-66          [-1, 192, 28, 28]             384
#              ReLU-67          [-1, 192, 28, 28]               0
#            Conv2d-68          [-1, 128, 28, 28]          24,576
#       BatchNorm2d-69          [-1, 128, 28, 28]             256
#              ReLU-70          [-1, 128, 28, 28]               0
#            Conv2d-71           [-1, 32, 28, 28]          36,864
#        DenseLayer-72          [-1, 224, 28, 28]               0
#       BatchNorm2d-73          [-1, 224, 28, 28]             448
#              ReLU-74          [-1, 224, 28, 28]               0
#            Conv2d-75          [-1, 128, 28, 28]          28,672
#       BatchNorm2d-76          [-1, 128, 28, 28]             256
#              ReLU-77          [-1, 128, 28, 28]               0
#            Conv2d-78           [-1, 32, 28, 28]          36,864
#        DenseLayer-79          [-1, 256, 28, 28]               0
#       BatchNorm2d-80          [-1, 256, 28, 28]             512
#              ReLU-81          [-1, 256, 28, 28]               0
#            Conv2d-82          [-1, 128, 28, 28]          32,768
#       BatchNorm2d-83          [-1, 128, 28, 28]             256
#              ReLU-84          [-1, 128, 28, 28]               0
#            Conv2d-85           [-1, 32, 28, 28]          36,864
#        DenseLayer-86          [-1, 288, 28, 28]               0
#       BatchNorm2d-87          [-1, 288, 28, 28]             576
#              ReLU-88          [-1, 288, 28, 28]               0
#            Conv2d-89          [-1, 128, 28, 28]          36,864
#       BatchNorm2d-90          [-1, 128, 28, 28]             256
#              ReLU-91          [-1, 128, 28, 28]               0
#            Conv2d-92           [-1, 32, 28, 28]          36,864
#        DenseLayer-93          [-1, 320, 28, 28]               0
#       BatchNorm2d-94          [-1, 320, 28, 28]             640
#              ReLU-95          [-1, 320, 28, 28]               0
#            Conv2d-96          [-1, 128, 28, 28]          40,960
#       BatchNorm2d-97          [-1, 128, 28, 28]             256
#              ReLU-98          [-1, 128, 28, 28]               0
#            Conv2d-99           [-1, 32, 28, 28]          36,864
#       DenseLayer-100          [-1, 352, 28, 28]               0
#      BatchNorm2d-101          [-1, 352, 28, 28]             704
#             ReLU-102          [-1, 352, 28, 28]               0
#           Conv2d-103          [-1, 128, 28, 28]          45,056
#      BatchNorm2d-104          [-1, 128, 28, 28]             256
#             ReLU-105          [-1, 128, 28, 28]               0
#           Conv2d-106           [-1, 32, 28, 28]          36,864
#       DenseLayer-107          [-1, 384, 28, 28]               0
#      BatchNorm2d-108          [-1, 384, 28, 28]             768
#             ReLU-109          [-1, 384, 28, 28]               0
#           Conv2d-110          [-1, 128, 28, 28]          49,152
#      BatchNorm2d-111          [-1, 128, 28, 28]             256
#             ReLU-112          [-1, 128, 28, 28]               0
#           Conv2d-113           [-1, 32, 28, 28]          36,864
#       DenseLayer-114          [-1, 416, 28, 28]               0
#      BatchNorm2d-115          [-1, 416, 28, 28]             832
#             ReLU-116          [-1, 416, 28, 28]               0
#           Conv2d-117          [-1, 128, 28, 28]          53,248
#      BatchNorm2d-118          [-1, 128, 28, 28]             256
#             ReLU-119          [-1, 128, 28, 28]               0
#           Conv2d-120           [-1, 32, 28, 28]          36,864
#       DenseLayer-121          [-1, 448, 28, 28]               0
#      BatchNorm2d-122          [-1, 448, 28, 28]             896
#             ReLU-123          [-1, 448, 28, 28]               0
#           Conv2d-124          [-1, 128, 28, 28]          57,344
#      BatchNorm2d-125          [-1, 128, 28, 28]             256
#             ReLU-126          [-1, 128, 28, 28]               0
#           Conv2d-127           [-1, 32, 28, 28]          36,864
#       DenseLayer-128          [-1, 480, 28, 28]               0
#      BatchNorm2d-129          [-1, 480, 28, 28]             960
#             ReLU-130          [-1, 480, 28, 28]               0
#           Conv2d-131          [-1, 128, 28, 28]          61,440
#      BatchNorm2d-132          [-1, 128, 28, 28]             256
#             ReLU-133          [-1, 128, 28, 28]               0
#           Conv2d-134           [-1, 32, 28, 28]          36,864
#       DenseLayer-135          [-1, 512, 28, 28]               0
#       DenseBlock-136          [-1, 512, 28, 28]               0
#      BatchNorm2d-137          [-1, 512, 28, 28]           1,024
#             ReLU-138          [-1, 512, 28, 28]               0
#           Conv2d-139          [-1, 256, 28, 28]         131,072
#        AvgPool2d-140          [-1, 256, 14, 14]               0
#      BatchNorm2d-141          [-1, 256, 14, 14]             512
#             ReLU-142          [-1, 256, 14, 14]               0
#           Conv2d-143          [-1, 128, 14, 14]          32,768
#      BatchNorm2d-144          [-1, 128, 14, 14]             256
#             ReLU-145          [-1, 128, 14, 14]               0
#           Conv2d-146           [-1, 32, 14, 14]          36,864
#       DenseLayer-147          [-1, 288, 14, 14]               0
#      BatchNorm2d-148          [-1, 288, 14, 14]             576
#             ReLU-149          [-1, 288, 14, 14]               0
#           Conv2d-150          [-1, 128, 14, 14]          36,864
#      BatchNorm2d-151          [-1, 128, 14, 14]             256
#             ReLU-152          [-1, 128, 14, 14]               0
#           Conv2d-153           [-1, 32, 14, 14]          36,864
#       DenseLayer-154          [-1, 320, 14, 14]               0
#      BatchNorm2d-155          [-1, 320, 14, 14]             640
#             ReLU-156          [-1, 320, 14, 14]               0
#           Conv2d-157          [-1, 128, 14, 14]          40,960
#      BatchNorm2d-158          [-1, 128, 14, 14]             256
#             ReLU-159          [-1, 128, 14, 14]               0
#           Conv2d-160           [-1, 32, 14, 14]          36,864
#       DenseLayer-161          [-1, 352, 14, 14]               0
#      BatchNorm2d-162          [-1, 352, 14, 14]             704
#             ReLU-163          [-1, 352, 14, 14]               0
#           Conv2d-164          [-1, 128, 14, 14]          45,056
#      BatchNorm2d-165          [-1, 128, 14, 14]             256
#             ReLU-166          [-1, 128, 14, 14]               0
#           Conv2d-167           [-1, 32, 14, 14]          36,864
#       DenseLayer-168          [-1, 384, 14, 14]               0
#      BatchNorm2d-169          [-1, 384, 14, 14]             768
#             ReLU-170          [-1, 384, 14, 14]               0
#           Conv2d-171          [-1, 128, 14, 14]          49,152
#      BatchNorm2d-172          [-1, 128, 14, 14]             256
#             ReLU-173          [-1, 128, 14, 14]               0
#           Conv2d-174           [-1, 32, 14, 14]          36,864
#       DenseLayer-175          [-1, 416, 14, 14]               0
#      BatchNorm2d-176          [-1, 416, 14, 14]             832
#             ReLU-177          [-1, 416, 14, 14]               0
#           Conv2d-178          [-1, 128, 14, 14]          53,248
#      BatchNorm2d-179          [-1, 128, 14, 14]             256
#             ReLU-180          [-1, 128, 14, 14]               0
#           Conv2d-181           [-1, 32, 14, 14]          36,864
#       DenseLayer-182          [-1, 448, 14, 14]               0
#      BatchNorm2d-183          [-1, 448, 14, 14]             896
#             ReLU-184          [-1, 448, 14, 14]               0
#           Conv2d-185          [-1, 128, 14, 14]          57,344
#      BatchNorm2d-186          [-1, 128, 14, 14]             256
#             ReLU-187          [-1, 128, 14, 14]               0
#           Conv2d-188           [-1, 32, 14, 14]          36,864
#       DenseLayer-189          [-1, 480, 14, 14]               0
#      BatchNorm2d-190          [-1, 480, 14, 14]             960
#             ReLU-191          [-1, 480, 14, 14]               0
#           Conv2d-192          [-1, 128, 14, 14]          61,440
#      BatchNorm2d-193          [-1, 128, 14, 14]             256
#             ReLU-194          [-1, 128, 14, 14]               0
#           Conv2d-195           [-1, 32, 14, 14]          36,864
#       DenseLayer-196          [-1, 512, 14, 14]               0
#      BatchNorm2d-197          [-1, 512, 14, 14]           1,024
#             ReLU-198          [-1, 512, 14, 14]               0
#           Conv2d-199          [-1, 128, 14, 14]          65,536
#      BatchNorm2d-200          [-1, 128, 14, 14]             256
#             ReLU-201          [-1, 128, 14, 14]               0
#           Conv2d-202           [-1, 32, 14, 14]          36,864
#       DenseLayer-203          [-1, 544, 14, 14]               0
#      BatchNorm2d-204          [-1, 544, 14, 14]           1,088
#             ReLU-205          [-1, 544, 14, 14]               0
#           Conv2d-206          [-1, 128, 14, 14]          69,632
#      BatchNorm2d-207          [-1, 128, 14, 14]             256
#             ReLU-208          [-1, 128, 14, 14]               0
#           Conv2d-209           [-1, 32, 14, 14]          36,864
#       DenseLayer-210          [-1, 576, 14, 14]               0
#      BatchNorm2d-211          [-1, 576, 14, 14]           1,152
#             ReLU-212          [-1, 576, 14, 14]               0
#           Conv2d-213          [-1, 128, 14, 14]          73,728
#      BatchNorm2d-214          [-1, 128, 14, 14]             256
#             ReLU-215          [-1, 128, 14, 14]               0
#           Conv2d-216           [-1, 32, 14, 14]          36,864
#       DenseLayer-217          [-1, 608, 14, 14]               0
#      BatchNorm2d-218          [-1, 608, 14, 14]           1,216
#             ReLU-219          [-1, 608, 14, 14]               0
#           Conv2d-220          [-1, 128, 14, 14]          77,824
#      BatchNorm2d-221          [-1, 128, 14, 14]             256
#             ReLU-222          [-1, 128, 14, 14]               0
#           Conv2d-223           [-1, 32, 14, 14]          36,864
#       DenseLayer-224          [-1, 640, 14, 14]               0
#      BatchNorm2d-225          [-1, 640, 14, 14]           1,280
#             ReLU-226          [-1, 640, 14, 14]               0
#           Conv2d-227          [-1, 128, 14, 14]          81,920
#      BatchNorm2d-228          [-1, 128, 14, 14]             256
#             ReLU-229          [-1, 128, 14, 14]               0
#           Conv2d-230           [-1, 32, 14, 14]          36,864
#       DenseLayer-231          [-1, 672, 14, 14]               0
#      BatchNorm2d-232          [-1, 672, 14, 14]           1,344
#             ReLU-233          [-1, 672, 14, 14]               0
#           Conv2d-234          [-1, 128, 14, 14]          86,016
#      BatchNorm2d-235          [-1, 128, 14, 14]             256
#             ReLU-236          [-1, 128, 14, 14]               0
#           Conv2d-237           [-1, 32, 14, 14]          36,864
#       DenseLayer-238          [-1, 704, 14, 14]               0
#      BatchNorm2d-239          [-1, 704, 14, 14]           1,408
#             ReLU-240          [-1, 704, 14, 14]               0
#           Conv2d-241          [-1, 128, 14, 14]          90,112
#      BatchNorm2d-242          [-1, 128, 14, 14]             256
#             ReLU-243          [-1, 128, 14, 14]               0
#           Conv2d-244           [-1, 32, 14, 14]          36,864
#       DenseLayer-245          [-1, 736, 14, 14]               0
#      BatchNorm2d-246          [-1, 736, 14, 14]           1,472
#             ReLU-247          [-1, 736, 14, 14]               0
#           Conv2d-248          [-1, 128, 14, 14]          94,208
#      BatchNorm2d-249          [-1, 128, 14, 14]             256
#             ReLU-250          [-1, 128, 14, 14]               0
#           Conv2d-251           [-1, 32, 14, 14]          36,864
#       DenseLayer-252          [-1, 768, 14, 14]               0
#      BatchNorm2d-253          [-1, 768, 14, 14]           1,536
#             ReLU-254          [-1, 768, 14, 14]               0
#           Conv2d-255          [-1, 128, 14, 14]          98,304
#      BatchNorm2d-256          [-1, 128, 14, 14]             256
#             ReLU-257          [-1, 128, 14, 14]               0
#           Conv2d-258           [-1, 32, 14, 14]          36,864
#       DenseLayer-259          [-1, 800, 14, 14]               0
#      BatchNorm2d-260          [-1, 800, 14, 14]           1,600
#             ReLU-261          [-1, 800, 14, 14]               0
#           Conv2d-262          [-1, 128, 14, 14]         102,400
#      BatchNorm2d-263          [-1, 128, 14, 14]             256
#             ReLU-264          [-1, 128, 14, 14]               0
#           Conv2d-265           [-1, 32, 14, 14]          36,864
#       DenseLayer-266          [-1, 832, 14, 14]               0
#      BatchNorm2d-267          [-1, 832, 14, 14]           1,664
#             ReLU-268          [-1, 832, 14, 14]               0
#           Conv2d-269          [-1, 128, 14, 14]         106,496
#      BatchNorm2d-270          [-1, 128, 14, 14]             256
#             ReLU-271          [-1, 128, 14, 14]               0
#           Conv2d-272           [-1, 32, 14, 14]          36,864
#       DenseLayer-273          [-1, 864, 14, 14]               0
#      BatchNorm2d-274          [-1, 864, 14, 14]           1,728
#             ReLU-275          [-1, 864, 14, 14]               0
#           Conv2d-276          [-1, 128, 14, 14]         110,592
#      BatchNorm2d-277          [-1, 128, 14, 14]             256
#             ReLU-278          [-1, 128, 14, 14]               0
#           Conv2d-279           [-1, 32, 14, 14]          36,864
#       DenseLayer-280          [-1, 896, 14, 14]               0
#      BatchNorm2d-281          [-1, 896, 14, 14]           1,792
#             ReLU-282          [-1, 896, 14, 14]               0
#           Conv2d-283          [-1, 128, 14, 14]         114,688
#      BatchNorm2d-284          [-1, 128, 14, 14]             256
#             ReLU-285          [-1, 128, 14, 14]               0
#           Conv2d-286           [-1, 32, 14, 14]          36,864
#       DenseLayer-287          [-1, 928, 14, 14]               0
#      BatchNorm2d-288          [-1, 928, 14, 14]           1,856
#             ReLU-289          [-1, 928, 14, 14]               0
#           Conv2d-290          [-1, 128, 14, 14]         118,784
#      BatchNorm2d-291          [-1, 128, 14, 14]             256
#             ReLU-292          [-1, 128, 14, 14]               0
#           Conv2d-293           [-1, 32, 14, 14]          36,864
#       DenseLayer-294          [-1, 960, 14, 14]               0
#      BatchNorm2d-295          [-1, 960, 14, 14]           1,920
#             ReLU-296          [-1, 960, 14, 14]               0
#           Conv2d-297          [-1, 128, 14, 14]         122,880
#      BatchNorm2d-298          [-1, 128, 14, 14]             256
#             ReLU-299          [-1, 128, 14, 14]               0
#           Conv2d-300           [-1, 32, 14, 14]          36,864
#       DenseLayer-301          [-1, 992, 14, 14]               0
#      BatchNorm2d-302          [-1, 992, 14, 14]           1,984
#             ReLU-303          [-1, 992, 14, 14]               0
#           Conv2d-304          [-1, 128, 14, 14]         126,976
#      BatchNorm2d-305          [-1, 128, 14, 14]             256
#             ReLU-306          [-1, 128, 14, 14]               0
#           Conv2d-307           [-1, 32, 14, 14]          36,864
#       DenseLayer-308         [-1, 1024, 14, 14]               0
#       DenseBlock-309         [-1, 1024, 14, 14]               0
#      BatchNorm2d-310         [-1, 1024, 14, 14]           2,048
#             ReLU-311         [-1, 1024, 14, 14]               0
#           Conv2d-312          [-1, 512, 14, 14]         524,288
#        AvgPool2d-313            [-1, 512, 7, 7]               0
#      BatchNorm2d-314            [-1, 512, 7, 7]           1,024
#             ReLU-315            [-1, 512, 7, 7]               0
#           Conv2d-316            [-1, 128, 7, 7]          65,536
#      BatchNorm2d-317            [-1, 128, 7, 7]             256
#             ReLU-318            [-1, 128, 7, 7]               0
#           Conv2d-319             [-1, 32, 7, 7]          36,864
#       DenseLayer-320            [-1, 544, 7, 7]               0
#      BatchNorm2d-321            [-1, 544, 7, 7]           1,088
#             ReLU-322            [-1, 544, 7, 7]               0
#           Conv2d-323            [-1, 128, 7, 7]          69,632
#      BatchNorm2d-324            [-1, 128, 7, 7]             256
#             ReLU-325            [-1, 128, 7, 7]               0
#           Conv2d-326             [-1, 32, 7, 7]          36,864
#       DenseLayer-327            [-1, 576, 7, 7]               0
#      BatchNorm2d-328            [-1, 576, 7, 7]           1,152
#             ReLU-329            [-1, 576, 7, 7]               0
#           Conv2d-330            [-1, 128, 7, 7]          73,728
#      BatchNorm2d-331            [-1, 128, 7, 7]             256
#             ReLU-332            [-1, 128, 7, 7]               0
#           Conv2d-333             [-1, 32, 7, 7]          36,864
#       DenseLayer-334            [-1, 608, 7, 7]               0
#      BatchNorm2d-335            [-1, 608, 7, 7]           1,216
#             ReLU-336            [-1, 608, 7, 7]               0
#           Conv2d-337            [-1, 128, 7, 7]          77,824
#      BatchNorm2d-338            [-1, 128, 7, 7]             256
#             ReLU-339            [-1, 128, 7, 7]               0
#           Conv2d-340             [-1, 32, 7, 7]          36,864
#       DenseLayer-341            [-1, 640, 7, 7]               0
#      BatchNorm2d-342            [-1, 640, 7, 7]           1,280
#             ReLU-343            [-1, 640, 7, 7]               0
#           Conv2d-344            [-1, 128, 7, 7]          81,920
#      BatchNorm2d-345            [-1, 128, 7, 7]             256
#             ReLU-346            [-1, 128, 7, 7]               0
#           Conv2d-347             [-1, 32, 7, 7]          36,864
#       DenseLayer-348            [-1, 672, 7, 7]               0
#      BatchNorm2d-349            [-1, 672, 7, 7]           1,344
#             ReLU-350            [-1, 672, 7, 7]               0
#           Conv2d-351            [-1, 128, 7, 7]          86,016
#      BatchNorm2d-352            [-1, 128, 7, 7]             256
#             ReLU-353            [-1, 128, 7, 7]               0
#           Conv2d-354             [-1, 32, 7, 7]          36,864
#       DenseLayer-355            [-1, 704, 7, 7]               0
#      BatchNorm2d-356            [-1, 704, 7, 7]           1,408
#             ReLU-357            [-1, 704, 7, 7]               0
#           Conv2d-358            [-1, 128, 7, 7]          90,112
#      BatchNorm2d-359            [-1, 128, 7, 7]             256
#             ReLU-360            [-1, 128, 7, 7]               0
#           Conv2d-361             [-1, 32, 7, 7]          36,864
#       DenseLayer-362            [-1, 736, 7, 7]               0
#      BatchNorm2d-363            [-1, 736, 7, 7]           1,472
#             ReLU-364            [-1, 736, 7, 7]               0
#           Conv2d-365            [-1, 128, 7, 7]          94,208
#      BatchNorm2d-366            [-1, 128, 7, 7]             256
#             ReLU-367            [-1, 128, 7, 7]               0
#           Conv2d-368             [-1, 32, 7, 7]          36,864
#       DenseLayer-369            [-1, 768, 7, 7]               0
#      BatchNorm2d-370            [-1, 768, 7, 7]           1,536
#             ReLU-371            [-1, 768, 7, 7]               0
#           Conv2d-372            [-1, 128, 7, 7]          98,304
#      BatchNorm2d-373            [-1, 128, 7, 7]             256
#             ReLU-374            [-1, 128, 7, 7]               0
#           Conv2d-375             [-1, 32, 7, 7]          36,864
#       DenseLayer-376            [-1, 800, 7, 7]               0
#      BatchNorm2d-377            [-1, 800, 7, 7]           1,600
#             ReLU-378            [-1, 800, 7, 7]               0
#           Conv2d-379            [-1, 128, 7, 7]         102,400
#      BatchNorm2d-380            [-1, 128, 7, 7]             256
#             ReLU-381            [-1, 128, 7, 7]               0
#           Conv2d-382             [-1, 32, 7, 7]          36,864
#       DenseLayer-383            [-1, 832, 7, 7]               0
#      BatchNorm2d-384            [-1, 832, 7, 7]           1,664
#             ReLU-385            [-1, 832, 7, 7]               0
#           Conv2d-386            [-1, 128, 7, 7]         106,496
#      BatchNorm2d-387            [-1, 128, 7, 7]             256
#             ReLU-388            [-1, 128, 7, 7]               0
#           Conv2d-389             [-1, 32, 7, 7]          36,864
#       DenseLayer-390            [-1, 864, 7, 7]               0
#      BatchNorm2d-391            [-1, 864, 7, 7]           1,728
#             ReLU-392            [-1, 864, 7, 7]               0
#           Conv2d-393            [-1, 128, 7, 7]         110,592
#      BatchNorm2d-394            [-1, 128, 7, 7]             256
#             ReLU-395            [-1, 128, 7, 7]               0
#           Conv2d-396             [-1, 32, 7, 7]          36,864
#       DenseLayer-397            [-1, 896, 7, 7]               0
#      BatchNorm2d-398            [-1, 896, 7, 7]           1,792
#             ReLU-399            [-1, 896, 7, 7]               0
#           Conv2d-400            [-1, 128, 7, 7]         114,688
#      BatchNorm2d-401            [-1, 128, 7, 7]             256
#             ReLU-402            [-1, 128, 7, 7]               0
#           Conv2d-403             [-1, 32, 7, 7]          36,864
#       DenseLayer-404            [-1, 928, 7, 7]               0
#      BatchNorm2d-405            [-1, 928, 7, 7]           1,856
#             ReLU-406            [-1, 928, 7, 7]               0
#           Conv2d-407            [-1, 128, 7, 7]         118,784
#      BatchNorm2d-408            [-1, 128, 7, 7]             256
#             ReLU-409            [-1, 128, 7, 7]               0
#           Conv2d-410             [-1, 32, 7, 7]          36,864
#       DenseLayer-411            [-1, 960, 7, 7]               0
#      BatchNorm2d-412            [-1, 960, 7, 7]           1,920
#             ReLU-413            [-1, 960, 7, 7]               0
#           Conv2d-414            [-1, 128, 7, 7]         122,880
#      BatchNorm2d-415            [-1, 128, 7, 7]             256
#             ReLU-416            [-1, 128, 7, 7]               0
#           Conv2d-417             [-1, 32, 7, 7]          36,864
#       DenseLayer-418            [-1, 992, 7, 7]               0
#      BatchNorm2d-419            [-1, 992, 7, 7]           1,984
#             ReLU-420            [-1, 992, 7, 7]               0
#           Conv2d-421            [-1, 128, 7, 7]         126,976
#      BatchNorm2d-422            [-1, 128, 7, 7]             256
#             ReLU-423            [-1, 128, 7, 7]               0
#           Conv2d-424             [-1, 32, 7, 7]          36,864
#       DenseLayer-425           [-1, 1024, 7, 7]               0
#       DenseBlock-426           [-1, 1024, 7, 7]               0
#      BatchNorm2d-427           [-1, 1024, 7, 7]           2,048
#             ReLU-428           [-1, 1024, 7, 7]               0
#           Conv2d-429            [-1, 512, 7, 7]         524,288
#        AvgPool2d-430            [-1, 512, 3, 3]               0
# AdaptiveAvgPool2d-431            [-1, 512, 1, 1]               0
#          Flatten-432                  [-1, 512]               0
#           Linear-433                 [-1, 1000]         513,000
#          Softmax-434                 [-1, 1000]               0
# ================================================================
# Total params: 7,991,144
# Trainable params: 7,991,144
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 383.54
# Params size (MB): 30.48
# Estimated Total Size (MB): 414.60
# ---------------------------------- end ------------------------------------ #


import torch
import torch.nn as nn
import torch.nn.functional as F

#vgg16 is conv3 version
__all__ = ['DenseNet121','DenseNet169','DenseNet201','DenseNet161']


class DenseLayer(nn.Module):

    def __init__(self,in_dim,bn_size,k,drop_rate,memory_efficient=False):
        super(DenseLayer,self).__init__()
        self.sublayer = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,bn_size*k,1,1,0,bias=False),
            nn.BatchNorm2d(bn_size*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*k,k,3,1,1,bias=False),)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient 
        
    def forward(self,input_):
        x = self.sublayer(input_)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate,training=self.training)
        x_final = torch.cat([input_ , x],dim=1)
        return x_final

class DenseBlock(nn.Module):
    ''' official pytorch use nn.ModuleDict, which has self.items'''
    def __init__(self,in_dim,bn_size,layer_num,k,drop_rate, memory_efficient):
        super(DenseBlock,self).__init__()
        self.subblock = self._make_layers(in_dim,layer_num,bn_size,k,
                                          drop_rate, memory_efficient)
    
    def _make_layers(self,in_dim,layer_num,bn_size,k,drop_rate, memory_efficient):
        model_layers = []
        for layer in range(layer_num):
            model_layers.append(DenseLayer(in_dim+layer*k,
            bn_size,k,drop_rate,memory_efficient)
            )
        return nn.Sequential(*model_layers)

    def forward(self,input_):
        x = input_
        for layer in self.subblock:
            x = layer(x)
        return x


class Transition(nn.Sequential):
    ''' nn.Sequential no need overload forward '''
    def __init__(self,in_dim,out_dim):
        super(Transition,self).__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_dim,out_dim,1,1,0,bias=False)
        self.pool = nn.AvgPool2d(2,2)
        
   
class DenseNet(nn.Module):
    def __init__(self,k,block_list,num_init_features=64, bn_size=4, 
                 drop_rate=0, memory_efficient=False):
        super(DenseNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,num_init_features,7,2,3,bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        self.dense_body, self.final_channels = self._make_layers(num_init_features,
                                  bn_size,block_list,k,drop_rate, memory_efficient)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_channels,1000),
            nn.Softmax(dim = 1),)
        self._initialization()
    
    def _initialization(self):
        for m in self.modules():
            if isinstance (m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layers(self,num_init_features,bn_size,block_list,k, 
                     drop_rate, memory_efficient):
        net_layers = []
        for layer_num in block_list:
            net_layers.append(DenseBlock(num_init_features,bn_size,layer_num,
                                         k, drop_rate, memory_efficient))
            net_layers.append(Transition(num_init_features+layer_num*k,
                                       (num_init_features+layer_num*k)//2))
            num_init_features = (num_init_features+layer_num*k)//2                           
        return nn.Sequential(*net_layers),num_init_features
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        x = self.maxpool_1(x)
        x = self.dense_body(x)
        x = self.avgpool_1(x)
        x = self.fc_1(x)
        return x       

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def DenseNet121(pretrained = False):
    model = DenseNet(32, [6, 12, 24, 16])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def DenseNet169(pretrained = False):
    model = DenseNet(32, [6, 12, 32, 32])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model

def DenseNet201(pretrained = False):
    model = DenseNet(32, [6, 12, 48, 32])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model

def DenseNet264(pretrained = False):
    model = DenseNet(32, [6, 12, 64, 48])
    return model


def _test():
    from torchsummary import summary
    model = DenseNet264()
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()

# ------------------------------- mistakes ---------------------------------- #
# nn.sequential no need overload forward
# if name is needed, use nn.ModuleDict
# ---------------------------------- end ------------------------------------ #
