import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
import cv2

# Read image
pixelType = sitk.sitkFloat32

img = sitk.ReadImage(sys.argv[1], pixelType)

sigma=img.GetSpacing()[0]
level=4

feature_img = sitk.GradientMagnitude(img)

ws_img = sitk.MorphologicalWatershed(feature_img, level=0, markWatershedLine=True, fullyConnected=False)

writer = sitk.ImageFileWriter()
writer.SetFileName(sys.argv[2])
writer.Execute(ws_img)
