from osgeo import gdal
import sys
import cv2

dataset = gdal.Open(sys.argv[1], gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    print ("band "+str(x))
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()
    cv2.imwrite("./"+str(x)+"sambomba.jpg",array)
    #cv2.imwrite("./COLOR"+str(x)+"sambomba.jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
