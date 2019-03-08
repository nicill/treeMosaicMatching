import SimpleITK as sitk
#from osgeo import gdal
import sys
import os


def command_iteration(filter) :
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                    filter.GetMetric()))

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile> [initialTransformFile] <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )

fixedImage = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
print(fixedImage.GetDimension())
print(fixedImage.GetSize())
movingImage = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
print(movingImage.GetDimension())
print(movingImage.GetSize())
#pixFixed = list(fixedImage.getdata())

matcher = sitk.HistogramMatchingImageFilter()
if ( fixedImage.GetPixelID() in ( sitk.sitkUInt8, sitk.sitkInt8 ) ):
    matcher.SetNumberOfHistogramLevels(128)
else:
    matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
movingImage = matcher.Execute(movingImage,fixedImage)

# The fast symmetric forces Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in SimpleITK
demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons.SetNumberOfIterations(200)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(1.0)

demons.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(demons) )

displacementField = demons.Execute(fixedImage, movingImage)

print("-------")
print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
print(" RMS: {0}".format(demons.GetRMSChange()))

outTx = sitk.DisplacementFieldTransform(displacementField)

sitk.WriteTransform(outTx, sys.argv[3])

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixedImage);
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(outTx)

out = resampler.Execute(movingImage)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixedImage), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)

writer = sitk.ImageFileWriter()
writer.SetFileName(sys.argv[4])
writer.Execute(simg2)
