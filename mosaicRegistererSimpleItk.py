from functools import reduce
import SimpleITK as sitk
import sys
import os


def command_iteration(method) :
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))

# Simple itk registration examples!!!!!!
#https://simpleitk.readthedocs.io/en/master/Examples/index.html#lbl-examples

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFile> <movingImageFile>  <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )

pixelType = sitk.sitkFloat32

#fixed = sitk.ReadImage(sys.argv[1], sitk.sitkVectorFloat32)
fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
fixed = sitk.Normalize(fixed)
fixed = sitk.DiscreteGaussian(fixed, 2.0)

#print(fixed.GetDimension())
#print(fixed.GetNumberOfComponentsPerPixel())
#print(fixed.GetPixel(0,0))
size=fixed.GetSize()
#count=0
#for i in range(size[0]):
#    for j in range(size[1]):
#        if (fixed.GetPixel(i,j)!=(0,0,0,0)):
            #print(fixed.GetPixel(i,j))
#            count+=1
#            if(count%100000==0):print(fixed.GetPixel(i,j))
#print("non black "+str(count)+"/"+str(size[0]*size[1])+" which is a percentage of "+str(100.*count/(size[0]*size[1])))

moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
moving = sitk.Normalize(moving)
moving = sitk.DiscreteGaussian(moving, 2.0)


R = sitk.ImageRegistrationMethod()

R.SetMetricAsJointHistogramMutualInformation()

R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                          numberOfIterations=200,
                                          convergenceMinimumValue=1e-5,
                                          convergenceWindowSize=5)

R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))



sitk.WriteTransform(outTx,  sys.argv[3])

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed);
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)

writer = sitk.ImageFileWriter()
writer.SetFileName(sys.argv[4])
writer.Execute(simg2)


#if ( not "SITK_NOSHOW" in os.environ ):

#    resampler = sitk.ResampleImageFilter()
#    resampler.SetReferenceImage(fixed);
#    resampler.SetInterpolator(sitk.sitkLinear)
#    resampler.SetDefaultPixelValue(1)
#    resampler.SetTransform(outTx)

#    out = resampler.Execute(moving)

#    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
#    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
#    sitk.Show( cimg, "ImageRegistration2 Composition" )
