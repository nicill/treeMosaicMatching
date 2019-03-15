#original code in https://simpleitk.readthedocs.io/en/master/Examples/index.html#lbl-examples


from functools import reduce
import SimpleITK as sitk
from math import pi
import sys
import os

# function to check that the parameters given are right
def parameterCheck():
    if len ( sys.argv ) < 6:
        print( "Usage: {0} <fixedImageFile> <movingImageFile>  <outputTransformFile> <outputImageFile> <RegistrationMode> ".format(sys.argv[0]))
        sys.exit ( 1 )
    return int(sys.argv[5])

#Helper classes to provide information on how the registration process is doing
def command_iteration(method) : print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),method.GetMetricValue(),method.GetOptimizerPosition()))

def command_iteration_Affine(method) :
    if (method.GetOptimizerIteration()==0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),method.GetMetricValue(),method.GetOptimizerPosition()))

def command_iteration_Demons(filter) : print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

def command_iteration_Bsplines(method, bspline_transform) :
    if method.GetOptimizerIteration() == 0:
        print(bspline_transform)
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),method.GetMetricValue()))

def command_multi_iteration(method) :
    # The sitkMultiResolutionIterationEvent occurs before the
    # resolution of the transform. This event is used here to print
    # the status of the optimizer from the previous registration level.
    if method.GetCurrentLevel() > 0:
        print("Optimizer stop condition: {0}".format(method.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(method.GetOptimizerIteration()))
        print(" Metric value: {0}".format(method.GetMetricValue()))

    print("--------- Resolution Changing ---------")



# Initial images must be the same size and grayscale
# Intensities are normalized and images smoothed
def collectImages(nameFixed,nameMoving,mode):
        pixelType = sitk.sitkFloat32

        fixed = sitk.ReadImage(nameFixed, pixelType)
        if(mode==0): fixed = sitk.Normalize(fixed)
        if(mode==0): fixed = sitk.DiscreteGaussian(fixed, 2.0)

        moving = sitk.ReadImage(nameMoving, pixelType)
        if(mode==0): moving = sitk.Normalize(moving)
        if(mode==0): moving = sitk.DiscreteGaussian(moving, 2.0)

        return fixed,moving

def registerImages(fixed,moving,mode):

    if(mode==0): transform=registerImagesRigid(fixed,moving)
    elif(mode==1): transform=registerImagesAffine(fixed,moving)
    elif(mode==2): transform=registerImagesDemons(fixed,moving,0)
    elif(mode==3): transform=registerImagesDemons(fixed,moving,1)
    elif(mode==4): transform=registerImagesDemons(fixed,moving,2)
    elif(mode==5): transform=registerImagesBsplines(fixed,moving)
    elif(mode==6): transform=registerImagesSyn(fixed,moving)
    else:
        print("Error in RegisterImages, unrecognized registration mode")
        sys.exit ( 1 )
    return transform

def registerImagesRigid(fixed,moving):

    R = sitk.ImageRegistrationMethod()
    # First, choose metric
    metric="mmi"
    if (metric=="jhmi"): R.SetMetricAsJointHistogramMutualInformation()
    elif (metric=="mmi"): R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
    elif (metric=="msq"): R.SetMetricAsMeanSquares()
    elif (metric=="cc"): R.SetMetricAsCorrelation()
    else:
        print("Error in RegisterImages, unrecognizes metric")
        sys.exit ( 1 )

    # optimizer
    optimizer="exh"
    if(optimizer=="lin"): R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,numberOfIterations=200,convergenceMinimumValue=1e-5,convergenceWindowSize=5)
    elif (optimizer=="reg"):
        #https://simpleitk.readthedocs.io/en/master/Examples/ImageRegistrationMethod3/Documentation.html
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,minStep=1e-4,numberOfIterations=500,gradientMagnitudeTolerance=1e-8 )
        R.SetOptimizerScalesFromIndexShift()
    elif (optimizer=="exh"):
        sample_per_axis=12
        # Set the number of samples (radius) in each dimension, with a
        # default step size of 1.0
        R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
        # Utilize the scale to set the step size for each dimension
        R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
    else:
        print("Error in RegisterImages, unrecognized optimizer")
        sys.exit ( 1 )


    # The transform chosen plays a huge part, options, translation, rigid, similarity
    #tx=sitk.TranslationTransform(fixed.GetDimension())
    # The centered transform needs to be initialised with a translation transform,
    tx = sitk.Euler2DTransform()

    #initiaize transform
    movingO=moving.GetOrigin()
    fixedO=fixed.GetOrigin()
    movingSpacing=moving.GetSpacing()
    fixedSpacing=fixed.GetSpacing()
    fixedSize=fixed.GetSize()
    movingSize=moving.GetSize()
    centerFixed=(fixedO[0]+fixedSpacing[0]*fixedSize[0]/2,fixedO[1]+fixedSpacing[1]*fixedSize[1]/2)
    centerMoving=(movingO[0]+movingSpacing[0]*movingSize[0]/2,movingO[1]+movingSpacing[1]*movingSize[1]/2)
    tx.SetTranslation((centerMoving[0]-centerFixed[0],centerMoving[1]-centerFixed[1]))
    tx.SetCenter(centerFixed)
    R.SetInitialTransform(tx)

    #interpolation usually linear
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    return outTx

def registerImagesAffine(fixed,moving):

    R = sitk.ImageRegistrationMethod()
    # First, choose metric
    metric="cc"
    if (metric=="jhmi"): R.SetMetricAsJointHistogramMutualInformation()
    elif (metric=="mmi"): R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
    elif (metric=="msq"): R.SetMetricAsMeanSquares()
    elif (metric=="cc"): R.SetMetricAsCorrelation()
    else:
        print("Error in RegisterImages, unrecognizes metric")
        sys.exit ( 1 )

    # optimizer
    optimizer="reg"
    if(optimizer=="lin"): R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,numberOfIterations=200,convergenceMinimumValue=1e-5,convergenceWindowSize=5)
    elif (optimizer=="reg"):
        #https://simpleitk.readthedocs.io/en/master/Examples/ImageRegistrationMethod3/Documentation.html
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,minStep=1e-4,numberOfIterations=500,gradientMagnitudeTolerance=1e-8 )
        R.SetOptimizerScalesFromIndexShift()
    elif (optimizer=="exh"):
        sample_per_axis=12
        # Set the number of samples (radius) in each dimension, with a
        # default step size of 1.0
        R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
        # Utilize the scale to set the step size for each dimension
        R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
    else:
        print("Error in RegisterImages, unrecognized optimizer")
        sys.exit ( 1 )

    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(2))

    R.SetInitialTransform(tx)

    #interpolation usually linear
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration_Affine(R) )

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    return outTx

def registerImagesDemons(fixedImage,movingImage,demonsType):

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
# Probably use these as diferent types!!!!!!
    if(demonsType==0): demons = sitk.DemonsRegistrationFilter()
    elif(demonsType==1): demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    elif(demonsType==2): demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    else:
        print("Error in registerImagesDemons, unrecognized demons type")
        sys.exit ( 1 )

    demons.SetNumberOfIterations(200)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)
    demons.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration_Demons(demons) )
    displacementField = demons.Execute(fixedImage, movingImage)

    print("-------")
    print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
    print(" RMS: {0}".format(demons.GetRMSChange()))

    outTx = sitk.DisplacementFieldTransform(displacementField)

    return outTx

def registerImagesBsplines(fixed,moving):

    transformDomainMeshSize=[2]*fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,transformDomainMeshSize )

    print("Initial Number of Parameters: {0}".format(tx.GetNumberOfParameters()))

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(5.0,
                                              100,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)

    R.SetInterpolator(sitk.sitkLinear)

    R.SetInitialTransformAsBSpline(tx,
                                   inPlace=True,
                                   scaleFactors=[1,2,5])
    R.SetShrinkFactorsPerLevel([4,2,1])
    R.SetSmoothingSigmasPerLevel([4,2,1])

    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration_Bsplines(R, tx) )
    R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R) )

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(tx)
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    return outTx

def registerImagesSyn(fixedImage,movingImage):

    initialTx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(fixed.GetDimension()))

    R = sitk.ImageRegistrationMethod()

    R.SetShrinkFactorsPerLevel([3,2,1])
    R.SetSmoothingSigmasPerLevel([2,1,1])

    R.SetMetricAsJointHistogramMutualInformation(20)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,estimateLearningRate = R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(initialTx,inPlace=True)

    R.SetInterpolator(sitk.sitkLinear)

#    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
#    R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R) )

    outTx = R.Execute(fixed, moving)

    displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,varianceForTotalField=1.5)
    R.SetMovingInitialTransform(outTx)
    R.SetInitialTransform(displacementTx, inPlace=True)

    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetShrinkFactorsPerLevel([3,2,1])
    R.SetSmoothingSigmasPerLevel([2,1,1])

    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsGradientDescent(learningRate=1,numberOfIterations=300,estimateLearningRate=R.EachIteration)

    outTx.AddTransform( R.Execute(fixed, moving) )

    return outTx

def outputResults(fixed,moving,outTx,transformFile,outputImageFile):

    sitk.WriteTransform(outTx,  transformFile)

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
    writer.SetFileName(outputImageFile)
    writer.Execute(simg2)

#registration modes: Rigid 0, affine 1, classical demons 2, diffeomorphic demons 3, simmetryc demons 4, bsplines 5, syn6
if __name__== '__main__':

    registrationMode = parameterCheck()
    print("starting registration of type "+str(registrationMode))
    fixed,moving = collectImages(sys.argv[1],sys.argv[2],registrationMode)
    transform=registerImages(fixed,moving,registrationMode)
    outputResults(fixed,moving,transform,sys.argv[3],sys.argv[4])
