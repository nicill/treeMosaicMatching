#original code in https://simpleitk.readthedocs.io/en/master/Examples/index.html#lbl-examples


from functools import reduce
import SimpleITK as sitk
from math import pi
import sys
import os

# function to check that the parameters given are right
def parameterCheck():
    if len ( sys.argv ) < 4:
        print( "Usage: {0} <fixedImageFile> <movingImageFile>  <imageMode> ".format(sys.argv[0]))
        sys.exit ( 1 )
    return int(sys.argv[3])

#Helper classes to provide information on how the registration process is doing
def command_iteration(method) : print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),method.GetMetricValue(),method.GetOptimizerPosition()))

def command_iteration_Bsplines(method, bspline_transform) :
    if method.GetOptimizerIteration() == 0:
        print(bspline_transform)
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),method.GetMetricValue()))

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

def computeMetric(fixed,moving,mode):

    R = sitk.ImageRegistrationMethod()
    # First, choose metric

    if(mode==0): metric="jhmi"
    elif(mode==1): metric="mmi"
    elif(mode==2): metric="msq"
    elif(mode==3): metric="cc"
    else:
        print("Error in computeMetric, unrecognized metric")
        sys.exit ( 1 )

    if (metric=="jhmi"):  R.SetMetricAsJointHistogramMutualInformation()
    elif (metric=="mmi"): R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
    elif (metric=="msq"): R.SetMetricAsMeanSquares()
    elif (metric=="cc"): R.SetMetricAsCorrelation()
    else:
        print("Error in RegisterImages, unrecognizes metric")
        sys.exit ( 1 )

    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,numberOfIterations=1,convergenceMinimumValue=1e-5,convergenceWindowSize=5)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(2))
    R.SetInitialTransform(tx)

    #interpolation usually linear
    R.SetInterpolator(sitk.sitkLinear)

    print(R.MetricEvaluate(fixed, moving))

    #outTx = R.Execute(fixed, moving)

    #print(" Metric value: {0}".format(R.GetMetricValue()))

    #return outTx

#registration modes: Rigid 0, affine 1, classical demons 2, diffeomorphic demons 3, simmetryc demons 4, bsplines 5, syn6
if __name__== '__main__':

    registrationMode = parameterCheck()
    fixed,moving = collectImages(sys.argv[1],sys.argv[2],registrationMode)
    computeMetric(fixed,moving,registrationMode)
    #outputResults(fixed,moving,transform,sys.argv[3],sys.argv[4])
