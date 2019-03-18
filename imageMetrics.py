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

    # First, set up "phony" registration
    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,numberOfIterations=1,convergenceMinimumValue=1e-5,convergenceWindowSize=5)
    R.SetInitialTransform(sitk.Transform(2,sitk.sitkIdentity)) # Transformation deliberately not using any initializer
    R.SetInterpolator(sitk.sitkLinear)

    #second, choose metric
    if(mode==0): R.SetMetricAsJointHistogramMutualInformation()
    elif(mode==1): R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
    elif(mode==2): R.SetMetricAsMeanSquares()
    elif(mode==3): R.SetMetricAsCorrelation()
    else:
        print("Error in computeMetric, unrecognized metric")
        sys.exit ( 1 )

    #third, get the metric value
    print(R.MetricEvaluate(fixed, moving),end=" ")


#registration modes: Rigid 0, affine 1, classical demons 2, diffeomorphic demons 3, simmetryc demons 4, bsplines 5, syn6
if __name__== '__main__':

    registrationMode = parameterCheck()
    fixed,moving = collectImages(sys.argv[1],sys.argv[2],registrationMode)
    computeMetric(fixed,moving,registrationMode)
    #outputResults(fixed,moving,transform,sys.argv[3],sys.argv[4])
