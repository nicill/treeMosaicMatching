import SimpleITK as sitk
import itk
from math import pi
import sys
import os

#DOES NOT WQRK!!!
def computeMetrics(nameFixed,nameMoving):

    print("NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!")
    pixelType = sitk.sitkFloat32

    fixedImage = sitk.ReadImage(nameFixed, pixelType)
    #fixedImage =itk.ImageFileReader.New(FileName=nameFixed).GetOutput()
    #if(mode==0): fixed = sitk.Normalize(fixed)
    #if(mode==0): fixed = sitk.DiscreteGaussian(fixed, 2.0)

    #movingImage =itk.ImageFileReader.New(FileName=nameMoving).GetOutput()
    movingImage = sitk.ReadImage(nameMoving, pixelType)
    #if(mode==0): moving = sitk.Normalize(moving)
    #if(mode==0): moving = sitk.DiscreteGaussian(moving, 2.0)

    metric=itk.MeanSquaresImageToImageMetric()
    interpolator=itk.LinearInterpolateImageFunction()
    transform=itk.TranslationTransform()

    interpolator.SetInputImage( fixedImage )

    metric.SetFixedImage( fixedImage )
    metric.SetMovingImage( movingImage )
    metric.SetFixedImageRegion( fixedImage.GetLargestPossibleRegion() )
    metric.SetTransform( transform )
    metric.SetInterpolator( interpolator )

    params=transform.GetNumberOfParameters()
    #params.Fill(0.0)

    metric.Execute(fixedImage,movingImage)

"""


 metric->SetFixedImage( fixedImage );
 metric->SetMovingImage( movingImage );
 metric->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
 metric->SetTransform( transform );
 metric->SetInterpolator( interpolator );

 TransformType::ParametersType params(transform->GetNumberOfParameters());
 params.Fill(0.0);

 metric->Initialize();
 for (double x = -10.0; x <= 10.0; x+=5.0)
   {
   params(0) = x;
   for (double y = -10.0; y <= 10.0; y+=5.0)
     {
     params(1) = y;
     std::cout << params << ": " << metric->GetValue( params ) << std::endl ;
     }
   }

 return EXIT_SUCCESS;

 """

if __name__== '__main__':
     computeMetrics(sys.argv[1],sys.argv[2])
