#!/usr/bin/env python
import itk
import sys

if len(sys.argv) != 3:
    print('Usage: ' + sys.argv[0] + ' <InputFileName> <OutputFileName>')
    sys.exit(1)

inputFileName = sys.argv[1]
outputFileName = sys.argv[2]

Dimension = 3

ComponentType = itk.UC
InputPixelType = itk.RGBPixel[ComponentType]
InputImageType = itk.Image[InputPixelType, Dimension]

OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]

reader = itk.ImageFileReader[InputImageType].New()
reader.SetFileName(inputFileName)

rgbFilter = itk.RGBToLuminanceImageFilter.New(reader)

writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName(outputFileName)
writer.SetInput(rgbFilter.GetOutput())

writer.Update()
