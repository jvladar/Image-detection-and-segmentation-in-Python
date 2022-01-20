## In this repository you can find files for 2nd home assignment for the its8030-2021 subject.
This solution was done by @onsche and @javlad.

Our report with all description of our steps and results can be found [hw2.pdf](https://github.com/jvladar/Image-detection-and-segmentation-in-Python/files/7908443/hw2.pdf) file

In our assigned we deal with problem of image detection and segmentation. We apply different approaches for template matching, descriptor detection and image segmentation of selected sea plants. In last task we improved our solution by applying deep leasrning. First we have annotated our data using the CVAT annotation tool. Then we have used the **COCO** (Common Objects in Context) library to load the exported annotations and we have prepared relevant mask and frame (original image with highlighted mask) images using it's build in functions. We have decided to approach the problem using prebuild models in PyTorch. Our implementation can be found in the hw2_4.py.
