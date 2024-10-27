# Interactive Image Processing Application With Python

An interactive Python tool for image processing, featuring manual implementations of filters, edge detection, and color transformations with a Tkinter interface.

## Project Features

- **Grayscale Conversion**: Convert images to grayscale manually.
- **Mean Filter**: Apply mean filtering for a blurring effect, computed manually.
- **Prewitt Edge Detection**: Detect edges in X and Y directions.
- **Histogram Stretching**: Enhance pixel values using histogram stretching.
- **Color Space Transformations**: Convert RGB to HSV, HSL, YCbCr, and Lab color spaces.
- **Morphological Operations**: Erosion, dilation, opening, and closing.
- **Histogram Visualization**: Display histograms of red, green, and blue channels.
- **Cropping Tool**: Crop specific regions of an image.

## Requirements

Install the following libraries before running the project:

```bash
pip install opencv-python
pip install numpy
pip install pillow
pip install matplotlib
```

## Usage

### Using Tkinter Interface:

When the program runs, a Tkinter-based graphical interface will open. You can select any image file to process through this interface.

### Without Using the Interface:

If you choose to run the code without the Tkinter interface, make sure the image file is in the project directory and the file path in the code (`img = cv2.imread("BMO.jpg")`) is accurate to avoid errors.
For example, to load an image:

```python
img = cv2.imread("BMO.jpg")  # Ensure the file path is correct
```

## About the Code

- **Manual Computations**: Some operations (e.g., mean filter and Prewitt edge detection) are calculated manually using loops. Instead of built-in `cv2` functions, these implementations help users understand how image processing algorithms work.
- **Commented-out Code Sections**: Certain parts of the code, such as `cv2.imshow` display functions, are commented out. These were initially used to manually check if methods were functioning before the interface was developed. If uncommented, images will display in a separate window, bypassing the interface.
- **Histogram Stretching Values**: The values for `entry_min_pixel` and `entry_max_pixel` can be adjusted to stretch pixel values within a specific range, enhancing image contrast.
- **Kernel Size and Sigma Value**: Parameters like `kernel_size` and `sigma` can be modified as needed to control the level of blur in filters such as the unsharp mask.

## Planned or Future Enhancements

- **Code Translation to English**: The code will be fully translated into English to improve accessibility for an international audience.
- **Reducing Redundant Code**: Some functions, like Prewitt and Sobel edge detection, share similarities and create code redundancy. Merging similar functions will improve readability.
- **Enhanced User Interface**: The graphical interface will be improved for a more interactive and modern user experience.
- **Additional Color Space Transformations**: Functions for color spaces like CMYK may be added.
- **Performance Optimization**: Manually calculated filters and morphological operations will be optimized for performance, or more efficient library functions may be used.
- **Documentation and Comments**: More detailed comments will be added to help developers and users better understand the project.

## Contribution

Those interested in contributing can submit pull requests to improve the code or add new features.
