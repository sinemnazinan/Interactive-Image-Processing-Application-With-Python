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

## Those interested in contributing can submit pull requests to improve the code or add new features.

# Python ile Etkileşimli Görüntü İşleme Uygulaması

Tkinter arayüzüyle manuel filtreleme, kenar algılama ve renk dönüşümlerini içeren etkileşimli bir Python görüntü işleme aracı.

## Proje Özellikleri

- **Gri Ton Dönüşümü**: Görüntüleri manuel olarak gri tonlamaya çevir.
- **Ortalama Filtresi**: Manuel hesaplanan ortalama filtreleme ile bulanıklaştırma etkisi uygula.
- **Prewitt Kenar Algılama**: X ve Y yönlerinde kenarları algıla.
- **Histogram Germe**: Piksel değerlerini histogram germe kullanarak iyileştir.
- **Renk Uzayı Dönüşümleri**: RGB'den HSV, HSL, YCbCr ve Lab renk uzaylarına dönüştürme.
- **Morfolojik İşlemler**: Erozyon, genişleme, açma ve kapama işlemleri.
- **Histogram Görselleştirme**: Kırmızı, yeşil ve mavi kanalların histogramlarını görüntüle.
- **Kırpma Aracı**: Görüntünün belirli bölgelerini kırp.

## Gereksinimler

Projeyi çalıştırmadan önce aşağıdaki kütüphaneleri yükleyin:

```bash
pip install opencv-python
pip install numpy
pip install pillow
pip install matplotlib
```

## Kullanım

### Tkinter Arayüzü Kullanarak:

Program çalıştığında, Tkinter tabanlı grafiksel bir arayüz açılacaktır. Bu arayüz üzerinden işlemek istediğiniz görüntü dosyasını seçebilirsiniz.

### Arayüz Olmadan Kullanma:

Kodu Tkinter arayüzü olmadan çalıştırmayı tercih ederseniz, görüntü dosyasının proje dizininde olduğundan ve koddaki dosya yolunun (`img = cv2.imread("BMO.jpg")`) doğru olduğundan emin olun.
Örneğin, bir görüntüyü yüklemek için:

```python
img = cv2.imread("BMO.jpg")  # Dosya yolunun doğru olduğundan emin olun
```

## Kod Hakkında

- **Manuel Hesaplamalar**: Bazı işlemler (örn. ortalama filtre ve Prewitt kenar algılama) döngüler kullanılarak manuel olarak hesaplanır. Bu implementasyonlar, `cv2` fonksiyonları yerine kullanılarak, kullanıcıların görüntü işleme algoritmalarının nasıl çalıştığını anlamalarına yardımcı olur.
- **Yorum Satırına Alınmış Kod Bölümleri**: `cv2.imshow` gibi bazı kod bölümleri yorum satırına alınmıştır. Bu bölümler, yöntemlerin çalışıp çalışmadığını manuel olarak kontrol etmek için arayüz geliştirilmeden önce kullanılmıştır. Eğer yorumdan kaldırılırsa, görüntüler ayrı bir pencerede açılacaktır ve arayüz devre dışı kalacaktır.
- **Histogram Germe Değerleri**: `entry_min_pixel` ve `entry_max_pixel` değerleri, belirli bir aralıkta piksel değerlerini germek ve görüntü kontrastını artırmak için ayarlanabilir.
- **Kernel Boyutu ve Sigma Değeri**: `kernel_size` ve `sigma` gibi parametreler, bulanıklık seviyesini kontrol etmek için unsharp mask gibi filtrelerde ihtiyaç duyulduğunda değiştirilebilir.

## Planlanan veya Gelecekteki İyileştirmeler

- **Kodun İngilizceye Çevirisi**: Kod, uluslararası bir kitle için erişilebilirliği artırmak amacıyla tamamen İngilizceye çevrilecektir.
- **Gereksiz Kodların Azaltılması**: Prewitt ve Sobel kenar algılama gibi bazı fonksiyonlar benzerlik göstererek kod tekrarı yaratmaktadır. Benzer fonksiyonların birleştirilmesi okunabilirliği artıracaktır.
- **Geliştirilmiş Kullanıcı Arayüzü**: Grafik arayüzü daha etkileşimli ve modern bir kullanıcı deneyimi sunacak şekilde geliştirilecektir.
- **Ek Renk Uzayı Dönüşümleri**: CMYK gibi renk uzayları için dönüşüm fonksiyonları eklenebilir.
- **Performans Optimizasyonu**: Manuel hesaplanan filtreler ve morfolojik işlemler performans için optimize edilecek veya daha verimli kütüphane fonksiyonları kullanılabilir.
- **Dokümantasyon ve Yorumlar**: Geliştiricilerin ve kullanıcıların projeyi daha iyi anlamalarına yardımcı olmak için daha ayrıntılı açıklamalar eklenecektir.

## Katkı

İlgilenenler, kodu iyileştirmek veya yeni özellikler eklemek için pull request göndererek projeye katkıda bulunabilir.
