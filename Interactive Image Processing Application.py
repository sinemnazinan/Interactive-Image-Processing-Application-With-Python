import cv2
import numpy as np  #yoksa pip install ile terminalden install edilmeli
from PIL import Image #yoksa pip install ile terminalden install edilmeli
import matplotlib.pyplot as plt  #yoksa pip install ile terminalden install edilmeli
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, Canvas, Scrollbar, Toplevel, messagebox
from tkinter.constants import VERTICAL, HORIZONTAL, RIGHT, LEFT, Y, X, BOTH, BOTTOM
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Resmi yükle
img = cv2.imread("BMO.jpg")  # Resmi yüklüyoruz.
if img is None:
    print("Error: Image not loaded. Check file path.")

 # Görüntünün boyutlarını al
height, width = img.shape[:2]

# Resmi BGR kanallarına ayır (cv2.split kullanmadan)
b = img[:,:,0]  # Mavi kanal
g = img[:,:,1]  # Yeşil kanal
r = img[:,:,2]  # Kırmızı kanal

def convert_to_grayscale(img):
    # R, G ve B kanallarını al (cv2'de renk formatı BGR'dir)
    B = b.astype(float)
    G = g.astype(float)
    R = r.astype(float)

    # Gri tonlamada kanalların ağırlıklı ortalamasını hesapla
    Gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Sonucu 8 bitlik bir görüntüye dönüştür
    Gray = np.clip(Gray, 0, 255).astype(np.uint8)
    return Gray

def apply_mean_filter(image, kernel_size):
    # Kernel matrisini oluştur
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # Çıktı görüntüsü için boş bir matris oluştur
    output = np.zeros_like(image)
    # Padding miktarını hesapla
    pad = kernel_size // 2
    # Görüntüyü pad et
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    # Her piksel için mean filtresini uygula
    for y in range(height):
        for x in range(width):
            # İlgili bölgenin mean değerini hesapla
            region = padded_image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.mean(region, axis=(0, 1))

    return output

def manual_adjust_contrast_brightness(image, alpha, beta):

    # Görüntüyü float tipine çevir ve işlemleri yap
    adjusted_image = image.astype(np.float32)  # Görüntüyü float olarak dönüştür

    # Karşıtlık ve parlaklık ayarını uygula
    adjusted_image = adjusted_image * alpha + beta

    # Değerleri 0-255 aralığına sıkıştır ve uint8 tipine dönüştür
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

# Örnek parametreler
alpha = 1.5  # Karşıtlık faktörü
beta = 50    # Parlaklık artışı

# Nearest neighbor interpolation ile resmi yeniden boyutlandır
def resize_image_nearest(image, new_width, new_height):

    # Hedef görüntü için boş bir dizi oluştur
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    # Yeniden boyutlandırma oranlarını hesapla
    x_ratio = width / new_width
    y_ratio = height / new_height

    # Her yeni piksel için, orijinal görüntüden karşılık gelen pikseli kopyala
    for i in range(new_height):
        for j in range(new_width):
            x = int(j * x_ratio)
            y = int(i * y_ratio)
            resized_image[i, j] = image[y, x]

    return resized_image

def manual_rotation(image, angle, scale):
    # Açıyı radyana dönüştür
    angle_rad = np.deg2rad(angle)

    # Döndürme matrisini hesapla
    alpha = np.cos(angle_rad) * scale
    beta = np.sin(angle_rad) * scale

    # Döndürme merkezini hesapla
    cx = width / 2
    cy = height / 2

    # Döndürülmüş görüntü için boş matris
    rotated_image = np.zeros_like(image)

    # Her piksel için yeni konumu hesapla
    for y in range(height):
        for x in range(width):
            # Orjinal piksel konumlarından merkeze göre ayar yap
            x_prime = x - cx
            y_prime = y - cy

            # Yeni konumları hesapla
            new_x = int(alpha * x_prime - beta * y_prime + cx)
            new_y = int(beta * x_prime + alpha * y_prime + cy)

            # Eğer yeni konum geçerliyse, pikseli yeni görüntüye aktar
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[y, x] = image[new_y, new_x]

    return rotated_image

# Örnek kullanım:
angle = 10  # Döndürme açısı
scale = 1   # Ölçeklendirme faktörü

# Kübik interpolasyon yardımcı fonksiyonu
def cubic_interpolation(x):
    x = abs(x)
    if x < 1:
        return (1.5 * x**3 - 2.5 * x**2 + 1)
    elif x < 2:
        return (-0.5 * x**3 + 2.5 * x**2 - 4 * x + 2)
    else:
        return 0


def manual_threshold(image, threshold_value, max_value, method):
    # Create an output image
    output = np.zeros_like(image)

    if method == 'binary':
        # Apply Binary Threshold
        output[image > threshold_value] = max_value
    elif method == 'binary_inv':
        # Apply Binary Inverse Threshold
        output[image <= threshold_value] = max_value
    elif method == 'trunc':
        # Apply Truncate Threshold
        output = np.where(image > threshold_value, threshold_value, image)
    elif method == 'tozero':
        # Apply To Zero Threshold
        output = np.where(image > threshold_value, image, 0)
    elif method == 'tozero_inv':
        # Apply To Zero Inverse Threshold
        output = np.where(image <= threshold_value, image, 0)
    else:
        raise ValueError("Invalid method type")

    return output

def otsu_threshold(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0,256])

    # Total number of pixels
    total = image.size

    sumB, wB, maximum = 0.0, 0, 0.0
    sum1 = np.dot(np.arange(256), hist)  # Total sum of pixels

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += i * hist[i]

        mB = sumB / wB
        mF = (sum1 - sumB) / wF

        # Between class variance
        varBetween = wB * wF * (mB - mF) ** 2

        if varBetween > maximum:
            maximum = varBetween
            threshold = i

    return threshold

# Example usage
threshold_value = 127
max_value = 255
gray_image = convert_to_grayscale(img) 

# Histogram oluşturma (manuel)
def calculate_histogram(image, bins):
    histogram = [0] * bins
    for row in image:
        for value in row:
            histogram[value] += 1
    return histogram

# Histogramları çizdirmek için bir canvas oluştur (tkinter)
def draw_histogram(histogram, title, color, canvas, pos):
    canvas.create_text(130 + 300 * pos, 30, text=title, fill=color)
    max_hist = max(histogram)
    scaling_factor = 100 / max_hist
    for i in range(len(histogram)):
        height = histogram[i] * scaling_factor
        canvas.create_rectangle(50 + i + 300 * pos, 150 - height, 51 + i + 300 * pos, 150, outline=color, fill=color)

hist_b = calculate_histogram(b, 256)
hist_g = calculate_histogram(g, 256)
hist_r = calculate_histogram(r, 256)

binary_threshold = manual_threshold(gray_image, threshold_value, max_value, 'binary')
binary_inv_threshold = manual_threshold(gray_image, threshold_value, max_value, 'binary_inv')
trunc_threshold = manual_threshold(gray_image, threshold_value, max_value, 'trunc')
tozero_threshold = manual_threshold(gray_image, threshold_value, max_value, 'tozero')
tozero_inv_threshold = manual_threshold(gray_image, threshold_value, max_value, 'tozero_inv')

# Compute Otsu's threshold and apply it
otsu_thresh_value = otsu_threshold(gray_image)
otsu_threshold = manual_threshold(gray_image, otsu_thresh_value, max_value, 'binary')

# Görüntü döndürme işlemi
rotated_image = manual_rotation(img, angle, scale)

# Mean filtresini uygula
filtered_image = apply_mean_filter(img, 5)

# Resmi griye çevir
gray_image = convert_to_grayscale(img)

adjusted_image = manual_adjust_contrast_brightness(img, 1.5, 50)

# Resmi yeniden boyutlandır
resized_image = resize_image_nearest(img, 300, 300)

# # Görüntüyü 5 kat büyüt
# scaled_image_cubic = resize_cubic(img, 5, 5)

# # Sonucu göster
# cv2.imshow("Original", img)
# cv2.imshow("Filtered", filtered_image)
# cv2.imshow('Grayscale Image', gray_image)
# cv2.imshow('Resized Image', resized_image)
# cv2.imshow('Adjusted Image', adjusted_image)
# cv2.imshow('Rotated Image', rotated_image)
# # cv2.imshow('Scaled Image (Cubic Interpolation)', scaled_image_cubic)
# cv2.imshow("Grayscale Image", gray_image)
# cv2.imshow("Binary Threshold", binary_threshold)
# cv2.imshow("Binary Inverse Threshold", binary_inv_threshold)
# cv2.imshow("Truncate Threshold", trunc_threshold)
# cv2.imshow("To Zero Threshold", tozero_threshold)
# cv2.imshow("To Zero Inverse Threshold", tozero_inv_threshold)
# cv2.imshow("Otsu's Threshold", otsu_threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Histogramları çizmek için Tkinter penceresi oluştur
# root = tk.Tk()
# root.title("Histogramlar")
# canvas = Canvas(root, width=920, height=200)
# canvas.pack()

# draw_histogram(hist_b, 'Mavi Bilesen Histogrami', 'blue', canvas, 0)
# draw_histogram(hist_g, 'Yesil Bilesen Histogrami', 'green', canvas, 1)
# draw_histogram(hist_r, 'Kirmizi Bilesen Histogrami', 'red', canvas, 2)

def detect_edges(image_path, threshold=128):
    # Görüntüyü dosyadan oku
    img = cv2.imread(image_path)

    # Görüntüyü gri tonlamaya çevir
    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Gaussian blur için çekirdek (kernel) oluştur
    def gaussian_kernel(size, sigma=1):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp( - ((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
            (size, size))
        return kernel / np.sum(kernel)

    # 2D konvolüsyon fonksiyonu
    def convolve2d(image, kernel):
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape

        # Padding boyutları
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Görüntüyü pad ile genişlet
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Konvolüsyonu gerçekleştirmek için boş bir çıktı görüntüsü oluştur
        new_image = np.zeros_like(image)

        # Konvolüsyon işlemi
        for i in range(image_height):
            for j in range(image_width):
                new_image[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])

        return new_image

    # Gaussian blur uygula
    kernel_size = 5
    sigma = 1.4
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = convolve2d(gray, kernel)

    # Sobel filtrelerini manuel olarak oluştur
    sobelx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
    sobely_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])

    # Sobel filtrelerini uygula
    sobelx = convolve2d(blurred, sobelx_kernel)
    sobely = convolve2d(blurred, sobely_kernel)

    # Kenarların büyüklüğünü hesaplayın
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Kenar büyüklüklerini 0-255 aralığına normalize edin
    magnitude = (255 * magnitude / np.max(magnitude)).astype(np.uint8)

    # İkili eşikleme yaparak kenarları belirleyin
    edges = np.zeros_like(magnitude)
    edges[magnitude > threshold] = 255

    return img, edges

original_image, edge_image = detect_edges("BMO.jpg")

# # Orijinal görüntüyü göster
# cv2.imshow("Original Image", original_image)
# # Kenarları göster
# cv2.imshow("Edges", edge_image)
# # Bir tuşa basılana kadar bekle
# cv2.waitKey(0)
# # Tüm pencereleri kapat
# cv2.destroyAllWindows()
# image_filename = 'BMO.jpg'
# root.mainloop()

class ImageProcessingApp:

    # Görüntünün boyutlarını al
    height, width = img.shape[:2]

    # Resmi BGR kanallarına ayır (cv2.split kullanmadan)
    b = img[:,:,0]  # Mavi 
    g = img[:,:,1]  # Yeşil 
    r = img[:,:,2]  # Kırmızı 

    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")

        # Görüntü dosyalarını saklamak için birkaç değişken
        self.original_image = None
        self.image2 = None

        # Görüntü display bölümü
        self.label_image = tk.Label(self.root)
        self.label_image.pack()

        # Ana canvas (görüntü gösterimi için)
        self.canvas = tk.Canvas(root, bg="white", width=600, height=400)
        self.canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Scrollable control panel setupı
        self.control_frame_outer = tk.Frame(root, width=200)
        self.control_frame_outer.pack(side=tk.RIGHT, fill=tk.Y)

        self.control_scrollbar = Scrollbar(self.control_frame_outer, orient="vertical")
        self.control_scrollbar.pack(side="right", fill="y")

        self.control_canvas = tk.Canvas(self.control_frame_outer, yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.config(command=self.control_canvas.yview)

        self.control_frame = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", self.onFrameConfigure)

        # Butonlar ve giriş alanları
        self.button_load = tk.Button(self.control_frame, text="Resim Yükle", command=self.load_image, bg="hot pink", fg="white", font=("Times New Roman", 12))
        self.button_load.pack(pady=5, padx=10,fill=X, side=tk.TOP, anchor=tk.NE)

        self.button_select_image2 = tk.Button(self.control_frame, text="İkinci Resmi Yükle", command=self.select_image2, bg="pink", fg="black", font=("Times New Roman", 12))
        self.button_select_image2.pack(pady=5, padx=10, fill=tk.X)

        self.button_gray = tk.Button(self.control_frame, text="Griye Çevir", command=self.convert_to_gray, bg="light gray", fg="black", font=("Times New Roman", 12))
        self.button_gray.pack(pady=5, padx=10,fill=X)

        # Gri tonlamaya dönüştürme için kullanıcı giriş alanı 
        self.label_r = Label(self.control_frame, text="Red(kırmızı) (R):")
        self.label_r.pack(pady=2)
        self.entry_r = Entry(self.control_frame, width=10)
        self.entry_r.pack(pady=2)
        self.entry_r.insert(0, "0.299")  # Default deger (kullanıcı değiştirmezse bu değer uygulanacak)

        self.label_g = Label(self.control_frame, text="Green(yeşil) (G):")
        self.label_g.pack(pady=2)
        self.entry_g = Entry(self.control_frame, width=10)
        self.entry_g.pack(pady=2)
        self.entry_g.insert(0, "0.587")  # Default deger (kullanıcı değiştirmezse bu değer uygulanacak)

        self.label_b = Label(self.control_frame, text="Blue(mavi) (B):")
        self.label_b.pack(pady=2)
        self.entry_b = Entry(self.control_frame, width=10)
        self.entry_b.pack(pady=2)
        self.entry_b.insert(0, "0.114")  # Default deger (kullanıcı değiştirmezse bu değer uygulanacak)

        # Aritmetik işlemler butonu
        self.button_arithmetic = tk.Button(self.control_frame, text="İki resim arasında aritmetik işlemler", command=self.arithmetic_operations, bg="light blue", fg="black", font=("Times New Roman", 9))
        self.button_arithmetic.pack(pady=5, padx=10, fill=tk.X)

        # Dropdown (açılır menü) ve değişken ekle
        self.operation_var = tk.StringVar(self.root)
        self.operation_var.set("Ekleme")  # Varsayılan değer
        operations = ["Ekleme", "Çıkartma", "Çarpma", "Bölme"]
        self.dropdown_operations = tk.OptionMenu(self.control_frame, self.operation_var, *operations)
        self.dropdown_operations.pack(pady=5, padx=10, fill=tk.X)

        self.button_adjust = tk.Button(self.control_frame, text="Kontrast ve parlaklık uygula", command=self.manual_adjust_contrast_brightness, bg="light green", fg="black", font=("Times New Roman", 12))
        self.button_adjust.pack(pady=5, padx=10,fill=X, anchor=tk.NE)
        
        # Karşıtlık ve Parlaklık Ayarı için Giriş Alanları
        self.label_alpha = Label(self.control_frame, text="Kontrast (Alpha):")
        self.label_alpha.pack(pady=2)
        self.entry_alpha = Entry(self.control_frame, width=10)
        self.entry_alpha.pack(pady=2)
        self.entry_alpha.insert(0, "1.5")  # Varsayılan değer

        self.label_beta = Label(self.control_frame, text="Parlaklık (Beta):")
        self.label_beta.pack(pady=2)
        self.entry_beta = Entry(self.control_frame, width=10)
        self.entry_beta.pack(pady=2)
        self.entry_beta.insert(0, "50")  # Varsayılan değer

        self.button_rotate = tk.Button(self.control_frame, text="Resmi Döndür", command=self.rotate_image, bg="turquoise", fg="black", font=("Times New Roman", 12))
        self.button_rotate.pack(pady=5, padx=10,fill=X)

        # Döndürme işlemi için giriş alanları ve buton
        self.label_angle = Label(self.control_frame, text="Döndürme Açısı (derece):")
        self.label_angle.pack(pady=2)
        self.entry_angle = Entry(self.control_frame, width=10)
        self.entry_angle.pack(pady=2)
        self.entry_angle.insert(0, "0")  # Varsayılan değer

        self.label_scale = Label(self.control_frame, text="Ölçeklendirme Faktörü:")
        self.label_scale.pack(pady=2)
        self.entry_scale = Entry(self.control_frame, width=10)
        self.entry_scale.pack(pady=2)
        self.entry_scale.insert(0, "1")  # Varsayılan değer
        self.entry_scale.bind("<FocusOut>", self.validate_scale)  # Focus kaybettiğinde doğrulama işlevini tetikle

        # Yülenen görüntünün boyutlarını gösteren label
        self.label_image_size = tk.Label(self.control_frame, text="Yüklenen görüntünün boyutu: Yüklenmedi")
        self.label_image_size.pack( pady=5, padx=10, fill=tk.X)

        # Ölçeklendirme butonu
        self.button_resize = tk.Button(self.control_frame, text="Yeniden Boyutlandır", command=self.resize_image, bg="green", fg="black", font=("Times New Roman", 12))
        self.button_resize.pack( pady=5, padx=10, fill=tk.X)

        # Ölçeklendirme için giriş alanları
        self.label_new_width = tk.Label(self.control_frame, text="Yeni Genişlik:")
        self.label_new_width.pack()
        self.entry_new_width = tk.Entry(self.control_frame, width=10)
        self.entry_new_width.pack()

        self.label_new_height = tk.Label(self.control_frame, text="Yeni Yükseklik:")
        self.label_new_height.pack()
        self.entry_new_height = tk.Entry(self.control_frame, width=10)
        self.entry_new_height.pack()

        # Görüntüyü kırpma butonu(Görüntüye manuel seçim yaparak çalışıyor)
        self.button_crop = tk.Button(self.control_frame, text="Görüntüyü Kırp", command=self.enable_crop, bg="purple", fg="black", font=("Times New Roman", 12))
        self.button_crop.pack(pady=5, padx=10, fill=tk.X)

        self.button_zoom_in = tk.Button(self.control_frame, text="Yakınlaştır", command=self.zoom_in, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_zoom_in.pack(pady=5, padx=10, fill=tk.X)

        self.button_zoom_out = tk.Button(self.control_frame, text="Uzaklaştır", command=self.zoom_out, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_zoom_out.pack(pady=5, padx=10, fill=tk.X)

        self.label = tk.Label(self.control_frame, text="Yakınlaştırmak veya uzaklaştırmak için bir ölçek faktörü girin:")
        self.label.pack()
        self.scale_entry = tk.Entry(self.control_frame)
        self.scale_entry.pack()

        self.button_blur = tk.Button(self.control_frame, text="Bulanıklaştır", command=self.apply_blur, bg="turquoise", fg="black", font=("Times New Roman", 12))
        self.button_blur.pack(pady=5, padx=10,fill=X)

        # Bulanıklaştırma Seviyesi için Giriş Alanı ve Etiketi
        self.label_blur_level = tk.Label(self.control_frame, text="Bulanıklaştırma için kullanılacak filtre boyutu:")
        self.label_blur_level.pack(pady=2)
        self.entry_blur_level = tk.Entry(self.control_frame, width=10)
        self.entry_blur_level.pack(pady=2)
        self.entry_blur_level.insert(0, "5")  # Varsayılan değer (kullanıcı değiştirmezse bu değer uygulanacak)

        self.button_add_noise = tk.Button(self.control_frame, text="Gürültü Ekle (Salt & Pepper)", command=self.add_noise, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_add_noise.pack(pady=5, padx=10, fill=tk.X)

        self.label_salt_prob = tk.Label(self.control_frame, text="Salt (Beyaz) Gürültü")
        self.label_salt_prob.pack(pady=2)
        self.entry_salt_prob = tk.Entry(self.control_frame, width=10)
        self.entry_salt_prob.pack(pady=2)
        self.entry_salt_prob.insert(0, "0.05")  # Varsayılan değer

        self.label_pepper_prob = tk.Label(self.control_frame, text="Pepper (Siyah) Gürültü")
        self.label_pepper_prob.pack(pady=2)
        self.entry_pepper_prob = tk.Entry(self.control_frame, width=10)
        self.entry_pepper_prob.pack(pady=2)
        self.entry_pepper_prob.insert(0, "0.05")  # Varsayılan değer

        # Normalde yukarıda bulanıklaştırma için filtremiz var ama bunu sadece görüntü temizliği için kullanacağız
        self.button_mean_filter = tk.Button(self.control_frame, text="Mean Filtre ile gürültü temizle", command=self.apply_mean, bg="light green", fg="black", font=("Times New Roman", 12))
        self.button_mean_filter.pack(pady=5, padx=10, fill=tk.X)

        # Mean filtre boyutu için giriş alanı ve etiketi
        self.label_mean_kernel_size = tk.Label(self.control_frame, text="Gürültü temizlemek için kullanılacak mean filtre boyutu:")
        self.label_mean_kernel_size.pack(pady=2)
        self.entry_mean_kernel_size = tk.Entry(self.control_frame, width=10)
        self.entry_mean_kernel_size.insert(0, "3")  # Varsayılan değer
        self.entry_mean_kernel_size.pack(pady=2)
        self.entry_mean_kernel_size.bind("<FocusOut>", self.validate_mean_kernel_size) # Focus kaybettiğinde doğrulama işlevini tetikle

        self.button_median_filter = tk.Button(self.control_frame, text="Median Filtre ile gürültü temizle", command=self.apply_median, bg="light green", fg="black", font=("Times New Roman", 12))
        self.button_median_filter.pack(pady=5, padx=10, fill=tk.X)

        # Median filtre boyutu için giriş alanı ve etiketi
        self.label_median_kernel_size = tk.Label(self.control_frame, text="Gürültü temizlemek için kullanılacak median filtre boyutu:")
        self.label_median_kernel_size.pack(pady=2)
        self.entry_median_kernel_size = tk.Entry(self.control_frame, width=10)
        self.entry_median_kernel_size.insert(0, "3")  # Varsayılan değer
        self.entry_median_kernel_size.pack(pady=2)
        self.entry_median_kernel_size.bind("<FocusOut>", self.validate_median_kernel_size) # Focus kaybettiğinde doğrulama işlevini tetikle

        self.button_unsharp = tk.Button(self.control_frame, text="Görüntüye unsharp filtre uygula", command=self.apply_unsharp_filter, bg="light blue", fg="black", font=("Times New Roman", 11))
        self.button_unsharp.pack(pady=5, padx=10, fill=tk.X)
        
        # Kernel Size için Giriş Alanı ve Etiket
        self.label_kernel_size = tk.Label(self.control_frame, text="Unsharp filtresi için kullanılacak kernel boyutu:")
        self.label_kernel_size.pack()
        self.unsharp_kernel_size = tk.Entry(self.control_frame, width=10)
        self.unsharp_kernel_size.insert(0, "9")  # Default kernel size
        self.unsharp_kernel_size.pack()

        # Threshold methods buttons
        self.button_binary_threshold = tk.Button(self.control_frame, text="Binary Threshold", command=lambda: self.apply_manual_threshold('binary'), bg="orange", fg="white", font=("Times New Roman", 12))
        self.button_binary_threshold.pack(pady=5, padx=10, fill=tk.X)

        self.button_binary_inv_threshold = tk.Button(self.control_frame, text="Binary Inverse Threshold", command=lambda: self.apply_manual_threshold('binary_inv'), bg="orange", fg="white", font=("Times New Roman", 12))
        self.button_binary_inv_threshold.pack(pady=5, padx=10, fill=tk.X)

        self.button_trunc_threshold = tk.Button(self.control_frame, text="Truncate Threshold", command=lambda: self.apply_manual_threshold('trunc'), bg="orange", fg="white", font=("Times New Roman", 12))
        self.button_trunc_threshold.pack(pady=5, padx=10, fill=tk.X)

        self.button_tozero_threshold = tk.Button(self.control_frame, text="To Zero Threshold", command=lambda: self.apply_manual_threshold('tozero'), bg="orange", fg="white", font=("Times New Roman", 12))
        self.button_tozero_threshold.pack(pady=5, padx=10, fill=tk.X)

        self.button_tozero_inv_threshold = tk.Button(self.control_frame, text="To Zero Inverse Threshold", command=lambda: self.apply_manual_threshold('tozero_inv'), bg="orange", fg="white", font=("Times New Roman", 12))
        self.button_tozero_inv_threshold.pack(pady=5, padx=10, fill=tk.X)

        self.label_kernel_size = tk.Label(self.control_frame, text="(Prewitt; Girdi olarak eşik değerini kullanır ve kenarları bulur)")
        self.label_kernel_size.pack()
        
        self.button_prewitt = tk.Button(self.control_frame, text="Prewitt Kenar Bulma Algoritması", command=self.apply_prewitt, bg="purple", fg="white", font=("Times New Roman", 12))
        self.button_prewitt.pack(pady=5, padx=10, fill=tk.X)
        
        # Eşik değeri için giriş alanı ve etiket
        self.label_threshold = tk.Label(self.control_frame, text="Eşik Değer (Threshold piksel yoğunluklarının karşılaştırılacağı sınır değer):")
        self.label_threshold.pack(pady=2)
        self.entry_threshold = tk.Entry(self.control_frame, width=10)
        self.entry_threshold.pack(pady=2)
        self.entry_threshold.insert(0, "127")  # Default threshold value

        # Max value için giriş alanı ve etiket
        self.label_max_value = tk.Label(self.control_frame, text="Max Değer (Eşik değerini aşan piksellere atanacak değeri belirtir):")
        self.label_max_value.pack(pady=2, fill=tk.X) 
        self.entry_max_value = tk.Entry(self.control_frame, width=10)
        self.entry_max_value.pack(pady=2)
        self.entry_max_value.insert(0, "255")  # Default max value

        self.button_dilation = tk.Button(self.control_frame, text="Morfolojik işlem/Genişleme", command=self.apply_dilation, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_dilation.pack(pady=5, padx=10, fill=tk.X)

        self.button_erosion = tk.Button(self.control_frame, text="Morfolojik işlem/Aşındırma", command=self.apply_erosion, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_erosion.pack(pady=5, padx=10, fill=tk.X)

        self.button_opening = tk.Button(self.control_frame, text="Morfolojik işlem/Açma", command=self.apply_opening, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_opening.pack(pady=5, padx=10, fill=tk.X)

        self.button_closing = tk.Button(self.control_frame, text="Morfolojik işlem/Kapama", command=self.apply_closing, bg="light blue", fg="black", font=("Times New Roman", 12))
        self.button_closing.pack(pady=5, padx=10, fill=tk.X)

        # Kernel size için giriş alanı ve etiketi
        self.label_kernel_size = tk.Label(self.control_frame, text="morfolojik işlemler için kullanılacak kernel boyutu:")
        self.label_kernel_size.pack(pady=2)
        self.entry_kernel_size = tk.Entry(self.control_frame, width=10)
        self.entry_kernel_size.insert(0, "3")  # Varsayılan değer olarak 3 gir
        self.entry_kernel_size.pack(pady=2)

        # Iteration için giriş alanı ve etiketi
        self.label_iterations = tk.Label(self.control_frame, text="morfolojik işlemler için kullanılacak Iterasyon Sayısı:")
        self.label_iterations.pack(pady=2)
        self.entry_iterations = tk.Entry(self.control_frame, width=10)
        self.entry_iterations.insert(0, "1")  # Varsayılan değer olarak 1 gir
        self.entry_iterations.pack(pady=2)

        self.button_color_space = tk.Button(self.control_frame, text="Renk Uzayı Dönüşümleri", command=self.open_color_space_window, bg="purple", fg="white", font=("Times New Roman", 12))
        self.button_color_space.pack(pady=5, padx=10, fill=tk.X)

        # Button to trigger stacked histogram calculation
        self.button_histogram = Button(self.control_frame, text="Histogram Oluştur", command=self.create_histogram_window, bg="yellow", fg="black", font=("Times New Roman", 12))
        self.button_histogram.pack( pady=5, padx=10, fill=tk.X)

        # Entry for histogram bins
        self.label_bins = Label(self.control_frame, text="Histogram için kullanılacak bins (kutu) sayısı:")
        self.label_bins.pack()
        self.entry_bins = Entry(self.control_frame)
        self.entry_bins.pack()
        self.entry_bins.insert(0, "256")

        # Histogram Genişletme için etiketler ve giriş alanları
        self.label_min_pixel = tk.Label(self.control_frame, text="Min piksel değeri:")
        self.label_min_pixel.pack(pady=2)
        self.entry_min_pixel = tk.Entry(self.control_frame, width=10)
        self.entry_min_pixel.pack(pady=2)
        self.entry_min_pixel.insert(0, "0")  # Varsayılan minimum değer

        self.label_max_pixel = tk.Label(self.control_frame, text="Max piksel değeri:")
        self.label_max_pixel.pack(pady=2)
        self.entry_max_pixel = tk.Entry(self.control_frame, width=10)
        self.entry_max_pixel.pack(pady=2)
        self.entry_max_pixel.insert(0, "255")  # Varsayılan maksimum değer

        # Histogram Genişletme butonu
        self.button_stretch_histogram = tk.Button(self.control_frame, text="Histogram Genişlet", command=self.apply_histogram_stretch, bg="purple", fg="white", font=("Times New Roman", 12))
        self.button_stretch_histogram.pack(pady=5, padx=10, fill=tk.X)

    def onFrameConfigure(self, event):
        '''Scrollregion'u frame'in boyutuna göre ayarla'''
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image)
                # Görüntü boyutunu güncelle
                self.update_image_size_label(self.original_image.shape[1], self.original_image.shape[0])
            else:
                print("Failed to load image.")

    def select_image2(self):
        file_path2 = filedialog.askopenfilename(title="İkinci Resmi Seç")
        if file_path2:
            self.image2 = cv2.imread(file_path2)
            if self.image2 is not None:
                self.display_image(self.image2)
                # Görüntü boyutunu güncelle
                self.update_image_size_label(self.original_image.shape[1], self.original_image.shape[0])
            else:
                print("İkinci resim yüklenemedi.")
    
    def bgr_to_rgb(self, image):
        if image is None:
            print("Görüntü yüklenmedi veya bulunamadı.")
            return None  # None döndürerek işlemi durdur
        if len(image.shape) == 3 and image.shape[2] == 3:  # Görüntünün üç renk kanalı olduğunu kontrol et
            # B ve R kanallarını yer değiştir
            return image[:, :, [2, 1, 0]]
        else:
            # Görüntü zaten gri tonlamalı veya uygun kanal sayısına sahip değilse, olduğu gibi döndür
            return image

    def display_image(self, image):
        if image is None:
            print("Görüntü gösterilemiyor: Görüntü None.")
            return
        # Görüntüyü RGB formatına dönüştür (bgr_to_rgb fonksiyonunu kullanarak)
        rgb_image = self.bgr_to_rgb(image)
        if rgb_image is None:
            print("RGB dönüşümü başarısız oldu.")
            return
        # PIL Image nesnesine dönüştür
        image_pil = Image.fromarray(rgb_image)
        # Tkinter PhotoImage nesnesi oluştur
        self.image_tk = ImageTk.PhotoImage(image=image_pil)
        # Canvas'ın boyutunu ayarla
        self.canvas.config(width=image_pil.width, height=image_pil.height)
        # Canvas üzerine görüntüyü yerleştir
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def zoom_in(self):
        # Kullanıcıdan ölçek faktörünü al
        scale_factor = float(self.scale_entry.get())
        if scale_factor <= 1:
            messagebox.showerror("Hata","Ölçek faktörü 1'den büyük olmalıdır.")
            return
        if hasattr(self, 'original_image'):
            zoomed_image = self.zoom_image(self.original_image, scale_factor)
            self.display_image(zoomed_image)
        else:
            messagebox.showerror("Hata","Görüntü yüklenmedi.")

    def zoom_out(self):
        # Kullanıcıdan ölçek faktörünü al
        scale_factor = float(self.scale_entry.get())
        if scale_factor >= 1:
            messagebox.showerror ("Hata","Ölçek faktörü 1'den küçük olmalıdır.")
            return
        if hasattr(self, 'original_image'):
            zoomed_image = self.zoom_image(self.original_image, scale_factor)
            self.display_image(zoomed_image)
        else:
            messagebox.showerror("Hata","Görüntü yüklenmedi.")

    def zoom_image(self, image, scale_factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        # Yeni boyutlarla boş bir görüntü matrisi oluştur
        zoomed_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        for i in range(new_height):
            for j in range(new_width):
                # Eski koordinatlara dönüşüm yap
                y = min(int(i / scale_factor), height - 1)
                x = min(int(j / scale_factor), width - 1)
                zoomed_image[i, j] = image[y, x]

        return zoomed_image

    def convert_to_gray(self):
        if hasattr(self, 'original_image'):
            gray_image = self.convert_to_grayscale(self.original_image)
            self.display_image(gray_image)
        else:
            print("No image loaded.")

    # Gri tonlama işlevi
    def convert_to_grayscale(self,img):
        # # Gri tonlamada kullanılacak ağırlıkları tanımla (KULLANICI GİRECEKSE DEVRE DIŞI)
        # weights = [0.299, 0.587, 0.114]
        weights = [float(self.entry_r.get()), float(self.entry_g.get()), float(self.entry_b.get())]
        # Görüntünün boyutlarını al
        height, width = img.shape[:2]
        # Gri tonlama için boş bir matris oluştur
        gray_img = np.zeros((height, width), dtype=np.uint8)
        # Her piksel için gri tonlama işlemi
        for y in range(height):
            for x in range(width):
                # Pikselin BGR kanallarını al
                b, g, r = img[y, x]
                # Pikselin gri tonlamasını hesapla
                gray_value = weights[0] * r + weights[1] * g + weights[2] * b
                # Gri değeri 0 ile 255 aralığına kısıtla
                gray_img[y, x] = np.clip(gray_value, 0, 255).astype(np.uint8)
        return gray_img

    # Ölçeklendirme işlevi
    def resize_two_image(self, image, target_size):
        height, width = target_size
        resized_image = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
        x_ratio = image.shape[1] / width
        y_ratio = image.shape[0] / height

        for i in range(height):
            for j in range(width):
                x = int(j * x_ratio)
                y = int(i * y_ratio)
                resized_image[i, j] = image[y, x]

        return resized_image

    def arithmetic_operations(self):
        if self.original_image is not None and self.image2 is not None:
            # Görüntü boyutlarını kontrol et ve eşitle
            if self.original_image.shape != self.image2.shape:
                print("Görüntüler aynı boyutta değil. Yeniden boyutlandırılıyor...")
                target_size = (self.original_image.shape[0], self.original_image.shape[1])  # Birinci resmin boyutuna göre
                self.image2 = self.resize_two_image(self.image2, target_size)
                print("İkinci resim yeniden boyutlandırıldı.")

            # Kullanıcıdan işlem seçimi al
            operation = self.operation_var.get()
            # Aritmetik işlemi uygula
            if operation == "Add":
                result_image = self.manual_add(self.original_image, self.image2)
            elif operation == "Subtract":
                result_image = self.manual_subtract(self.original_image, self.image2)
            elif operation == "Multiply":
                result_image = self.manual_multiply(self.original_image, self.image2)
            elif operation == "Divide":
                result_image = self.manual_divide(self.original_image, self.image2)
            else:
                print("Invalid operation selected.")
                return

            # Sonuçları göster
            self.display_image(result_image)
        else:
            print("İki resim de seçilmelidir.")

    def manual_add(self, img1, img2):
        return np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(np.uint8)

    def manual_subtract(self, img1, img2):
        return np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(np.uint8)

    def manual_multiply(self, img1, img2):
        return np.clip(img1.astype(np.float32) * img2.astype(np.float32) / 255, 0, 255).astype(np.uint8)

    def manual_divide(self, img1, img2):
        return np.clip(img1.astype(np.float32) / (img2.astype(np.float32) + 1e-5) * 255, 0, 255).astype(np.uint8)

    def update_image_size_label(self, width, height):
        self.label_image_size.config(text=f"Görüntü Boyutu: {width} x {height} piksel")

    def manual_adjust_contrast_brightness(self):
        if hasattr(self, 'original_image'):
            alpha = float(self.entry_alpha.get())
            beta = float(self.entry_beta.get())
            adjusted_image = self.adjust_contrast_brightness(self.original_image, alpha, beta)
            self.display_image(adjusted_image)
        else:
            print("No image loaded to adjust.")

    def adjust_contrast_brightness(self, image, alpha, beta):
        adjusted_image = image.astype(np.float32)  # Görüntüyü float olarak dönüştür
        adjusted_image = adjusted_image * alpha + beta  # Karşıtlık ve parlaklık ayarını uygula
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)  # Değerleri sıkıştır ve uint8'e çevir
        return adjusted_image

    def rotate_image(self):
        angle = float(self.entry_angle.get())
        scale = float(self.entry_scale.get())
        rotated_image = self.manual_rotation(self.original_image, angle, scale)
        self.display_image(rotated_image)

    def manual_rotation(self, image, angle, scale):
        angle_rad = np.deg2rad(angle)
        original_height, original_width = image.shape[:2]
        cx, cy = original_width / 2, original_height / 2
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)

        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])
        new_width = int((original_height * sin_angle) + (original_width * cos_angle))
        new_height = int((original_height * cos_angle) + (original_width * sin_angle))

        rotation_matrix[0, 2] += (new_width / 2) - cx
        rotation_matrix[1, 2] += (new_height / 2) - cy

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        # Eğer döndürülmüş görüntü canvas boyutlarından büyükse, ölçeklendir
        if new_width > self.canvas.winfo_width() or new_height > self.canvas.winfo_height():
            scale_factor = min(self.canvas.winfo_width() / new_width, self.canvas.winfo_height() / new_height)
            new_size = (int(new_width * scale_factor), int(new_height * scale_factor))
            rotated_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_AREA)

        return rotated_image
    
    def validate_scale(self, event):
        try:
            scale = float(self.entry_scale.get())
            if scale <= 0:
                messagebox.showerror("Hata", "Ölçeklendirme faktörü 0'dan büyük olmalıdır.")
            elif scale < 0.5:
                messagebox.showwarning("Uyarı", "Yetersiz Ölçeklendirme: Görüntü çok küçülecek ve bazı bölümler eksik olabilir.")
            elif scale > 5:
                messagebox.showwarning("Uyarı", "Aşırı Ölçeklendirme: Görüntü çok büyüyecek ve kullanışsız hale gelebilir.")
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir sayısal değer girin.")

    def resize_image(self,):
        new_width = int(self.entry_new_width.get())
        new_height = int(self.entry_new_height.get())

        if hasattr(self, 'original_image'):
            resized_image = self.resize_image_nearest(self.original_image, new_width, new_height)
            self.display_image(resized_image)
        else:
            print("Görüntü yüklenmedi.")

    def resize_image_nearest(self, image, new_width, new_height):
        # Hedef görüntü için boş bir dizi oluştur
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        # Yeniden boyutlandırma oranlarını hesapla
        x_ratio = image.shape[1] / new_width
        y_ratio = image.shape[0] / new_height

        # Her yeni piksel için, orijinal görüntüden karşılık gelen pikseli kopyala
        for i in range(new_height):
            for j in range(new_width):
                x = int(j * x_ratio)
                y = int(i * y_ratio)
                resized_image[i, j] = image[y, x]

        return resized_image

    def apply_blur(self):
        # Kullanıcıdan alınan bulanıklaştırma seviyesini kontrol et
        blur_level = self.entry_blur_level.get()
        if blur_level.isdigit():  # Girilen değer bir sayı mı kontrol et
            blur_level = int(blur_level)
        else:
            blur_level = 5  # Eğer girilen değer uygun değilse varsayılan değeri kullan

        # Bulanıklaştırma işlemini uygula
        if hasattr(self, 'original_image'):
            blurred_image = self.blur_image(self.original_image, blur_level)
            self.display_image(blurred_image)
        elif hasattr(self, 'noisy_image'):
            mean_filtered_image = self.blur_image(self.noisy_image, blur_level)  # Örnek kernel boyutu
            self.display_image(mean_filtered_image)
        else:
            print("Gürültülü görüntü mevcut değil.")

    def blur_image(self, image, kernel_size):
        height, width = image.shape[:2]
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        pad = kernel_size // 2
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        output = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                region = padded_image[y:y+kernel_size, x:x+kernel_size]
                output[y, x] = np.mean(region, axis=(0, 1))
        return output

    def add_salt_pepper_noise(self, image, salt_prob, pepper_prob):
        # Image format and total pixels
        row, col, ch = image.shape
        num_salt = np.ceil(salt_prob * image.size)
        num_pepper = np.ceil(pepper_prob * image.size)

        # Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        image[coords[0], coords[1], :] = 1

        # Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image[coords[0], coords[1], :] = 0
        return image

    def add_noise(self):
        salt_prob = float(self.entry_salt_prob.get())
        pepper_prob = float(self.entry_pepper_prob.get())
        if self.original_image is not None:
            self.noisy_image = self.add_salt_pepper_noise(self.original_image.copy(), salt_prob, pepper_prob)
            self.display_image(self.noisy_image)
        else:
            print("Görüntü yüklenmedi.")

    def apply_mean(self):
        if self.noisy_image is not None:
            kernel_size = int(self.entry_mean_kernel_size.get())  # Kullanıcıdan alınan kernel boyutu
            mean_filtered_image = self.apply_mean_filter(self.noisy_image, kernel_size)
            self.display_image(mean_filtered_image)
        else:
            print("Gürültülü görüntü mevcut değil.")

    def apply_median(self):
        if self.noisy_image is not None:
            kernel_size = int(self.entry_median_kernel_size.get())  # Kullanıcıdan alınan kernel boyutu
            median_filtered_image = self.apply_median_filter(self.noisy_image, kernel_size)
            self.display_image(median_filtered_image)
        else:
            print("Gürültülü görüntü mevcut değil.")

    def apply_mean_filter(self, image, kernel_size):
        # Görüntünün boyutlarını al
        height, width = image.shape[:2]
        # Çıktı görüntüsü için boş bir dizi oluştur
        output = np.zeros_like(image)
        # Padding miktarını hesapla
        pad = kernel_size // 2
        # Görüntüyü pad et
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        # Her piksel için kernel üzerinden geç ve ortalama hesapla
        for y in range(height):
            for x in range(width):
                # Kernel üzerinden geçilen bölgeyi al
                region = padded_image[y:y+kernel_size, x:x+kernel_size]
                # Ortalamayı hesapla ve çıktı dizisine ata
                output[y, x] = np.mean(region, axis=(0, 1))

        return output

    def apply_median_filter(self, image, kernel_size):
        # Görüntünün boyutlarını al
        height, width = image.shape[:2]
        # Çıktı görüntüsü için boş bir dizi oluştur
        output = np.zeros_like(image)
        # Padding miktarını hesapla
        pad = kernel_size // 2
        # Görüntüyü pad et
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        # Her piksel için kernel üzerinden geç ve median hesapla
        for y in range(height):
            for x in range(width):
                # Kernel üzerinden geçilen bölgeyi al
                region = padded_image[y:y+kernel_size, x:x+kernel_size]
                # Medyanı hesapla ve çıktı dizisine ata
                output[y, x] = np.median(region, axis=(0, 1))

        return output
    
    def validate_mean_kernel_size(self, event):
        try:
            kernel_size = int(self.entry_mean_kernel_size.get())
            if kernel_size % 2 == 0 or kernel_size < 3:
                messagebox.showerror("Hata", "Mean filtre boyutu tek ve 3 veya daha büyük bir değer olmalıdır.")
            elif kernel_size > 9:
                messagebox.showwarning("Uyarı", "Çok büyük bir mean filtre boyutu seçildi. Görüntü fazla bulanıklaşabilir.")
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir sayısal değer girin.")

    def validate_median_kernel_size(self, event):
        try:
            kernel_size = int(self.entry_median_kernel_size.get())
            if kernel_size % 2 == 0 or kernel_size < 3:
                messagebox.showerror("Hata", "Median filtre boyutu tek ve 3 veya daha büyük bir değer olmalıdır.")
            elif kernel_size > 7:
                messagebox.showwarning("Uyarı", "Çok büyük bir median filtre boyutu seçildi. Görüntü fazla bulanıklaşabilir.")
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir sayısal değer girin.")

    def apply_unsharp_filter(self):
        kernel_size = self.unsharp_kernel_size.get()
        try:
            kernel_size = int(kernel_size)
            if not 3 <= kernel_size <= 31 or kernel_size % 2 == 0:  # Kernel size must be odd and within reasonable bounds
                raise ValueError("Hata", "Kernel boyutu 3 ile 31 arasında tek sayı olmalıdır.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        if self.original_image is not None:
            unsharp_image = self.unsharp_filter(self.original_image, kernel_size)
            self.display_image(unsharp_image)
        else:
            print("Görüntü yüklenmedi.")

    def unsharp_filter(self, image, kernel_size):
        # Create a blurred version of the image
        sigma = 10.0  # Sigma value can be adjusted or made dynamic as well
        image_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Apply the unsharp mask
        image_sharp = cv2.addWeighted(image, 1.5, image_blur, -0.5, 0)
        return image_sharp

    def apply_manual_threshold(self, method):
        if hasattr(self, 'original_image'):
            # Görüntüyü griye çevir
            gray_image = self.convert_to_grayscale(self.original_image)
            # Kullanıcıdan eşik ve maksimum değer al
            threshold_value = int(self.entry_threshold.get())
            max_value = int(self.entry_max_value.get())
            # Gri görüntü üzerinde eşikleme yap
            thresholded_image = manual_threshold(gray_image, threshold_value, max_value, method)
            self.display_image(thresholded_image)
        else:
            print("No image loaded.")

    def apply_prewitt(self):
        if hasattr(self, 'original_image'):
            # Görüntüyü griye çevir
            gray_image = self.convert_to_grayscale(self.original_image)
            # Prewitt çekirdekleri
            Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # X yönü
            Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Y yönü
            # Çekirdekleri görüntüye uygula
            sobelx = self.apply_filter(gray_image, Gx)
            sobely = self.apply_filter(gray_image, Gy)
            # Kenarların büyüklüğünü hesapla
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Kenar büyüklüklerini 0-255 aralığına normalize edin
            magnitude = (255 * magnitude / np.max(magnitude)).astype(np.uint8)
            # Kullanıcıdan eşik değeri al
            threshold = int(self.entry_threshold.get())

            edges = np.zeros_like(magnitude)
            edges[magnitude > threshold] = 255

            # Görüntüyü göster
            self.display_image(edges)

    def apply_filter(self, image, kernel):
        # Görüntü üzerine çekirdeği uygula
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
        output = np.zeros_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                output[y, x] = np.sum(kernel * padded_image[y:y+kernel_size, x:x+kernel_size])
        return output

    def apply_morphological_operation(self, image, kernel_size, iterations, operation):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == "dilation":
            result = cv2.dilate(image, kernel, iterations=iterations)
        elif operation == "erosion":
            result = cv2.erode(image, kernel, iterations=iterations)
        elif operation == "opening":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "closing":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result

    def apply_dilation(self):
        if self.original_image is not None:
            kernel_size = int(self.entry_kernel_size.get())
            iterations = int(self.entry_iterations.get())
            result_image = self.apply_morphological_operation(self.original_image, kernel_size, iterations, "dilation")
            self.display_image(result_image)
        else:
            print("No image loaded.")

    def apply_erosion(self):
        if self.original_image is not None:
            kernel_size = int(self.entry_kernel_size.get())
            iterations = int(self.entry_iterations.get())
            result_image = self.apply_morphological_operation(self.original_image, kernel_size, iterations, "erosion")
            self.display_image(result_image)
        else:
            print("No image loaded.")

    def apply_opening(self):
        if self.original_image is not None:
            kernel_size = int(self.entry_kernel_size.get())
            iterations = int(self.entry_iterations.get())
            result_image = self.apply_morphological_operation(self.original_image, kernel_size, iterations, "opening")
            self.display_image(result_image)
        else:
            print("No image loaded.")

    def apply_closing(self):
        if self.original_image is not None:
            kernel_size = int(self.entry_kernel_size.get())
            iterations = int(self.entry_iterations.get())
            result_image = self.apply_morphological_operation(self.original_image, kernel_size, iterations, "closing")
            self.display_image(result_image)
        else:
            print("No image loaded.")

    def calculate_histogram(self, channel_data, bins):
        histogram = [0] * bins
        for value in channel_data.flatten():
            index = int(value * (bins - 1) / 255)
            histogram[index] += 1
        return histogram

    def create_histogram_window(self):
        # New window for histograms
        histogram_window = Toplevel(self.root)
        histogram_window.title("Histogramlar")

        # Get color weights from entries
        r_weight = float(self.entry_r.get())
        g_weight = float(self.entry_g.get())
        b_weight = float(self.entry_b.get())

        # Calculate histograms for each channel
        bins = int(self.entry_bins.get())
        hist_r = self.calculate_histogram(self.r * r_weight, bins)
        hist_g = self.calculate_histogram(self.g * g_weight, bins)
        hist_b = self.calculate_histogram(self.b * b_weight, bins)

        # Draw histograms on new canvas in new window
        histogram_canvas = tk.Canvas(histogram_window, width=900, height=300)
        histogram_canvas.pack()

        self.draw_histogram(hist_r, 'Red Channel Histogram', 'red', histogram_canvas, 0)
        self.draw_histogram(hist_g, 'Green Channel Histogram', 'green', histogram_canvas, 1)
        self.draw_histogram(hist_b, 'Blue Channel Histogram', 'blue', histogram_canvas, 2)

    def draw_histogram(self, histogram, title, color, canvas, pos):
        canvas.create_text(150 + 300 * pos, 20, text=title, fill=color)
        max_hist = max(histogram)
        scaling_factor = 100 / max_hist
        for i in range(len(histogram)):
            height = histogram[i] * scaling_factor
            canvas.create_rectangle(50 + 3 * i + 300 * pos, 150 - height, 52 + 3 * i + 300 * pos, 150, outline=color, fill=color)

    def enable_crop(self):
        if self.original_image is not None:
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
            self.rect = None
            self.start_x = None
            self.start_y = None
        else:
            print("No image loaded.")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        self.crop_image(self.start_x, self.start_y, end_x, end_y)

    def crop_image(self, start_x, start_y, end_x, end_y):
        if self.original_image is not None:
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            self.cropped_image = self.original_image[y1:y2, x1:x2]
            self.display_image(self.cropped_image)
        else:
            print("No image to crop.")

    def convert_rgb_to_hsv(self, image):
        # Normalize the image to range [0, 1]
        image = image / 255.0
        hsv_image = np.zeros_like(image)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        max_val = np.max(image, axis=-1)
        min_val = np.min(image, axis=-1)
        delta = max_val - min_val
        
        # Hue calculation
        hsv_image[..., 0] = np.where(delta == 0, 0, np.where(max_val == r, (60 * ((g - b) / delta) + 360) % 360,
                                    np.where(max_val == g, (60 * ((b - r) / delta) + 120) % 360,
                                            (60 * ((r - g) / delta) + 240) % 360)))
        # Saturation calculation
        hsv_image[..., 1] = np.where(max_val == 0, 0, delta / max_val)
        # Value calculation
        hsv_image[..., 2] = max_val
        
        hsv_image[..., 0] = hsv_image[..., 0] / 2  # Convert hue to 0-180 range
        hsv_image[..., 1] = hsv_image[..., 1] * 255  # Convert saturation to 0-255 range
        hsv_image[..., 2] = hsv_image[..., 2] * 255  # Convert value to 0-255 range
        
        return hsv_image.astype(np.uint8)

    def convert_rgb_to_ycbcr(self, image):
        ycbcr_image = np.zeros_like(image, dtype=np.float32)
        ycbcr_image[..., 0] = 16 + (65.481 * image[..., 2] + 128.553 * image[..., 1] + 24.966 * image[..., 0]) / 255
        ycbcr_image[..., 1] = 128 + (-37.797 * image[..., 2] - 74.203 * image[..., 1] + 112.0 * image[..., 0]) / 255
        ycbcr_image[..., 2] = 128 + (112.0 * image[..., 2] - 93.786 * image[..., 1] - 18.214 * image[..., 0]) / 255
        return np.clip(ycbcr_image, 0, 255).astype(np.uint8)

    def convert_rgb_to_lab(self, image):
        image = image / 255.0  # Normalize the image to range [0, 1]
        xyz_image = np.zeros_like(image)

        # Convert to XYZ space
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        xyz_image[..., 0] = 0.412453 * r + 0.357580 * g + 0.180423 * b
        xyz_image[..., 1] = 0.212671 * r + 0.715160 * g + 0.072169 * b
        xyz_image[..., 2] = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # Normalize for D65 white point
        xyz_image[..., 0] = xyz_image[..., 0] / 0.950456
        xyz_image[..., 2] = xyz_image[..., 2] / 1.088754

        # Convert to Lab space
        def f(t):
            delta = 6 / 29
            return np.where(t > delta**3, t**(1/3), (1/3) * (29/6)**2 * t + 4/29)

        lab_image = np.zeros_like(xyz_image)
        lab_image[..., 0] = 116 * f(xyz_image[..., 1]) - 16
        lab_image[..., 1] = 500 * (f(xyz_image[..., 0]) - f(xyz_image[..., 1]))
        lab_image[..., 2] = 200 * (f(xyz_image[..., 1]) - f(xyz_image[..., 2]))

        return np.clip(lab_image, 0, 255).astype(np.uint8)

    def open_color_space_window(self):
        # Yeni pencere oluştur
        color_space_window = Toplevel(self.root)
        color_space_window.title("Renk Uzayı Dönüşümleri")

        # Her renk uzayı dönüşümü için butonları ekle

        button_rgb = tk.Button(color_space_window, text="RGB", command=lambda: self.apply_color_space_conversion('RGB'))
        button_rgb.pack(pady=5, padx=10, fill=tk.X)

        button_hsv = tk.Button(color_space_window, text="RGB to HSV", command=lambda: self.apply_color_space_conversion('HSV'))
        button_hsv.pack(pady=5, padx=10, fill=tk.X)

        button_hsl = tk.Button(color_space_window, text="RGB to HSL", command=lambda: self.apply_color_space_conversion('HSL'))
        button_hsl.pack(pady=5, padx=10, fill=tk.X)

        button_ycbcr = tk.Button(color_space_window, text="RGB to YCbCr", command=lambda: self.apply_color_space_conversion('YCbCr'))
        button_ycbcr.pack(pady=5, padx=10, fill=tk.X)

        button_lab = tk.Button(color_space_window, text="RGB to Lab", command=lambda: self.apply_color_space_conversion('Lab'))
        button_lab.pack(pady=5, padx=10, fill=tk.X)

        button_gray = tk.Button(color_space_window, text="RGB to Grayscale", command=lambda: self.apply_color_space_conversion('Grayscale'))
        button_gray.pack(pady=5, padx=10, fill=tk.X)

    def apply_color_space_conversion(self, color_space):
        if self.original_image is not None:
            if color_space == 'RGB':
                converted_image = self.original_image.copy()
            if color_space == 'HSV':
                converted_image = self.convert_rgb_to_hsv(self.original_image)
            elif color_space == 'HSL':
                converted_image = self.convert_rgb_to_hsl(self.original_image)
            elif color_space == 'YCbCr':
                converted_image = self.convert_rgb_to_ycbcr(self.original_image)
            elif color_space == 'Lab':
                converted_image = self.convert_rgb_to_lab(self.original_image)
            elif color_space == 'Grayscale':
                converted_image = self.convert_to_grayscale(self.original_image)
            self.display_image(converted_image)
        else:
            print("No image loaded.")

    # Renk uzayı dönüşüm fonksiyonlarını buraya ekleyin
    def convert_rgb_to_hsv(self, image):
        # Normalize the image to range [0, 1]
        image = image / 255.0
        hsv_image = np.zeros_like(image)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        max_val = np.max(image, axis=-1)
        min_val = np.min(image, axis=-1)
        delta = max_val - min_val
        
        # Hue calculation
        hsv_image[..., 0] = np.where(delta == 0, 0, np.where(max_val == r, (60 * ((g - b) / delta) + 360) % 360,
                                    np.where(max_val == g, (60 * ((b - r) / delta) + 120) % 360,
                                            (60 * ((r - g) / delta) + 240) % 360)))
        # Saturation calculation
        hsv_image[..., 1] = np.where(max_val == 0, 0, delta / max_val)
        # Value calculation
        hsv_image[..., 2] = max_val
        
        hsv_image[..., 0] = hsv_image[..., 0] / 2  # Convert hue to 0-180 range
        hsv_image[..., 1] = hsv_image[..., 1] * 255  # Convert saturation to 0-255 range
        hsv_image[..., 2] = hsv_image[..., 2] * 255  # Convert value to 0-255 range
        
        return hsv_image.astype(np.uint8)

    def convert_rgb_to_hsl(self, image):
        # Normalize the image to range [0, 1]
        image = image / 255.0
        hsl_image = np.zeros_like(image)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        max_val = np.max(image, axis=-1)
        min_val = np.min(image, axis=-1)
        delta = max_val - min_val

        # Lightness calculation
        hsl_image[..., 2] = (max_val + min_val) / 2

        # Saturation calculation
        hsl_image[..., 1] = np.where(delta == 0, 0, np.where(hsl_image[..., 2] <= 0.5, delta / (max_val + min_val), delta / (2.0 - max_val - min_val)))

        # Hue calculation
        hsl_image[..., 0] = np.where(delta == 0, 0, np.where(max_val == r, ((g - b) / delta) % 6,
                                    np.where(max_val == g, ((b - r) / delta) + 2, ((r - g) / delta) + 4))) * 60

        hsl_image[..., 0] = hsl_image[..., 0] / 360 * 255  # Convert hue to 0-255 range
        hsl_image[..., 1] = hsl_image[..., 1] * 255  # Convert saturation to 0-255 range
        hsl_image[..., 2] = hsl_image[..., 2] * 255  # Convert lightness to 0-255 range

        return hsl_image.astype(np.uint8)

    def convert_rgb_to_ycbcr(self, image):
        ycbcr_image = np.zeros_like(image, dtype=np.float32)
        ycbcr_image[..., 0] = 16 + (65.481 * image[..., 2] + 128.553 * image[..., 1] + 24.966 * image[..., 0]) / 255
        ycbcr_image[..., 1] = 128 + (-37.797 * image[..., 2] - 74.203 * image[..., 1] + 112.0 * image[..., 0]) / 255
        ycbcr_image[..., 2] = 128 + (112.0 * image[..., 2] - 93.786 * image[..., 1] - 18.214 * image[..., 0]) / 255
        return np.clip(ycbcr_image, 0, 255).astype(np.uint8)

    def convert_rgb_to_lab(self, image):
        image = image / 255.0  # Normalize the image to range [0, 1]
        xyz_image = np.zeros_like(image)

        # Convert to XYZ space
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        xyz_image[..., 0] = 0.412453 * r + 0.357580 * g + 0.180423 * b
        xyz_image[..., 1] = 0.212671 * r + 0.715160 * g + 0.072169 * b
        xyz_image[..., 2] = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # Normalize for D65 white point
        xyz_image[..., 0] = xyz_image[..., 0] / 0.950456
        xyz_image[..., 2] = xyz_image[..., 2] / 1.088754

        # Convert to Lab space
        def f(t):
            delta = 6 / 29
            return np.where(t > delta**3, t**(1/3), (1/3) * (29/6)**2 * t + 4/29)

        lab_image = np.zeros_like(xyz_image)
        lab_image[..., 0] = 116 * f(xyz_image[..., 1]) - 16
        lab_image[..., 1] = 500 * (f(xyz_image[..., 0]) - f(xyz_image[..., 1]))
        lab_image[..., 2] = 200 * (f(xyz_image[..., 1]) - f(xyz_image[..., 2]))

        return np.clip(lab_image, 0, 255).astype(np.uint8)

    def apply_histogram_stretch(self):
        if hasattr(self, 'original_image'):
            # Kullanıcıdan alınan min ve max değerler
            min_pixel = int(self.entry_min_pixel.get())
            max_pixel = int(self.entry_max_pixel.get())
            # Histogram Genişletme işlemini uygula
            stretched_image = self.histogram_stretch(self.original_image, min_pixel, max_pixel)
            self.display_image(stretched_image)
        else:
            print("Görüntü yüklenmedi.")

    def histogram_stretch(self, image, min_pixel, max_pixel):
        # Piksel değerlerini 0-255 aralığına sıkıştır
        image = np.clip(image, min_pixel, max_pixel)
        # Minimum ve maximum arasındaki aralığı kullanarak normalize et
        image = (image - min_pixel) / (max_pixel - min_pixel) * 255
        return image.astype(np.uint8)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()