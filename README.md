# Fashion-MNIST Image Reconstruction using Variational Autoencoder (VAE)

Projek ini merupakan implementasi jaringan saraf tiruan **Variational Autoencoder (VAE)** berbasis PyTorch untuk melakukan kompresi dimensi (*dimensionality reduction*) dan rekonstruksi gambar menggunakan dataset **Fashion-MNIST**. Eksperimen difokuskan pada analisis pengaruh ukuran ruang laten (*latent space/z_dim*) terhadap kualitas gambar yang didekodekan kembali.

---

## 📌 Fitur Utama
* **Arsitektur Konvolusional:** Menggunakan komponen `Conv2d` pada *Encoder* dan `ConvTranspose2d` pada *Decoder* untuk menangkap fitur spasial gambar pakaian secara optimal.
* **Eksperimen Multi-Dimensi:** Melatih dan menguji model dengan 3 variasi ruang laten yang berbeda: **Dimensi 2**, **Dimensi 8**, dan **Dimensi 32**.
* **Skrip Inferensi Lokal:** Dilengkapi skrip pengujian mandiri di perangkat lokal (`reconstruct.py`) untuk memuat bobot terlatih (`.pth`) dan menguji gambar eksternal baru.

---

## 📐 Detail Arsitektur Model

### 1. Encoder (EncoderVAE)
* **Input:** Gambar Grayscale ($1 \times 32 \times 32$)
* **Lapisan Fitur:** 3 Tingkat Lapisan Konvolusi bertahap ($32 \rightarrow 64 \rightarrow 128$ saluran) dikombinasikan dengan fungsi aktivasi `ReLU`.
* **Lapisan Proyeksi:** Mengubah fitur terkompresi berukuran 2048 ($128 \times 4 \times 4$) melalui lapisan linear terpisah untuk menghasilkan parameter statistik Mean ($\mu$) dan Log-Variance ($\sigma^2$).
* **Reparameterization Trick:** Mengambil sampel variabel laten $z$ secara stokastik agar fungsi model dapat diturunkan (*differentiable*).

### 2. Decoder
* **Input:** Vektor Laten $z$ (Sesuai konfigurasi: 2, 8, atau 32)
* **Lapisan Fitur:** Proyeksi balik ke bentuk matriks $128 \times 4 \times 4$.
* **Lapisan Upsampling:** 3 Tingkat Lapisan Konvolusi Transpose bertahap untuk mengembalikan resolusi spasial objek citra.
* **Output:** Gambar Rekonstruksi berukuran ($1 \times 32 \times 32$) dengan aktivasi `Sigmoid` (nilai piksel 0-1).

---

## 🚀 Cara Menjalankan Projek di Perangkat Lokal

### 1. Prasyarat
Pastikan komputer Anda telah terinstal Python (Minimal versi 3.10 ke atas) dan library pendukung berikut:
```bash
pip install torch torchvision matplotlib pillow
