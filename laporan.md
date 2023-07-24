# Laporan Proyek Machine Learning

## Domain Proyek

Industri alat angkutan atau otomotif di Indonesia menunjukkan pertumbuhan yang positif berdasarkan data dari Badan Pusat Statistik (BPS). Pada tahun 2022, produk domestik bruto (PDB) industri otomotif atas dasar harga konstan (ADHK) mencapai Rp207,79 triliun, mengalami kenaikan sebesar 10,67% dibandingkan tahun sebelumnya yang sebesar Rp187,75 triliun. Setelah mengalami penurunan akibat pandemi Covid-19 pada tahun 2020, industri otomotif mulai pulih sejak tahun 2021 dan pertumbuhannya terus berlanjut pada tahun lalu. PDB industri otomotif juga menempati posisi ketiga terbesar di antara subsektor industri pengolahan lainnya, di bawah industri logam dasar yang tumbuh 14,8% dan industri mesin yang naik 11,37% [[1]](https://www.bps.go.id/indikator/indikator/view_data_pub/0000/api_pub/aDJqeEc4MTJlOWRaRWFOQ1Q4UE5UQT09/da_15/1).

Pertumbuhan industri otomotif didukung oleh kebijakan pemerintah dalam bentuk diskon pajak penjualan atas barang mewah (PPnBM) sepanjang tahun 2022. Kebijakan ini berhasil meningkatkan penjualan mobil sebesar 18,1%. Industri alat angkutan, sebagai bagian dari industri pengolahan, berkontribusi sebesar 8,67% terhadap PDB industri pengolahan pada tahun 2022 [[2]](https://kemenperin.go.id/artikel/22994/Produksi-Mobil-Lampaui-Target,-Penjualan-Naik-71-Persen). Data ini menunjukkan pentingnya industri otomotif dalam ekonomi nasional dan perannya dalam memulihkan pertumbuhan ekonomi setelah masa-masa sulit akibat pandemi [[3]](https://dataindonesia.id/sektor-riil/detail/kinerja-industri-otomotif-melonjak-1067-pada-2022).

Penjualan mobil adalah salah satu sektor bisnis yang memiliki peran penting dalam industri otomotif. Di tengah persaingan yang semakin ketat dan meningkatnya permintaan konsumen, strategi pemasaran yang tepat menjadi kunci keberhasilan dalam menjalankan bisnis penjualan mobil. Salah satu aspek penting dalam strategi pemasaran adalah menentukan harga yang tepat untuk mobil yang ditawarkan. Menetapkan harga yang ideal adalah suatu tantangan, mengingat perbedaan preferensi konsumen, persaingan pasar, dan faktor-faktor lainnya yang mempengaruhi harga mobil [[4]](https://ejournal.unsrat.ac.id/index.php/emba/article/view/21893).

Dalam upaya menentukan harga mobil yang realistis, diperlukan pendekatan yang dapat memberikan hasil prediksi yang akurat. Salah satu pendekatan yang dapat digunakan adalah penerapan teknik _machine learning_. Dengan menggunakan teknik _machine learning_, kita dapat memanfaatkan data historis penjualan mobil, variabel-variabel yang relevan, dan algoritma pembelajaran mesin untuk melakukan prediksi harga mobil yang lebih akurat [[5]](https://slims.ahmaddahlan.ac.id/index.php?p=fstream-pdf&fid=40&bid=3192).

Penerapan teknik _machine learning_ dalam menentukan harga mobil memiliki relevansi yang kuat dengan kebutuhan pasar dan preferensi konsumen. Konsumen sering kali ingin memiliki gambaran yang jelas tentang harga yang realistis sebelum mereka memutuskan untuk membeli mobil. Dengan adanya prediksi harga yang lebih akurat melalui teknik _machine learning_, konsumen dapat melakukan perencanaan keuangan dengan lebih baik, membandingkan harga dari berbagai pilihan mobil, dan membuat keputusan pembelian yang lebih informasional.

Oleh karena itu, penggunaan teknik _machine learning_ dalam menentukan harga mobil memberikan manfaat yang signifikan dalam meningkatkan keputusan bisnis, memberikan keuntungan kompetitif, dan memberikan kepuasan bagi konsumen.

## Business Understanding

### Problem Statements

- Fitur apa yang paling berpengaruh dalam menentukan harga penjualan mobil?
- Bagaimana pendekatan _machine learning_ dapat digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur yang berpengaruh tersebut?

### Goals

- Memahami faktor-faktor yang mempengaruhi harga penjualan mobil untuk dapat mengidentifikasi variabel yang paling signifikan dalam menentukan harga sehingga memberikan wawasan yang lebih baik dalam strategi penetapan harga, mendukung pengambilan keputusan yang lebih informasional, dan memperkuat strategi pemasaran dalam industri penjualan mobil.
- Memahami pendekatan _machine learning_ yang digunakan untuk prediksi harga mobil, sehingga melalui penerapan pendekatan _machine learning_ dalam prediksi harga mobil mampu meningkatkan efisiensi dan kecepatan dalam penentuan harga, meminimalkan kesalahan manusia, serta memberikan keunggulan kompetitif dalam bisnis penjualan mobil.

### Solution Approach

Pada proyek ini, penulis menawarkan untuk menggunakan 3 algoritma dalam melakukan prediksi harga mobil dan menentukan algoritma yang paling baik prediksinya diantara lainnya. Adapun algoritma yang dipilih yaitu:

- Regresi Linier
- _Decision Tree_
- KNN (_K-Nearest Neighbors_)

Berikut adalah penjelasan yang lebih mendalam mengenai pemilihan ketiga algoritma (Regresi Linier, _Decision Tree_, KNN) sebagai solusi untuk memprediksi harga mobil dan bagaimana masing-masing algoritma dapat membantu menjawab pertanyaan masalah:

- Regresi Linier: Algoritma ini dipilih karena sederhana dan memiliki interpretabilitas tinggi. Regresi Linier dapat memberikan pemahaman yang jelas tentang kontribusi setiap fitur (tipe, model, tahun mobil) terhadap prediksi harga mobil. Dengan demikian, kita dapat mengidentifikasi variabel yang paling signifikan dalam menentukan harga.

- _Decision Tree_: Algoritma ini dipilih karena kemampuannya untuk memodelkan hubungan yang kompleks antara fitur-fitur dan variabel target. _Decision tree_ membagi _dataset_ berdasarkan pemilihan fitur yang paling informatif pada setiap langkahnya. Hal ini memungkinkan pemodelan yang lebih kompleks dibandingkan dengan regresi Linier. _Decision tree_ dapat membantu menggali hubungan non-Linier antara fitur-fitur dan harga mobil.

- KNN (_K-Nearest Neighbors_): Algoritma ini dipilih karena sifatnya yang sederhana, intuitif, dan mampu menangani hubungan lokal antara fitur-fitur dan target. KNN mencari tetangga terdekat dari suatu data dan memprediksi berdasarkan mayoritas label atau rata-rata nilai tetangga terdekatnya. KNN dapat membantu dalam mengidentifikasi pola lokal yang mungkin terdapat dalam data, sehingga dapat memberikan prediksi yang lebih akurat dalam kasus harga mobil.

Pada proyek ini, matriks evaluasi yang digunakan adalah _Mean Absolute Error_ (MAE) dan _Mean Squared Error_ (MSE). Hal ini bertujuan untuk mengetahui kesalahan prediksi yang terkecil dari setiap algoritma yang digunakan.

## Data Understanding

_Dataset_ yang digunakan pada proyek _machine learning_ ini adalah [_Car Sales Data_](https://www.kaggle.com/datasets/suraj520/car-sales-data) yang bersumber dari situs Kaggle. _Dataset_ ini berisi 2.500.000 baris informasi tentang penjualan mobil dari _dealer_ mobil selama setahun. _Dataset_ mencakup sembilan kolom data untuk setiap penjualan mobil. Berikut penjelasan dari setiap kolom yang dimiliki oleh _dataset_ ini.

- **_Date_** merupakan data tanggal penjualan mobil dilakukan.
- **_Salesperson_** merupakan data nama penjual yang melakukan penjualan.
- **_Customer Name_** merupakan data nama pelanggan yang membeli mobil.
- **_Car Make_** merupakan data merk mobil yang dibeli seperti, Toyota, Honda, Ford, Chevrolet, atau Nissan.
- **_Car Model_** merupakan data model dari merk mobil yang dibeli seperti, Corolla, Civic, F-150, Silverado, atau Altima.
- **_Car Year_** merupakan data tahun pembuatan mobil yang dibeli, mulai dari tahun 2010 hingga 2022.
- **_Sale Price_** merupakan data harga mobil yang dijual dalam bentuk USD.
- **_Commission Rate_** merupakan tingkat komisi yang dibayarkan kepada penjual atas penjualan, nilainya berkisar 0,05-0,15.
- **_Commission Earned_** merupakan Jumlah komisi yang diperoleh penjual atas penjualan tersebut.

Kolom '_Car Make_', '_Car Model_' dan '_Car Year_' merupakan variabel yang berhubungan dengan mobil yang dijual.

Kumpulan data ini berguna untuk menganalisis tren penjualan mobil dari waktu ke waktu dan menganalisis dampak dari berbagai faktor pada penjualan mobil. Hal ini yang menjadikan penulis menggunakan _dataset_ ini sebagai sumber data yang digunakan pada proyek.

Contoh visualisasi data yang digunakan adalah diantaranya sebagai berikut:

- _Box Plot_, merupakan bentuk visualisasi statistik yang bertujuan untuk melihat adanya _outlier_ pada data.

[![boxplot.jpg](https://i.postimg.cc/fL7yPVdZ/boxplot.jpg)](https://postimg.cc/qhRphMQZ)

Gambar 1. Contoh bentuk _boxplot_ [[6]](https://eksplorasidatasaham.wordpress.com/2017/02/08/melihat-pola-sebaran-data-saham-analisa-boxplot/).

_Outlier_ adalah nilai yang secara signifikan berbeda dari nilai-nilai lain dalam suatu _dataset_. _Outlier_ bisa merupakan nilai yang sangat tinggi (_outlier_ atas) atau sangat rendah (_outlier_ bawah) dibandingkan dengan sebagian besar nilai lainnya.

- _Line Chart_,merupakan bentuk visualisasi statistik yang bertujuan untuk menampilkan data yang berkelanjutan dengan periode atau rentang nilai tertentu. Garis yang dihasilkan dapat menunjukkan tren kenaikan atau penurunan pada data. Misalnya pada Gambar 2, terdapat empat data penjualan tahunan per kuartal selama 3 bulan yang terlihat mengalami kenaikan setiap bulannya.

[![line-chart.png](https://i.postimg.cc/1zFycnDW/line-chart.png)](https://postimg.cc/cKsqdJ9Y)

Gambar 2. Contoh bentuk _line chart_ [[7]](https://bitlabs.id/blog/diagram-garis-adalah/).

- _Bar Chart_, merupakan bentuk visualisasi statistik yang bertujuan untuk menampilkan perbandingan antara beberapa variabel. Misalnya pada Gambar 3, menunjukkan perbandingan antara banyaknya populasi (sumbu Y) selama tahun 1900 sampai 2009 (sumbu X).

[![barchart.png](https://i.postimg.cc/yx8M4Nd2/barchart.png)](https://postimg.cc/ZBX7646L)

Gambar 3. Contoh bentuk _bar chart_ [[8]](https://id.wikipedia.org/wiki/Templat:Vertical_bar_chart).

Grafik-grafik diatas akan digunakan dalam melakukan visualisasi pada hasil _Exploratory Data Analysis_ (EDA).

## DATA PREPARATION

Tahapan _data preparation_ dapat dilakukan sebagai berikut:

**1. Melakukan EDA (_Exploratory Data Analysis_)**, bertujuan untuk mendapatkan informasi fitur mengenai _dataset_ yang digunakan.

**1a. Menampilkan beberapa isi data yaitu dengan menampilkan 5 data teratas dan 5 data terakhir. Hasilnya dapat dilihat pada Tabel 1 berikut.**

Tabel 1. 5 data teratas dan terbawah dari _Car Sales Dataset_.

| Date       | Salesperson     | Customer Name   | Car Make  | Car Model | Car Year | Sale Price | Commission Rate | Commission Earned |
| ---------- | --------------- | --------------- | --------- | --------- | -------- | ---------- | --------------- | ----------------- |
| 2022-08-01 | Monica Moore MD | Mary Butler     | Nissan    | Altima    | 2018     | 15983      | 0.070495        | 1126.73           |
| 2023-03-15 | Roberto Rose    | Richard Pierce  | Nissan    | F-150     | 2016     | 38474      | 0.134439        | 5172.40           |
| 2023-04-29 | Ashley Ramos    | Sandra Moore    | Ford      | Civic     | 2016     | 33340      | 0.114536        | 3818.63           |
| 2022-09-04 | Patrick Harris  | Johnny Scott    | Ford      | Altima    | 2013     | 41937      | 0.092191        | 3866.20           |
| 2022-06-16 | Eric Lopez      | Vanessa Jones   | Honda     | Silverado | 2022     | 20256      | 0.113490        | 2298.85           |
| 2022-05-26 | Isabella Moore  | Shirley Lee     | Chevrolet | Silverado | 2021     | 49823      | 0.062977        | 3137.70           |
| 2022-10-03 | Kimberly Snow   | Tara Rodgers    | Ford      | F-150     | 2022     | 18803      | 0.068339        | 1284.97           |
| 2022-06-07 | Jessica Young   | Jennifer Moore  | Chevrolet | Civic     | 2010     | 30863      | 0.088915        | 2744.19           |
| 2023-02-15 | Donald Barber   | Ashley Diaz     | Honda     | Silverado | 2014     | 26125      | 0.088260        | 2305.80           |
| 2023-03-24 | Kayla Fowler    | Nathan Thompson | Honda     | Civic     | 2010     | 20762      | 0.137105        | 2846.57           |

Tabel 1 merupakan gabungan dari 5 data teratas dan 5 data terbawah. Hal ini bertujuan untuk mengetahui perwakilan isi data.

**1b. Melakukan pengecekan tipe data pada setiap kolom.**

Tabel 2. Nama kolom dan tipe data dari _dataset_ yang digunakan.
| # | Column | Dtype |
|-----|--------------------|-------------|
| 0 | Date | object |
| 1 | Salesperson | object |
| 2 | Customer Name | object |
| 3 | Car Make | object |
| 4 | Car Model | object |
| 5 | Car Year | int64 |
| 6 | Sale Price | int64 |
| 7 | Commission Rate | float64 |
| 8 |Commission Earned | float64 |

Tabel 2 memberikan informasi tentang struktur dataset yang terdiri dari 9 kolom. Kolom '_Date_', '_Salesperson_', '_Customer Name_', '_Car Make_' dan '_Car Model_' memiliki tipe data `object` yang merupakan tipe data kategori. Sedangkan, kolom '_Car Year_' dan '_Sale Price_' memiliki tipe data `int64` serta kolom '_Commission Rate_' dan '_Commission Earned_' memiliki tipe data `float64` yang merupakan tipe data numerik.

**1c. Menampilkan distribusi data pada kolom yang merupakan tipe data numerik.**

Tabel 3. Distribusi data pada kolom numerik.

|       | Car Year | Sale Price | Commission Rate | Commission Earned |
| ----- | -------- | ---------- | --------------- | ----------------- |
| count | 2.5e+06  | 2.5e+06    | 2.5e+06         | 2.5e+06           |
| mean  | 2015.996 | 30012.18   | 0.09998766      | 3001.005          |
| std   | 3.739    | 11545.14   | 0.02887202      | 1481.467          |
| min   | 2010     | 10000      | 0.05000014      | 501.34            |
| 25%   | 2013     | 20019      | 0.0749645       | 1821.71           |
| 50%   | 2016     | 30006      | 0.1000058       | 2741.91           |
| 75%   | 2019     | 40022      | 0.1250065       | 3978.142          |
| max   | 2022     | 50000      | 0.15            | 7494.53           |

Tabel 3 menyajikan informasi statistik tentang empat kolom numerik dalam _dataset_, yaitu '_Car Year_', '_Sale Price_', '_Commission Rate_', dan '_Commission Earned_'.

- **count**: Jumlah data yang ada dalam setiap kolom.
- **mean**: Nilai rata-rata dari setiap kolom, menunjukkan angka tengah atau pusat distribusi data.
- **std**: Standar deviasi, menggambarkan sejauh mana data tersebar di sekitar nilai rata-rata.
- **min**: Nilai terendah dalam setiap kolom, menunjukkan batas bawah data.
- **25%**: Kuartil pertama, nilai di mana 25% data terendah berada di bawahnya.
- **50%**: Kuartil kedua atau median, nilai di mana 50% data terletak di bawahnya.
- **75%**: Kuartil ketiga, nilai di mana 75% data terendah berada di bawahnya.
- **max**: Nilai tertinggi dalam setiap kolom, menunjukkan batas atas data.

Informasi ini memberikan gambaran tentang sebaran dan karakteristik data dalam kolom-kolom numerik seperti melihat nilai rata-rata, rentang nilai, dan deviasi standar dari '_Sale Price_', yang mengindikasikan variasi harga jual mobil dalam _dataset_.

**1d. Melakukan pengecekan terhadap data yang _null_ atau duplikat, yang bertujuan untuk mengidentifikasi data yang tidak lengkap atau mengandung duplikasi. Hasilnya dapat dilihat pada Tabel 4 berikut.**

Tabel 4. Jumlah data _null_ pada setiap kolom.

| Column            | Null Count |
| ----------------- | ---------- |
| Date              | 0          |
| Salesperson       | 0          |
| Customer Name     | 0          |
| Car Make          | 0          |
| Car Model         | 0          |
| Car Year          | 0          |
| Sale Price        | 0          |
| Commission Rate   | 0          |
| Commission Earned | 0          |

Informasi yang didapatkan dari Tabel 4 adalah bahwa tidak ada nilai _null_ (_missing value_) pada setiap kolom dalam _dataset_. Hal ini menunjukkan bahwa _dataset_ memiliki data lengkap untuk setiap kolom.

**1e. Memeriksa apakah terdapat nilai 0 pada kolom '_Car Year_' dan '_Sale Price_'. Hal ini dilakukan karena tidak mungkin tahun pembuatan dan harga mobil bernilai 0. Hasilnya dapat dilihat pada tabel 5 berikut.**

Tabel 5. Jumlah nilai 0 pada kolom '_Car Year_' dan '_Sale Price_'.

| Column     | Count of 0 |
| ---------- | ---------- |
| Car Year   | 0          |
| Sale Price | 0          |

Berdasarkan Tabel 5 dapat diketahui bahwa tidak ada nilai 0 pada kolom '_Car Year_' dan '_Sale Price_'. Hal ini menunjukkan bahwa dalam _dataset_ memiliki data yang rasional.

**1f. Memeriksa data _outlier_ menggunakan metode IQR (_Interquartile Range_) dan menghapus data yang _outlier_ pada data numerik yaitu '_Car Year_', '_Sale Price_', '_Commission Rate_', dan '_Commission Earned_'.**

[![boxplot-dataoutliers.png](https://i.postimg.cc/PrTvy9X8/boxplot-dataoutliers.png)](https://postimg.cc/XX1v72yV)

Gambar 4. Hasil visualisasi menggunakan _box plot_ untuk melihat data _outlier_.

Berdasarkan Gambar 4, dapat diketahui bahwa ternyata terdapat data _outlier_ pada kolom '_Commission Earned_'. Jumlah dari data _outlier_ dapat dilihat pada Tabel 6.

Tabel 6. Jumlah data _outlier_ pada data numerik.

| Column            | Count of Outliers |
| ----------------- | ----------------- |
| Car Year          | 0                 |
| Sale Price        | 0                 |
| Commission Rate   | 0                 |
| Commission Earned | 3432              |

Berdasarkan Tabel 6, dapat diketahui bahwa pada kolom '_Car Year_', '_Sale Price_' dan '_Commission Rate_' tidak terdapat _outlier_, sedangkan pada kolom '_Commission Earned_' terdapat 3432 data yang dianggap sebagai _outlier_. Hal ini menunjukkan bahwa nilai-nilai pada kolom '_Commission Earned_' memiliki variasi yang lebih besar dibandingkan dengan kolom lainnya. Data outlier tersebut akan dihapus untuk mengurangi bias pada data. Setelah data outlier dihapus maka total data saat ini adalah 2496568. Jumlah data saat ini akan digunakan untuk melihat distribusi data pada setiap kolom dan diteruskan ke tahap pembentukan model.

**1g. Melihat frekuensi data berdasarkan kolom '_Date_'.**

[![frequency-of-date.png](https://i.postimg.cc/fTTt6byf/frequency-of-date.png)](https://postimg.cc/YjZCFM1v)

Gambar 5. Grafik frekuensi data berdasarkan kolom '_Date_'.

Gambar 5 menunjukkan bahwa data yang digunakan memiliki pola data dengan gerakan musiman [[9]](https://amikjtc.com/jurnal/index.php/jurnal/article/view/91/85). Pola data musiman mengacu pada fluktuasi atau pola yang terulang dalam data dengan periode waktu yang tetap atau hampir tetap, yang terkait dengan faktor musiman atau perubahan berulang dalam suatu periode waktu tertentu.

Tabel 7. Frekuensi data berdasarkan tahun.

| Year | Frequency |
| ---- | --------- |
| 2022 | 1672227   |
| 2023 | 824341    |

Berdasarkan Tabel 7, dapat dilihat bahwa pada tahun penjualan lebih banyak terjadi pada tahun 2022 dengan frekuensi sebanyak 1.672.227 data dibandingkan dengan tahun 2023 dengan jumlah frekuensi sebanyak 824.341 data.

**1h. Analisis _sales_ yang paling banyak melakukan penjualan dan _customer_ yang paling banyak melakukan pembelian.**

Tabel 8. Frekuensi penjualan _sales_.

| Salesperson      | Frequency |
| ---------------- | --------- |
| Michael Smith    | 1227      |
| Michael Johnson  | 976       |
| David Smith      | 821       |
| James Smith      | 795       |
| Michael Williams | 752       |

Tabel 8 menunjukkan frekuensi masing-masing _sales_. _Sales_ dengan nama "Michael Smith" memiliki frekuensi tertinggi, yaitu sebanyak 1227 data. Diikuti oleh "Michael Johnson" dengan frekuensi 976, "David Smith" dengan frekuensi 821, "James Smith" dengan frekuensi 795, dan "Michael Williams" dengan frekuensi 752.

Tabel 9. Frekuensi pembelian pada _customer_.

| Customer Name   | Frequency |
| --------------- | --------- |
| Michael Smith   | 1163      |
| Michael Johnson | 889       |
| David Smith     | 795       |
| James Smith     | 788       |
| Jennifer Smith  | 786       |

Tabel 9 menunjukkan frekuensi masing-masing nama pelanggan dalam _dataset_. Nama pelanggan "Michael Smith" memiliki frekuensi tertinggi, yaitu sebanyak 1163 data. Diikuti oleh "Michael Johnson" dengan frekuensi 889, "David Smith" dengan frekuensi 795, "James Smith" dengan frekuensi 788, dan "Jennifer Smith" dengan frekuensi 786.

Berdasarkan Tabel 8 dan Tabel 9, dapat diketahui bahwa ternyata memiliki kesamaan untuk jumlah frekuensi tertinggi. Oleh karena itu, perlu diketahui nama yang terdapat pada kolom '_Salesperson_' dan '_Customer Name_'.

Tabel 10. 10 Contoh Hasil pencarian data '_Salesperson_' dan '_Customer Name_' menggunakan _union function_.

| Nama                |
| ------------------- |
| Wesley Stanton      |
| Christina Khan      |
| Mrs. Janice Edwards |
| Kylie Dunlap        |
| Dalton Yu           |
| Chad Frey           |
| Deborah Mcguire     |
| Martha Cox DDS      |
| Judy Anderson DDS   |
| Lynn Mcclure        |

Pada Tabel 10 merupakan contoh data yang terdapat pada kolom '_Salesperson_' dan '_Customer Name_'. Totalnya terdapat 698362 nama yang merupakan bagian dari _Sales_ dan _Customer_.

**1i. Analisis pada kolom '_Car Make_' dan '_Car Model_' serta mencari merk dan model mobil yang paling mahal dan murah.**

Tabel 11. Frekuensi data pada kolom '_Car Make_'.

| Car Make  | Frequency |
| --------- | --------- |
| Honda     | 500048    |
| Chevrolet | 499719    |
| Toyota    | 499491    |
| Ford      | 499072    |
| Nissan    | 498238    |

Tabel 11 menunjukkan frekuensi masing-masing merek mobil. Merek "Honda" memiliki frekuensi tertinggi, yaitu sebanyak 500.048 data. Diikuti oleh merek "Chevrolet" dengan frekuensi 499.719, "Toyota" dengan frekuensi 499.491, "Ford" dengan frekuensi 499.072, dan "Nissan" dengan frekuensi 498.238. Dari data tersebut, dapat diketahui bahwa merek "Honda" merupakan merek mobil yang paling banyak diminati.

Tabel 12. Frekuensi data pada kolom '_Car Model_'.

| Car Model | Frequency |
| --------- | --------- |
| Silverado | 500114    |
| Civic     | 499833    |
| Corolla   | 499483    |
| F-150     | 498984    |
| Altima    | 498154    |

Tabel 12 menunjukkan frekuensi masing-masing model mobil. Model "Silverado" memiliki frekuensi tertinggi, yaitu sebanyak 500.114 data. Diikuti oleh model "Civic" dengan frekuensi 499.833, "Corolla" dengan frekuensi 499.483, "F-150" dengan frekuensi 498.984, dan "Altima" dengan frekuensi 498.154. Dari data tersebut, dapat diketahui bahwa model "Silverado" merupakan model mobil yang paling banyak diminati.

[![car-distribution.png](https://i.postimg.cc/QxgCx3WB/car-distribution.png)](https://postimg.cc/crH0XPhZ)

Gambar 6. Grafik penjualan mobil berdasarkan tahun pembuatan mobil.

Berdasarkan Gambar 6, dapat diketahui bahwa mobil dengan pembuatan tahun 2013 merupakan yang paling laku dan tahun 2022 merupakan tahun mobil yang paling sedikit dibandingkan yang lainnya.

Tabel 13. Data mobil paling mahal dan murah.

| Merk Mobil | Model Mobil | Tahun Mobil | Harga  |
| ---------- | ----------- | ----------- | ------ |
| Nissan     | F-150       | 2013        | 50,000 |
| Nissan     | Altima      | 2012        | 10,000 |

Tabel 13 menunjukkan informasi tentang mobil dengan harga paling mahal dan paling murah dalam _dataset_. Mobil dengan harga paling mahal adalah Nissan F-150 tahun 2013 dengan harga 50,000, sedangkan mobil dengan harga paling murah adalah Nissan Altima tahun 2012 dengan harga 10,000.

**2. Melakukan drop pada kolom yang tidak relevan untuk diproses ke model seperti kolom '_Date_', '_Salesperson_', '_Customer Name_' dan '_Year_'.**

Kolom-kolom tersebut dihapus dikarenakan alasan berikut:

- '_Date_': Tanggal penjualan dianggap tidak relevan dalam memprediksi harga mobil, karena tanggalnya tidak memberikan informasi yang signifikan terkait faktor-faktor yang mempengaruhi harga mobil seperti merk, model, atau tahun mobil.

- '_Salesperson_' dan '_Customer Name_': Nama penjual dan nama pelanggan tidak memberikan kontribusi yang signifikan terhadap prediksi harga mobil.

- '_Year_': Kolom '_Year_' merupakan sebuah kolom baru yang terbentuk dari kolom '_Date_'. Sehingga kolom ini merupakan bagian yang sama dari kolom '_Date_'.

Dengan menghapus kolom-kolom tersebut, fokus pembentukan model untuk prediksi dapat difokuskan pada variabel yang dianggap lebih penting dan memiliki hubungan yang lebih erat dengan harga mobil, seperti merk, model, tahun, dan atribut lain yang dianggap lebih relevan.

**3. Melakukan Teknik _Encoding_ pada variabel kategorikal yaitu kolom '_Car Make_' dan '_Car Model_'.**

Teknik _encoding_ yang digunakan disebut sebagai "_One-Hot Encoding_" dengan menggunakan fungsi `get_dummies()` dari _library_ pandas. Teknik ini digunakan untuk mengubah variabel kategorikal menjadi representasi numerik dalam bentuk kolom biner (0 dan 1) [[10]](https://www.mdpi.com/1099-4300/23/10/1258).Kolom '_Car Make_' dan '_Car Model_' dijadikan sebagai variabel kategorikal yang akan di-_encode_ sehingga hasilnya dapat dilihat pada Tabel 14.

Tabel 14. Contoh hasil penerapan metode _One-Hot Encoding_ pada data.

| Car Year | Sale Price | Commission Rate | Commission Earned | Car Make_Chevrolet | Car Make_Ford | Car Make_Honda | Car Make_Nissan | Car Make_Toyota | Car Model_Altima | Car Model_Civic | Car Model_Corolla | Car Model_F-150 | Car Model_Silverado |
| -------- | ---------- | --------------- | ----------------- | ------------------ | ------------- | -------------- | --------------- | --------------- | ---------------- | --------------- | ----------------- | --------------- | ------------------- |
| 2018     | 15983      | 0.070495        | 1126.73           | 0                  | 0             | 0              | 1               | 0               | 1                | 0               | 0                 | 0               | 0                   |
| 2016     | 38474      | 0.134439        | 5172.40           | 0                  | 0             | 0              | 1               | 0               | 0                | 0               | 0                 | 1               | 0                   |
| 2016     | 33340      | 0.114536        | 3818.63           | 0                  | 1             | 0              | 0               | 0               | 0                | 1               | 0                 | 0               | 0                   |
| 2013     | 41937      | 0.092191        | 3866.20           | 0                  | 1             | 0              | 0               | 0               | 1                | 0               | 0                 | 0               | 0                   |
| 2022     | 20256      | 0.113490        | 2298.85           | 0                  | 0             | 1              | 0               | 0               | 0                | 0               | 0                 | 0               | 1                   |

Tabel 14 merupakan representasi data setelah dilakukan _encoding_ menggunakan metode _One-Hot Encoding_. Kolom-kolom dengan nama '_Car Make_' dan '_Car Model_' dipecah menjadi kolom-kolom baru sesuai dengan nilai unik yang ada. Nilai 1 pada kolom-kolom tersebut menunjukkan keberadaan nilai tersebut dalam data, sedangkan nilai 0 menunjukkan ketiadaan nilai tersebut. Hal ini memudahkan dalam representasi data kategorikal menjadi data numerik yang dapat digunakan dalam pemodelan _machine learning_.

**4. Melakukan _Feature Scaling_ pada data numerik yaitu kolom '_Car Year_', '_Commission Rate_', dan '_Commission Earned_' menggunakan `MinMaxScaler()`.**

Metode _MinMaxScaler_ bekerja dengan mengubah setiap nilai dalam kolom menjadi nilai antara 0 dan 1, dengan mempertahankan proporsi relatif antara nilai-nilai tersebut. Tujuan dari _Feature Scaling_ adalah untuk memastikan bahwa semua fitur atau kolom memiliki pengaruh yang seimbang pada proses pemodelan dan mencegah fitur yang memiliki rentang nilai yang lebih besar mendominasi fitur lainnya.

Dengan melakukan Feature Scaling pada data numerik dapat membantu meningkatkan performa model dalam beberapa cara seperti berikut [[11]](https://ieeexplore.ieee.org/abstract/document/9898687) :

1. Menghilangkan perbedaan skala: Beberapa algoritma _machine learning_, seperti regresi Linier dan KNN, mengasumsikan bahwa semua fitur memiliki skala yang serupa. Jika fitur-fitur memiliki skala yang berbeda, algoritma tersebut mungkin tidak dapat memberikan hasil yang optimal. Dengan melakukan _Feature Scaling_ dapat menghilangkan perbedaan skala tersebut sehingga algoritma dapat bekerja dengan lebih baik.

2. Mencegah dominasi fitur: Jika terdapat fitur-fitur dengan rentang nilai yang lebih besar daripada fitur lainnya, algoritma cenderung akan memberikan bobot yang lebih besar pada fitur tersebut. Hal ini dapat menyebabkan fitur-fitur lain yang memiliki rentang nilai yang lebih kecil menjadi kurang berpengaruh dalam pembentukan model. Dengan melakukan _Feature Scaling_ dapat mencegah fitur-fitur dengan rentang nilai yang lebih besar mendominasi fitur-fitur lainnya, sehingga setiap fitur memiliki kontribusi yang seimbang dalam model.

3. Meningkatkan kecepatan konvergensi: Beberapa algoritma, seperti regresi menggunakan metode optimasi yang membutuhkan konvergensi yang cepat. Dengan melakukan _Feature Scaling_, skala data yang serupa dapat membantu algoritma mencapai konvergensi lebih cepat. Hal ini dapat menghemat waktu dan sumber daya komputasi yang diperlukan dalam proses pemodelan.

Dengan melakukan _Feature Scaling_, data numerik memiliki skala yang serupa dan memenuhi asumsi algoritma _machine learning_.

**5. Memisahkan kolom fitur dan target.**

Kolom fitur terdiri dari semua kolom _dataset_ yang telah melewati tahapan sebelumnya kecuali kolom '_Sale Price_', sedangkan kolom target adalah kolom '_Sale Price_'. Tujuan dari pemisahan ini adalah untuk mempersiapkan data yang akan digunakan dalam proses pembelajaran dan prediksi pada model _machine learning_. Berdasarkan tujuan dari pelaksanaan proyek ini, maka kolom target yakni 'Sale Price' sebagai variabel yang ingin diprediksi.

**6. Setelah melewati tahapan sebelumnya, kemudian _dataset_ dibagi menjadi 80% data latih dan 20% data uji.**

Data latih akan digunakan untuk melatih model, sedangkan data uji digunakan untuk menguji seberapa baik model yang telah dilatih dapat melakukan prediksi pada data baru.

## MODELLING

Dalam proyek ini, penulis menggunakan tiga algoritma yaitu Regresi Linier, _Decision Tree_, dan KNN (_K-Nearest Neighbors_) untuk memodelkan prediksi harga mobil. Algoritma-algoritma ini dipilih berdasarkan karakteristik dan kelebihan masing-masing seperti yang telah dijelaskan sebelumnya. Berikut ini adalah penjelasan lebih rinci tentang masing-masing algoritma beserta nilai parameter default yang digunakan:

1. Regresi Linier:
   Regresi Linier adalah algoritma sederhana yang digunakan untuk memodelkan hubungan linier antara variabel independen dan variabel dependen. Algoritma ini cocok untuk memprediksi nilai kontinu seperti harga mobil. Kelebihan Regresi Linier adalah kemampuannya yang mudah diinterpretasikan dan memberikan pemahaman yang jelas tentang bagaimana setiap fitur berkontribusi terhadap prediksi harga mobil. Namun, Regresi Linier tidak dapat menangkap hubungan non-linier antara variabel dan rentan terhadap asumsi yang tidak terpenuhi. Dalam penggunaan _default_-nya, Regresi Linier tidak memerlukan parameter tambahan [[12]](https://jastt.org/index.php/jasttpath/article/view/57).

2. _Decision Tree_:
   _Decision Tree_ adalah algoritma yang memodelkan keputusan berdasarkan serangkaian pemilihan fitur yang paling informatif pada setiap langkahnya. Algoritma ini dapat menangani data dengan jenis variabel campuran dan tidak memerlukan _preprocessing_ data yang rumit. Kelebihan _Decision Tree_ adalah kemudahannya dalam dipahami dan diinterpretasikan. Namun, algoritma ini cenderung _overfitting_ jika tidak diatur dengan baik dan tidak stabil terhadap perubahan kecil pada data pelatihan. Dalam penggunaan _default_-nya, parameter yang digunakan adalah kriteria pemilihan fitur menggunakan _Entropy_ dengan nilai _default_ dari parameter tersebut adalah 0. _Entropy_ adalah metrik yang mengukur tingkat ketidakberaturan dalam data. Dalam algoritma _Decision Tree_, _Entropy_ digunakan untuk memilih fitur yang paling informatif dalam memisahkan data. Semakin tinggi nilai _Entropy_, semakin acak dan tidak teratur data tersebut. Dengan menggunakan _Entropy_, _Decision Tree_ mencari fitur yang dapat memisahkan data dengan lebih baik berdasarkan tingkat ketidakberaturan [[13]](https://ieeexplore.ieee.org/abstract/document/5991826).

3. KNN (_K-Nearest Neighbors_):
   KNN adalah algoritma yang menggunakan tetangga terdekat dari suatu data untuk melakukan prediksi. Algoritma ini tidak mengasumsikan distribusi data tertentu dan mampu menangani data dengan variasi tinggi. Kelebihan KNN adalah sederhana dalam implementasinya dan tidak memerlukan asumsi yang kuat tentang data. Namun, KNN memerlukan komputasi yang lebih tinggi karena melibatkan perhitungan jarak antara data, sensitif terhadap skala data, dan memerlukan pemilihan parameter K yang tepat. Dalam penggunaan _default_-nya, parameter yang digunakan adalah nilai `K=5` dan metrik jarak menggunakan _Euclidean Distance_. _Euclidean Distance_ digunakan untuk mengukur seberapa dekat atau seberapa jauh suatu titik dengan tetangganya dalam ruang fitur [[14]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916348/).

Dengan menggunakan nilai parameter _default_ pada ketiga algoritma, maka kita dapat membandingkan performa dan karakteristik masing-masing algoritma secara objektif.

## EVALUATION

Dalam proyek ini, matriks evaluasi yang digunakan adalah _Mean Squared Error_ (MSE) dan _Mean Absolute Error_ (MAE). MSE digunakan sebagai ukuran kesalahan antara nilai prediksi dan nilai sebenarnya dalam kasus regresi. Semakin kecil nilai MSE, semakin baik model dalam melakukan prediksi yang akurat. MAE juga digunakan sebagai ukuran kesalahan, tetapi memberikan perhitungan kesalahan yang lebih "absolut" daripada MSE.

Dalam evaluasi model, tiga algoritma yaitu Regresi Linier, _Decision Tree_, dan KNN telah dievaluasi menggunakan kedua metrik tersebut. Dengan membandingkan nilai MSE dan MAE dari masing-masing algoritma maka dapat ditentukan algoritma mana yang memberikan prediksi yang lebih akurat dan sesuai dengan tujuan proyek. Hasil dari penggunaan MSE dan MAE dapat dilihat pada Tabel 15:

Tabel 15. Hasil evaluasi MSE dan MAE.

|     Model      |      MSE       |    MAE    |
| :------------: | :------------: | :-------: |
| Regresi Linier | 25429158814.05 | 137809.55 |
| Decision Tree  |  532215260.84  | 19978.21  |
|      KNN       |  458917127.95  | 18142.63  |

Dari hasil evaluasi pada Tabel 15, dapat dilihat bahwa model Regresi Linier memiliki MSE yang sangat tinggi, menunjukkan tingkat kesalahan yang besar dalam memprediksi harga mobil. Model _Decision Tree_ dan KNN memiliki MSE yang lebih rendah, menunjukkan performa yang lebih baik dalam memprediksi harga mobil.

Dalam hal MAE, model _Decision Tree_ dan KNN memiliki tingkat kesalahan yang lebih rendah dibandingkan dengan model Regresi Linier, menunjukkan keakuratan yang lebih baik dalam memprediksi harga mobil. Oleh karena itu, untuk dataset ini, model _Decision Tree_ dan KNN lebih disarankan untuk digunakan dalam melakukan prediksi harga mobil.

Dari hasil pengujian prediksi harga mobil yang dilakukan, dapat dilihat bahwa setiap model memiliki prediksi harga penjualan mobil yang berbeda. Prediksi Regresi Linier menghasilkan harga penjualan mobil sebesar 13125, prediksi Decision Tree menghasilkan harga 10008, dan prediksi KNN menghasilkan harga 10142.2.

Berdasarkan hasil pengujian prediksi harga mobil tersebut, dapat diketahui hasil prediksi pada algoritma _Decision Tree_ yang paling mendekati kemudian disusul oleh KNN dan Regresi Linier. Meski begitu, hasil prediksi masih jauh dari nilai asli. Berikut fitur yang paling berpengaruh dalam menentukan harga mobil dapat dilihat pada Tabel 16.

Tabel 16. Fitur yang paling berpengaruh dalam menentukan harga mobil.

| Fitur               | Importance Score      |
| ------------------- | --------------------- |
| Commission Earned   | 0.7083596131830072    |
| Commission Rate     | 0.2916393378207174    |
| Car Year            | 3.55698297028073e-07  |
| Car Make_Honda      | 7.013526072413751e-08 |
| Car Make_Chevrolet  | 7.011923623450332e-08 |
| Car Make_Nissan     | 6.991766843127469e-08 |
| Car Model_Corolla   | 6.962389941150295e-08 |
| Car Make_Toyota     | 6.95229435721429e-08  |
| Car Model_Altima    | 6.931034801210895e-08 |
| Car Make_Ford       | 6.904791528382239e-08 |
| Car Model_F-150     | 6.888334649110333e-08 |
| Car Model_Civic     | 6.875956488531319e-08 |
| Car Model_Silverado | 6.7977795174018e-08   |

Hasil yang didapatkan menunjukkan skor pentingnya fitur dalam menentukan harga mobil. Setiap fitur memiliki skor penting yang dinyatakan sebagai angka.

Misalnya, '_Commission Earned_' memiliki skor penting sebesar 0.7083596164141285, yang berarti fitur tersebut memiliki pengaruh yang paling signifikan dalam menentukan harga mobil. Skor penting ini dapat diinterpretasikan sebagai persentase kontribusi fitur tersebut dalam mempengaruhi data pada kolom '_Sale Price_'.

Selanjutnya, '_Commission Rate_' memiliki skor penting sebesar 0.2916393396038089, yang menunjukkan bahwa fitur ini juga memiliki pengaruh yang cukup signifikan dalam menentukan harga mobil, meskipun tidak sebesar '_Commission Earned_'.

Fitur-fitur lainnya, seperti '_Car Year_', '_Car Model_' dan '_Car Make_' memiliki skor penting yang sangat kecil, bahkan mendekati nol. Hal ini menunjukkan bahwa fitur-fitur tersebut memiliki pengaruh yang relatif kecil atau kurang signifikan dalam memprediksi harga mobil.

## CONCLUSION

Berdasarkan analisis, fitur yang paling berpengaruh terhadap harga penjualan mobil (_Sale Price_) adalah '_Commission Earned_' (0.7083596131830072) dan '_Commission Rate_' (0.2916393378207174). Hal ini menunjukkan bahwa komisi yang diperoleh dan tingkat komisi yang diterapkan memiliki pengaruh yang signifikan terhadap harga penjualan mobil. Fitur-fitur lainnya seperti '_Car Year_' dan '_Car Make_' memiliki pengaruh yang lebih kecil dalam menentukan harga penjualan mobil.

Kemudian, berdasarkan hasil evaluasi model _machine learning_ yang dilakukan, dapat disimpulkan bahwa pendekatan _machine learning_ dengan menggunakan model _Decision Tree_ dan KNN dapat digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur yang berpengaruh. Model _Decision Tree_ dan KNN menunjukkan performa yang lebih baik dengan MSE yang lebih rendah dan tingkat kesalahan (MAE) yang lebih kecil dibandingkan dengan model Regresi Linier. Oleh karena itu, pendekatan _machine learning_ ini dapat memberikan prediksi yang lebih akurat dalam menentukan harga mobil berdasarkan fitur-fitur yang paling berpengaruh seperti '_Commission Earned_' dan '_Commission Rate_'.

## REFERENSI

[[1]](https://www.bps.go.id/indikator/indikator/view_data_pub/0000/api_pub/aDJqeEc4MTJlOWRaRWFOQ1Q4UE5UQT09/da_15/1) B. P. Statistik, “Laju Pertumbuhan Produk Domestik Bruto Atas Dasar Harga Konstan 2010 Menurut Lapangan Usaha (persen), 2022,” Badan Pusat Statistik, https://www.bps.go.id/indikator/indikator/view_data_pub/0000/api_pub/aDJqeEc4MTJlOWRaRWFOQ1Q4UE5UQT09/da_15/1 (accessed Jul. 5, 2023).

[[2]](https://kemenperin.go.id/artikel/22994/Produksi-Mobil-Lampaui-Target,-Penjualan-Naik-71-Persen) K. Perindustrian, “Kemenperin: Produksi Mobil Lampaui Target, Penjualan Naik 71 persen,” Kementerian Perindustrian, https://kemenperin.go.id/artikel/22994/Produksi-Mobil-Lampaui-Target,-Penjualan-Naik-71-Persen (accessed Jul. 5, 2023).

[[3]](https://dataindonesia.id/sektor-riil/detail/kinerja-industri-otomotif-melonjak-1067-pada-2022) S. Sadya, “Kinerja Industri Otomotif melonjak 10,67% PADA 2022,” Dataindonesia.id, https://dataindonesia.id/sektor-riil/detail/kinerja-industri-otomotif-melonjak-1067-pada-2022 (accessed Jul. 5, 2023).

[[4]](https://ejournal.unsrat.ac.id/index.php/emba/article/view/21893) G. A. Taroreh, L. Mananeke, and F. Roring, “ANALISIS STRATEGI PEMASARAN DALAM MENINGKATKAN VOLUME PENJUALAN MOBIL MITSUBISHI XPANDER PADA PT. BOSOWA BERLIANMOTOR KAIRAGI MARKETING STRATEGY ANALYSIS IN INCREASING THE SALES VOLUME OF MITSUBISHI XPANDER CARS AT PT. BOSOWA BERLIAN MOTOR KAIRAGI,” Analisis Strategi…… 3683 Jurnal EMBA, vol. 6, no. 4, pp. 3683–3692, 2018.

[[5]](https://slims.ahmaddahlan.ac.id/index.php?p=fstream-pdf&fid=40&bid=3192) I. Daqiqil, MACHINE LEARNING: Teori, Studi Kasus Dan Implementasi Menggunakan Python. Riau, Universitas Riau: UR Press, 2021.

[[6]](https://eksplorasidatasaham.wordpress.com/2017/02/08/melihat-pola-sebaran-data-saham-analisa-boxplot/) Ts. Priyadi, “Melihat Pola Sebaran data saham (Analisa Boxplot),” Perjalanan Menuju Happy Investing, https://eksplorasidatasaham.wordpress.com/2017/02/08/melihat-pola-sebaran-data-saham-analisa-boxplot/ (accessed Jul. 6, 2023).

[[7]](https://bitlabs.id/blog/diagram-garis-adalah/) Minlab, “Diagram Garis: Pengertian, Cara Membuat, & Serba-Serbinya” Bitlabs, https://bitlabs.id/blog/diagram-garis-adalah/ (accessed Jul. 10, 2023).

[[8]](https://id.wikipedia.org/wiki/Templat:Vertical_bar_chart) W. Ensiklopedia, “Templat:Vertical Bar Chart,” Wikipedia, https://id.wikipedia.org/wiki/Templat:Vertical_bar_chart (accessed Jul. 6, 2023).

[[9]](https://amikjtc.com/jurnal/index.php/jurnal/article/view/91/85) K. Nugroho, “MODEL ANALISIS PREDIKSI MENGGUNAKAN METODE FUZZY TIME SERIES,” INFOKAM: Informasi Komputer Akuntansi dan Manajemen, vol. 12, no. 1, pp. 46–50, 2016.

[[10]](https://www.mdpi.com/1099-4300/23/10/1258) T. Al-Shehari and R. A. Alsowail, “An Insider Data Leakage Detection Using One-Hot Encoding, Synthetic Minority Oversampling and Machine Learning Techniques,” Entropy, vol. 23, no. 10, p. 1258, Sep. 2021, doi: 10.3390/e23101258.

[[11]](https://ieeexplore.ieee.org/abstract/document/9898687) D. U. Ozsahin, M. Taiwo Mustapha, A. S. Mubarak, Z. Said Ameen and B. Uzun, "Impact of feature scaling on machine learning models for the diagnosis of diabetes," 2022 International Conference on Artificial Intelligence in Everything (AIE), Lefkosa, Cyprus, pp. 87-94, 2022, doi: 10.1109/AIE57029.2022.00024.

[[12]](https://jastt.org/index.php/jasttpath/article/view/57) D. Maulud and A. M. Abdulazeez, “A Review on Linear Regression Comprehensive in Machine Learning”, JASTT, vol. 1, no. 4, pp. 140-147, Dec. 2020.

[[13]](https://ieeexplore.ieee.org/abstract/document/5991826) A. Navada, A. N. Ansari, S. Patil and B. A. Sonkamble, "Overview of use of decision tree algorithms in machine learning," 2011 IEEE Control and System Graduate Research Colloquium, Shah Alam, Malaysia, pp. 37-42, 2011, doi: 10.1109/ICSGRC.2011.5991826.

[[14]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916348/) Z. Zhang, “Introduction to machine learning: K-Nearest Neighbors,” Annals of Translational Medicine, vol. 4, no. 11, pp. 218–218, 2016. doi:10.21037/atm.2016.03.37

---Ini adalah bagian akhir laporan---
