from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from bs4 import BeautifulSoup
from django.urls import reverse
import requests
import pandas as pd
import numpy as np
import csv
import os
# Importing the libraries
from urllib.request import urlopen
from time import time
from django.views.decorators.csrf import csrf_exempt
import time


status_desc = ''
tag_h1 = 0
has_og = 0
has_title = 0
has_alt = 0
load_status = 0
load_num = 0
desc_con = ''


@csrf_exempt
def indexs(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


@csrf_exempt
def about(request):
    template = loader.get_template('about.html')
    return HttpResponse(template.render())


@csrf_exempt
def diagnose(request):

    # Get input value from url input
    data = request.POST.get('url')

    try:
        # Turn url text to lxml
        url_text = requests.get(data).text
        soup = BeautifulSoup(url_text, 'lxml')

        # Get Meta Title
        title = soup.find('title').text
        title_count = len(title)
        stats_title = ''

        if title is not None:
            if title_count < 35:
                has_title = 0
                stats_title = 'Too Short'
            else:
                has_title = 1
        else:
            has_title = 0

        # Get Description
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc is None:
            status_desc = 0
        else:
            desc_con = soup.find(
                'meta', attrs={'name': 'description'}).get('content')
            jumlah_desc = len(desc_con)

            # Cek panjang karakter meta description
            # Jika dibawah 120 maka terlalu pendek, jika diataas 170 terlalu panjang
            if jumlah_desc < 120:
                status_desc = 1
            elif jumlah_desc > 170:
                status_desc = 2
            elif jumlah_desc == 0:
                status_desc = 0
            elif jumlah_desc > 120 and jumlah_desc < 170:
                status_desc = 3

        # Cek Image and alt
        images = soup.find_all('img', alt=True)
        noalt = []
        i = 0

        while i < len(images):
            if images[i]['alt'] == "":
                noalt.append(images[i]['src'])

            i += 1

        if noalt == []:
            has_alt = 1
        else:
            has_alt = 0

        # Cek OG Image
        og_image = soup.find('meta', property='og:image')
        if og_image is None:
            has_og = 0
        else:
            og_content = soup.find('meta', property='og:image').get('content')
            if og_content is "":
                has_og = 0
            else:
                has_og = 1

        def check_heading_order(url):
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    return "Failed to retrieve the webpage."

                soup = BeautifulSoup(response.text, 'html.parser')
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

                # Inisialisasi variabel untuk melacak keberadaan h1
                h1_found = False

                for heading in headings:
                    if heading.name == 'h1':
                        if h1_found:
                            issues = "The webpage has more than one <h1> tag."
                            return (0, issues)
                        h1_found = True
                    else:
                        if not h1_found:
                            issues = "The webpage has a non-h1 heading"
                            return (0, issues)

                min_heading_level = None  # Tingkat heading terendah yang muncul di halaman

                for heading in headings:
                    current_heading_level = int(heading.name[1])

                    if min_heading_level is None:
                        min_heading_level = current_heading_level

                    if current_heading_level < min_heading_level:
                        issues = f"The webpage has a heading hierarchy issue"
                        return (0, issues)

                issues = f"The webpage has valid heading tag order"
                return (1, issues)

            except Exception as e:
                return str(e)

        result = check_heading_order(data)

        # Fungsi untuk mengukur waktu pemuatan situs web

        def measure_load_time(url):
            start_time = time.time()
            response = requests.get(url)
            end_time = time.time()

            if response.status_code != 200:
                return "Failed to retrieve the webpage."

            return end_time - start_time

        # Memanggil fungsi untuk mengukur waktu pemuatan
        load_time = measure_load_time(data)

        if load_time == 'Failed to retrieve the webpage.':
            load_num = 0
            load_status = 'Error'
        else:
            if round(load_time, 2) < 2.5:
                load_status = 'Cepat'
                load_num = 2
            elif round(load_time, 2) > 2.5 and round(load_time, 2) < 4:
                load_status = 'Sedang'
                load_num = 1
            elif round(load_time, 2) > 4:
                load_status = 'Lambat'
                load_num = 0

        # Perhitungan Decision Tree
        # Ambil dataset
        workpath = os.path.dirname(os.path.abspath(__file__))
        seo = open(os.path.join(workpath, 'assets/dataset_v6.csv'), 'rb')
        seo = pd.read_csv(seo)

        # Fungsi Menghitung Entropy
        def entropy(data):
            _, counts = np.unique(data, return_counts=True)
            probabilities = counts / counts.sum()
            return -np.sum(probabilities * np.log2(probabilities))

        # # Fungsi untuk menghitung information gain
        def information_gain(data, feature_name, target_name):
            total_entropy = entropy(data[target_name])
            values, counts = np.unique(data[feature_name], return_counts=True)
            weighted_entropy = -np.sum((counts / counts.sum()) * [entropy(data.where(
                data[feature_name] == val).dropna()[target_name])
                for val in values])
            return total_entropy - weighted_entropy

        # # Fungsi untuk memilih atribut terbaik untuk splitting
        def choose_best_attribute(data, features, target_name):
            information_gains = [information_gain(
                data, feature, target_name) for feature in features]
            return features[np.argmax(information_gains)]

        # # Fungsi untuk membangun pohon keputusan
        def decision_tree(
                data, features, target_name, parent_node_class=None):
            if len(np.unique(data[target_name])) <= 1:
                return np.unique(data[target_name])[0]
            elif len(data) == 0:
                return np.unique(parent_node_class)[0]
            elif len(features) == 0:
                return parent_node_class
            else:
                parent_node_class = np.unique(data[target_name])[np.argmax(
                    np.unique(data[target_name], return_counts=True)[1])]
                best_attribute = choose_best_attribute(
                    data, features, target_name)
                tree = {best_attribute: {}}
                features = [i for i in features if i != best_attribute]
                for value in np.unique(data[best_attribute]):
                    subset_data = data.where(
                        data[best_attribute] == value).dropna()
                    subtree = decision_tree(
                        subset_data, features, target_name, parent_node_class)
                    tree[best_attribute][value] = subtree
                return tree

        # penggunaan algoritma C4.5
        target_column_name = 'Tingkat Optimal'
        features = [col for col in seo.columns if col != target_column_name]

        decision_tree = decision_tree(seo, features, target_column_name)

        # Fungsi untuk membuat prediksi dengan pohon keputusan
        def predict(tree, data):
            for key in tree.keys():
                value = data[key]
                tree = tree[key][value]
                prediction = 0
                if type(tree) is dict:
                    prediction = predict(tree, data)
                else:
                    prediction = tree
                    break
            return prediction

        # Penggunaan Decision Tree pada data baru
        new_data = {
            'Load Speed': load_num,
            'Heading Tag Order': result[0],
            'Meta Title': has_title,
            'Description': status_desc,
            'OG Image': has_og,
            'Alt. Text': has_alt,
        }

        predicted_class = predict(decision_tree, new_data)

        def switch(predicted_class):
            if predicted_class == 4.0:
                return "Optimizing"
            elif predicted_class == 3.0:
                return "Advanced"
            elif predicted_class == 2.0:
                return "Progressing"
            elif predicted_class == 1.0:
                return "Intermediate"
            elif predicted_class == 0.0:
                return "Foundational"

    except requests.exceptions.RequestException as e:
        # Handle the case when scraping fails
        return render(request, '404.html')  # Redirect to a 404

    except KeyError as e:
        # Handle the case when scraping fails
        return render(request, '404.html')

    # print(f'Predicted Class: {switch(predicted_class)}')

    # Reccomendation List
    heading_recom = [
        ('Gunakan Heading Tags Sesuai Hirarki', 'Pastikan tag-heading digunakan dengan benar sesuai hirarki, yaitu <h1> sebagai judul utama, diikuti oleh <h2>, <h3>, dan seterusnya. Ini membantu dalam membuat halaman lebih terstruktur dan mudah dibaca.'),
        ('Gunakan Heading Tags secara Konsisten',
            'Pastikan penggunaan tag-heading konsisten di seluruh halaman web. Ini berarti jangan lompat dari <h1> ke <h3> tanpa ada <h2> di antaranya.'),
        ('Optimalkan untuk SEO', 'Saat merapikan urutan tag-heading, pertimbangkan juga faktor SEO. Pastikan judul halaman menggunakan <h1>, dan tag-heading lainnya mempertimbangkan kata kunci dan optimasi SEO.'),
    ]

    title_recom = [
        ('Title Relevan', 'Buat meta title yang relevan dengan isi halaman. Pastikan judul mencerminkan konten yang sebenarnya di halaman tersebut.'),
        ('Gunakan Kata Kunci pada Title',
            'Gunakan kata kunci yang relevan dalam meta title, tetapi jangan berlebihan. Meta title yang terlalu penuh dengan kata kunci dapat merugikan performa SEO Anda.'),
        ('Menarik dan Informatif', 'Menjadi bagian yang tampil di mesin pencari, buat meta title yang menarik dan informatif, sehingga pengguna akan tertarik untuk mengkliknya.'),
        ('Panjang Ideal', 'Batasi panjang meta title agar sesuai dengan pedoman mesin pencari. Panjang meta title yang ideal adalah sekitar 50-60 karakter.'),
        ('Setiap Halaman Unik', 'Usahakan agar setiap halaman memiliki meta title yang unik. Ini membantu dalam membedakan halaman-halaman Anda di hasil pencarian.'),
    ]

    desc_recom = [
        ('Deskripsi Relevan', 'Pastikan meta description Anda relevan dengan konten halaman. Ini membantu pengguna memahami apa yang diharapkan saat mereka mengklik tautan.'),
        ('Gunakan Kata Kunci',
            'Meskipun meta description bukan lagi faktor peringkat langsung, penggunaan kata kunci yang relevan dalam deskripsi masih penting. Kata kunci ini akan ditampilkan tebal dalam hasil pencarian.'),
        ('Menarik dan Informatif', 'Meta description harus menarik dan informatif. Cobalah untuk menjadikannya singkat dan jelas, menjelaskan dengan baik apa yang pengguna akan temukan di halaman tersebut.'),
        ('Panjang Ideal', 'Mesin telusur umumnya menampilkan sekitar 150-170 karakter dari meta description di hasil pencarian. Oleh karena itu, pastikan deskripsi Anda berada dalam batasan karakter ini.'),
        ('Perbarui Secara Berkala',
            'Jika isi halaman berubah, perbarui juga meta description-nya agar tetap relevan.'),
    ]

    alt_recom = [
        ('Deskripsi yang Jelas dan Relevan', 'Alternative text harus memberikan deskripsi yang jelas dan relevan tentang gambar. Gunakan kata-kata yang memberikan pemahaman tentang apa yang ada dalam gambar. '),
        ('Hindari Alternative Text yang Terlalu Panjang',
            'Alternative text sebaiknya singkat dan langsung ke point. Hindari deskripsi yang terlalu panjang dan rinci.'),
        ('Gunakan Alternative Text pada Semua Gambar',
            'Pastikan semua gambar yang membantu dalam pemahaman atau konten halaman memiliki alternative text. Ini termasuk gambar utama, ikon, grafik, dan lainnya.'),
        ('Deskripsi Khusus untuk Gambar Grafis atau Grafik Data',
            'Jika gambar adalah grafik atau grafik data, berikan alternative text yang merinci informasi yang terdapat dalam gambar. Misalnya, "Grafik lingkaran menunjukkan perbandingan penjualan tahunan.'),
    ]

    og_recom = [
        ('Ukuran Gambar yang Tepat', 'Pastikan gambar OG memiliki ukuran yang sesuai. Mesin pencari sosial dan platform media sosial biasanya memiliki rekomendasi ukuran gambar tertentu (misalnya, 1200 x 630 piksel untuk gambar OG di Facebook).'),
        ('Gambar yang Berkualitas Tinggi',
            'Pastikan gambar OG yang Anda pilih berkualitas tinggi. Ini akan memberikan tampilan yang lebih profesional dan menarik saat tautan dibagikan.'),
        ('Optimalkan untuk Kecepatan Pemuatan',
            'Pastikan gambar OG di-host di server yang dapat diakses dengan cepat. Gambar yang memerlukan waktu lama untuk dimuat dapat mengurangi pengalaman pengguna.'),
        ('Resolusi Gambar',
            'Pastikan bahwa gambar yang Anda tautkan ke gambar OG memiliki resolusi yang cukup tinggi untuk mendukung tampilan responsif di berbagai perangkat.'),
    ]

    speed_recom = [
        ('Optimasi Gambar', 'Pastikan gambar yang digunakan di halaman web telah dioptimasi. Gunakan format gambar yang tepat (seperti JPEG untuk gambar berwarna dan PNG untuk gambar transparan), kompres gambar jika mungkin, dan sesuaikan ukuran gambar agar sesuai dengan dimensi yang diperlukan.'),
        ('Penggunaan Lazy Loading',
            'Terapkan lazy loading untuk gambar dan elemen-elemen yang tidak perlu dimuat segera.'),
        ('Minifikasi CSS dan JavaScript',
            'Hapus spasi dan karakter yang tidak diperlukan dari berkas CSS dan JavaScript untuk mengurangi ukuran berkas dan waktu muatan.'),
        ('Pemantauan Kinerja',
            'Gunakan alat pemantauan kinerja seperti Google PageSpeed Insights atau GTmetrix untuk mengidentifikasi masalah kecepatan muatan dan mendapatkan rekomendasi khusus untuk perbaikan.'),
    ]

    # Rekomendasi untuk Heading Tags

    if result[0] == 0:
        heading_recommendations = heading_recom
    else:
        heading_recommendations = []

    if has_title == 0:
        title_recomendation = title_recom
    else:
        title_recomendation = []

    if status_desc < 3:
        desc_recomendation = desc_recom
    else:
        desc_recomendation = []

    if has_alt == 0:
        alt_recomendation = alt_recom
    else:
        alt_recomendation = []

    if has_og == 0:
        og_recomendation = og_recom
    else:
        og_recomendation = []

    if load_num < 2:
        speed_recomendation = speed_recom
    else:
        speed_recomendation = []

    return render(request, 'diagnose.html', {
        'h1': result[1],
        'heading_stats': result[0],
        'title': title,
        'title_stats': stats_title,
        'desc': desc_con,
        'status_desc': status_desc,
        'image': has_alt,
        'og': has_og,
        'time': round(load_time, 2),
        'loadstatus': load_status,
        'hasil': switch(predicted_class),
        'heading_recomen': heading_recommendations,
        'title_recomen': title_recomendation,
        'desc_recomen': desc_recomendation,
        'alt_recomen': alt_recomendation,
        'og_recomen': og_recomendation,
        'speed_recomen': speed_recomendation})
