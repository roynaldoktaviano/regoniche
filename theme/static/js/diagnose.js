let listRecomen = document.querySelectorAll('.recomen-list input[type="checkbox"]');
// Mengambil referensi tombol dengan id "reloadButton"
const reloadButton = document.getElementById('reloadButton');

// Menambahkan event listener untuk mereload halaman saat tombol ditekan
reloadButton.addEventListener('click', function() {
    location.reload();
});

listRecomen.forEach(function(checkbox) {
    checkbox.addEventListener('change', function() {
        if (this.checked) {
            // Jika checkbox dicentang, hilangkan elemen rekomendasi
            const parentRecomen = this.closest('.recomen-list');
            parentRecomen.style.display = 'none';
        }
    });
});


function showLoadingScreen() {
    document.querySelector(".loading-screen").style.display = "block";
}

// Fungsi untuk menyembunyikan loading screen
function hideLoadingScreen() {
    document.getElementById("loading-screen").style.display = "none";
}

// Panggil showLoadingScreen() saat pengguna mengirim permintaan analisis
showLoadingScreen();

// Panggil hideLoadingScreen() setelah analisis selesai
window.addEventListener('load', function () {
    hideLoadingScreen();
});


let loader = document.querySelector('.preloader')

window.addEventListener('load', function(){
    loader.style.display = "none"
})