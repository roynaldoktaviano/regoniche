let loader = document.querySelector('.preloader')
let buttonScan = document.querySelector('.scan')

buttonScan.addEventListener('click', function(){
    loader.classList.toggle('hidden')
})