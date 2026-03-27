'use client';
import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Button } from '@/components/ui/button';

// Interface untuk State
interface Location {
  lat: number;
  lng: number;
}

// Load Map tanpa SSR (Client Side Only)
const MapWithNoSSR = dynamic(() => import('../components/MapComponents'), {
  ssr: false,
  loading: () => <div className=" h-full w-full flex items-center justify-center bg-gray-100 text-gray-500">Memuat Peta...</div>
});

export default function Home() {
  const [currentLocation, setCurrentLocation] = useState<Location | null>(null);
  const [clickedLocation, setClickedLocation] = useState<Location | null>(null);
  const [currentAddress, setCurrentAddress] = useState<string>("");
  const [clickedAddress, setClickedAddress] = useState<string>("");

  // Ambil lokasi dari browser
  useEffect(() => {
    if (typeof window !== "undefined" && "geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setCurrentLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          console.error("Gagal mendapatkan lokasi:", error);
        }
      );
    }
  }, []);

  const getAddress = async (lat: number, lng: number) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lng}`,
        {
          headers: {
            "Accept-Language": "id", // Agar hasil dalam Bahasa Indonesia
          }
        }
      );
      const data = await response.json();
      console.log("Data reverse geocode:", data);
       // Simpan alamat lengkap ke state
      return data.display_name; // Ini adalah nama lokasi lengkapnya
    } catch (error) {
      console.error("Gagal reverse geocode:", error);
      return "Lokasi tidak diketahui";
    }
  };

  useEffect(() => {
    if (clickedLocation) {
      getAddress(clickedLocation.lat, clickedLocation.lng).then((address) => {
        console.log("Alamat lokasi yang diklik:", address);
        setClickedAddress(address); // Simpan alamat lokasi yang diklik ke state
      });
    }
  }, [clickedLocation]);
  useEffect(() => {
    if (currentLocation) {
      getAddress(currentLocation.lat, currentLocation.lng).then((address) => {
        console.log("Alamat lokasi saat ini:", address);
        setCurrentAddress(address);
      });
    }
  }, [currentLocation]);

  const handleScrollToMain = () => {
    const mainSection = document.getElementById('main');
    if (mainSection) {
      mainSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="flex flex-col w-full overflow-hidden bg-gray-100">
      {/* Header */}
      {/* <header className=" text-white p-10 z-10">
        <h1 className="pl-20 text-xl font-bold tracking-tight"></h1>
      </header> */}

      {/* jumbotron */}
      <div className="text-white text-center relative">
        <img src="./images/earthquake-pictures.jpg" alt="..." className="absolute z-1 w-full h-full object-cover " />
        <div className="after absolute z-2 w-full h-full bg-linear-to-tr from-[rgba(0,0,0,0.8)] to-[rgba(0,0,0,0.1)]"></div>
        <div className="relative z-2 pt-32 px-4 pb-10 font-(Inter)">
          <h2 className="text-4xl font-bold mb-2 text-shadow-lg pb-3">Selamat Datang di <span className='text-amber-300'>Waspada</span>.Gempa</h2>
          <p className="text-2xl font-light text-shadow-lg pb-10">Pantau potensi bencana gempa di lingkungan sekitar kita<br />dengan mudah dan cepat.</p>

          <div className="flex px-20 gap-10 justify-center pb-10">
            <div className="flex-1 max-w-120 p-4">
              <h3 className="text-xl font-bold mb-2 text-shadow-lg">⚠️ Potensi Tingkat Bahaya Dan Dampak Gempa</h3>
              <p className="text-md font-light text-shadow-lg text-justify">
                Kita dapat dengan mudah memprediksi potensi dampak gempa di lingkungan sekitar berdasarkan data kejadian gempa sebelumnya. Dengan memanfaatkan data historis dan model prediktif, kita dapat memberikan informasi yang berguna untuk meningkatkan kewaspadaan dan kesiapsiagaan terhadap potensi bencana gempa.
              </p>
            </div>
            <div className="flex-1 max-w-120 p-4">
              <h3 className="text-xl font-bold mb-2 text-shadow-lg">📈🎯 Prediksi Persentase Kejadian Gempa Mendatang</h3>
              <p className="text-md font-light text-shadow-lg text-left">
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Sequi cumque eligendi iste nemo dolores aperiam eveniet optio obcaecati quibusdam dolorem. Lorem ipsum dolor sit amet consectetur, adipisicing elit. Adipisci, voluptatum.
              </p>
            </div>
          </div>
          {/* <div className="btn-double-effect bg-amber-200"></div> */}
          <button onClick={handleScrollToMain} className="relative my-10 px-4 py-2 text-lg bg-black font-bold rounded-3xl hover:bg-linear-to-r from-[#ef9917] to-[#e2eb60] cursor-pointer hover:text-gray-700 transition-colors
          after:content-['']
          after:absolute 
          after:top-1/2 
          after:left-1/2
          after:translate-x-[-50%]
          after:translate-y-[-50%]
          after:w-full
          after:h-full
          after:z-[-1]
          after:box-content
          after:p-0.5
          after:rounded-[inherit]
          after:bg-[conic-gradient(from_var(--angle),transparent_70%,#fafafa)]
          after:blur-[10px]
          after:opacity-[0.6]

          before:content-['']
          before:absolute 
          before:top-1/2 
          before:left-1/2
          before:translate-x-[-50%]
          before:translate-y-[-50%]
          before:w-full
          before:h-full
          before:z-[-1]
          before:box-content
          before:p-0.5
          before:rounded-[inherit]
          before:bg-[conic-gradient(from_var(--angle),transparent_70%,#fafafa)]
          
          after:animate-spin-slow
          before:animate-spin-slow
          ">Coba Sekarang</button>
        </div>
      </div>

      {/* Main Content */}

      <main id='main' className="pt-10 flex flex-1 relative px-40 gap-5 pb-5">
        {/* Peta (Kiri) */}
        <div className="flex-3 z-0 bg-slate-500 rounded-lg h-125">
          <MapWithNoSSR
            currentLocation={currentLocation}
            clickedLocation={clickedLocation}
            setClickedLocation={setClickedLocation}
          />
        </div>

        {/* Info Panel (Kanan) */}
        <aside className="w-80 bg-white z-20 p-6 overflow-y-auto rounded-lg">
          <h1 className='text-lg mb-3 font-semibold'>Petunjuk Penggunaan</h1>
          <ol className="pl-5 list-decimal list-outside" type='1'>
            <li className="text-sm text-gray-600 mb-2">Klik pada peta untuk memilih lokasi yang ingin Anda analisis dengan mengeklik suatu wilayah di peta.</li>
            <li className="text-sm text-gray-600 mb-2">Setelah memilih lokasi, dilakan klik tombol orange di bawah untuk memulai analisis lokasi. Sistem akan memproses data historis dan model prediktif untuk memberikan hasil analisis potensi dampak gempa .....</li>
            <li className="text-sm text-gray-600 mb-2">Hasil analisis akan ditampilkan di bagian bawah halaman, termasuk tingkat prediksi dampak gempa dan data historis terkait......</li>
            <li className="text-sm text-gray-600 mb-2">Gunakan informasi ini untuk meningkatkan kewaspadaan dan kesiapsiagaan terhadap potensi bencana gempa di lingkungan sekitar Anda.</li>
          </ol>
        </aside>
      </main>
      {/* aksi tombol */}
      <div className=" px-40 mb-10">
        <div className="mb-4">
          <h1 className="text-gray-500">📍<span className='font-semibold text-black'>Lokasi saat ini:</span> {currentAddress}.</h1>
          <h1 className="text-gray-500">📍<span className='font-semibold text-black'>Lokasi yang dipilih:</span> {clickedAddress == "" ? "Belum ada yang diklik." : clickedAddress}</h1>
        </div>
        <div className="flex justify-center gap-5">
          <button className="isolate relative text-md rounded-2xl font-semibold
          " onClick={() => { }}>
            <div className="bg-amber-400 z-10 py-2 px-4 rounded-[inherit]">
              Mulai Analisis Lokasi
            </div>
          </button>
          <button className='border border-black text-md rounded-2xl py-2 px-4' onClick={() => setClickedLocation(null)}>Reset Lokasi</button>
        </div>
      </div>
      {/* hasil */}
      <div className="px-40 mb-20">
        <h2 className="text-3xl font-bold mb-4 text-center">Hasil Analisis Lokasi</h2>
        <div className="flex justify-center">
          <div className="flex-1 p-6 bg-white rounded-lg ">
            <h3 className="text-xl font-semibold mb-2">📊 Potensi Dampak Gempa Dari Kejadian Lampau</h3>
            <p className="text-sm mb-4">Berdasarkan data historis dan model prediktif, berikut adalah potensi dampak bencana untuk lokasi yang Anda pilih:</p>

            <p className="text-md mb-4"><b>Hasil:</b> Lokasi tergolong area dengan potensial dampak gempa .... </p>

            <p className="text-sm mb-1">Lokasi cukup dekat dengan zona gempa .... dataset grid. Dengan hasil analisis data sebagai berikut:</p>


            <ul className="list-disc list-inside text-gray-600">
              <li>Frekuensi terjadinya gempa di area sekitar:....</li>
              <li>Magnitudo terbesar yang pernah terjadi: .....</li>
              <li>Rata-rata magnitudo: ....</li>
              <li>Rata-rata kedalaman sumber gempa: .....</li>
            </ul>
            <p className="text-xs text-gray-500 mt-4">*Tingkatan prediksi dampak gempa dihitung berdasarkan kombinasi data historis dan model prediktif <i>Machine Learning.</i></p>
          </div>
          <div className="flex-1 p-6 bg-white rounded-lg ml-5">
            <p className="">Lorem, ipsum dolor sit amet consectetur adipisicing elit. Nemo nihil blanditiis labore doloribus iusto iure accusantium dolores alias tenetur et.</p>
          </div>
        </div>
      </div>
      <div className="text-center text-sm text-gray-500 mb-5">
        <h1>copyright &copy; 2026 Waspada.Gempa. All rights reserved.</h1>

      </div>
    </div>
  );
}