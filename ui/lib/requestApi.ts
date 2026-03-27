export interface PredictionResponse {
    status: string;
    data: {
        hazard_level: number;
        lat: number;
        lng: number;
    };
    // message?: string; // Untuk menangani pesan error jika ada
}

export const fetchHazardPrediction = async (lat: number, lng: number): Promise<PredictionResponse | null> => {
    try {
        // Sesuaikan URL dengan port Flask kamu (default 5000)
        
        // Sementara untuk LOCALAN
        const url = `http://localhost:5000/predict?lat=${lat}&lng=${lng}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'x-api-key': 'jangkrik97' // Sesuaikan dengan API key yang kamu gunakan di Flask
            },
            // Penting: Jika kamu mendeploy ini, ganti localhost dengan IP server
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: PredictionResponse = await response.json();
        console.log("Prediksi berhasil diterima:", result);
        return result;
    } catch (error) {
        console.error("Gagal mengambil prediksi dari Flask:", error);
        return null;
    }
};