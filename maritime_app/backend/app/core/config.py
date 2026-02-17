from pydantic import BaseModel


class Settings(BaseModel):
    test_area_name: str = "Golfo di Napoli"
    test_area_bounds: dict = {
        "lon_min": 13.90,
        "lon_max": 14.45,
        "lat_min": 40.50,
        "lat_max": 40.95,
    }
    validation_source: str = "https://data.meteo.uniparthenope.it/instruments/aisnet0/csv/aisnet_20260120Z082324.csv"
    cors_proxy: str = "https://corsproxy.io/?"
    raw_data_dir: str = "./cache/ais/raw"
    model_weights_path: str = "./cache/mstffn/mstffn_weights.pt"


settings = Settings()
