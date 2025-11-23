import re
import os
import sys
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import argparse
import urllib.error

PCDS_PATH = os.path.join(os.path.dirname(__file__), 'pcds.csv')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'geolocalizadas.csv')

def extract_pcd_city_uf(line):
    match = re.match(r"(\d+)-([A-Z]{2})-(.+?)(?:,|$)", line.strip())
    if match:
        pcd_id, uf, city = match.groups()
        return pcd_id.strip(), f"{city.strip()}, {uf.strip()}"
    return None, None

def get_pcds_and_cities(path):
    df = pd.read_csv(
        path,
        header=None,
        names=['pcd_id', 'uf', 'city', 'extra'],
        engine='python',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines='skip'
    )
    df = df.dropna(subset=['pcd_id', 'uf', 'city'])
    return [
        (str(row['pcd_id']), f"{str(row['city']).strip()}, {str(row['uf']).strip()}")
        for _, row in df.iterrows()
    ]

def geocode_cities_serial(pcd_city_list):
    geolocator = Nominatim(user_agent="pcd_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    results = []

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['pcd_id', 'city', 'lat', 'lon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for pcd_id, city in tqdm(pcd_city_list, desc="Geocodificando cidades"):
            try:
                location = geocode(city + ", Brasil")
                if location:
                    row = {
                        'pcd_id': pcd_id,
                        'city': city,
                        'lat': location.latitude,
                        'lon': location.longitude
                    }
                    results.append(row)
                    writer.writerow(row)
                    csvfile.flush()
            except urllib.error.URLError as e:
                print(f"\nErro de rede ao geocodificar '{city}': {e.reason}")
                print("Verifique sua conexão e permissões de rede. Saindo...")
                sys.exit(1)
            except PermissionError as e:
                print(f"\nErro de permissão ao geocodificar '{city}': {e}")
                print("Pode ser bloqueio do sistema operacional para conexões de rede. Saindo...")
                sys.exit(1)
            except Exception as e:
                print(f"Erro ao geocodificar '{city}': {e}")
    return results

def plot_map(locations, output_path):
    # Usar dataset online para o Brasil
    try:
        world = gpd.read_file('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson')
        brazil = world[world['ADMIN'] == 'Brazil']
    except Exception as e:
        print(f"Erro ao carregar mapa do Brasil online: {e}")
        from shapely.geometry import box
        brazil = gpd.GeoDataFrame({'geometry': [box(-74, -34, -34, 6)]}, crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy([l['lon'] for l in locations], [l['lat'] for l in locations]),
        crs="EPSG:4326"
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    brazil.plot(ax=ax, color='white', edgecolor='black')
    gdf.plot(ax=ax, color='red', markersize=10, alpha=0.7)
    ax.set_title('Localização das PCDs no Brasil')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Mapa salvo em {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um mapa das PCDs do Brasil.")
    parser.add_argument('--offline', action='store_true',
                        help="Usar CSV salvo para geocodificação, sem acessar internet.")
    args = parser.parse_args()

    pcd_city_list = get_pcds_and_cities(PCDS_PATH)

    if args.offline:
        if not os.path.exists(CSV_PATH):
            print(f"Arquivo CSV '{CSV_PATH}' não encontrado. Rode sem --offline primeiro.")
            sys.exit(1)
        print(f"Carregando geocodificação do arquivo '{CSV_PATH}'...")
        df = pd.read_csv(CSV_PATH)
        locations = df.to_dict(orient='records')
    else:
        locations = geocode_cities_serial(pcd_city_list)

    output_file = os.path.join(os.path.dirname(__file__), 'mapa_pcds.pdf')
    plot_map(locations, output_file)
