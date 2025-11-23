import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import numpy as np
import scienceplots

plt.style.use('science')

plt.rc('font', size=16)          # tamanho da fonte geral (eixos, ticks)
plt.rc('axes', titlesize=22)     # tamanho da fonte do título dos eixos
plt.rc('axes', labelsize=22)     # tamanho da fonte dos rótulos dos eixos (xlabel, ylabel)
plt.rc('xtick', labelsize=16)    # tamanho da fonte dos ticks no eixo x
plt.rc('ytick', labelsize=16)    # tamanho da fonte dos ticks no eixo y
plt.rc('legend', fontsize=16)    # tamanho da fonte das legendas (se houver)
plt.rc('figure', titlesize=22)   # tamanho da fonte do título da figura (plt.suptitle)

# === 1. Ler os PCDs ===
df = pd.read_csv("geolocalizadas.csv", header=None, names=["id", "local", "lat", "lon"])
geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
gdf_pcds = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

# === 2. Carregar estados brasileiros ===
estados = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")
estados = estados[["name", "geometry"]].to_crs(epsg=3857)

# === 3. Atribuir cada PCD ao estado correspondente ===
pcd_com_estado = gpd.sjoin(gdf_pcds, estados, how="inner", predicate="within")

# === 4. Contar PCDs por estado ===
pcd_por_estado = pcd_com_estado.groupby("name").size().reset_index()
pcd_por_estado.columns = ["name", "pcd_count"]

# === 5. Juntar contagens com GeoDataFrame de estados ===
estados = estados.merge(pcd_por_estado, on="name", how="left").fillna(0)

# === 6. Criar categorias para legenda setorizada ===
def criar_categorias(valor):
    if valor == 0:
        return "0"
    elif valor <= 10:
        return "1-10"
    elif valor <= 20:
        return "11-20"
    elif valor <= 30:
        return "21-30"
    elif valor <= 50:
        return "31-50"
    else:
        return "50+"

estados['categoria'] = estados['pcd_count'].apply(criar_categorias)

# Definir cores do azul ao vermelho
cores = {
    "0": "#f7fbff",      # Azul muito claro
    "1-10": "#deebf7",   # Azul claro
    "11-20": "#c6dbef",  # Azul médio claro
    "21-30": "#9ecae1",  # Azul médio
    "31-50": "#6baed6",  # Azul
    "50+": "#3182bd"     # Azul escuro
}

# Mapear categorias para cores
estados['cor'] = estados['categoria'].apply(lambda x: cores[x])

# brasil = estados.unary_union
brasil = estados.geometry.union_all()

# === 7. Plotar ===
fig, ax = plt.subplots(figsize=(14, 12))

# Definir limites do mapa (um pouco maior que o Brasil)
bounds = estados.total_bounds
margin = 0.1  # 10% de margem
x_min, y_min, x_max, y_max = bounds
x_range = x_max - x_min
y_range = y_max - y_min

# Criar retângulo branco de fundo
from matplotlib.patches import Rectangle
background = Rectangle(
    (x_min - x_range * margin, y_min - y_range * margin),
    x_range * (1 + 2 * margin),
    y_range * (1 + 2 * margin),
    facecolor='white',
    edgecolor='none',
    zorder=0
)
ax.add_patch(background)

# Plotar choropleth dos estados
estados.plot(
    ax=ax,
    color=estados['cor'],
    edgecolor="black",
    linewidth=0.8,
    zorder=1
)

# Plotar pontos de PCD
gdf_pcds.plot(
    ax=ax,
    color='red',
    markersize=8,
    alpha=0.7,
    label="PCDs",
    zorder=2
)

# Ajustar limites para o polígono do Brasil com margem pequena
x_min, y_min, x_max, y_max = brasil.bounds
margin = 0.05
x_range = x_max - x_min
y_range = y_max - y_min
ax.set_xlim(x_min - x_range * margin, x_max + x_range * margin)
ax.set_ylim(y_min - y_range * margin, y_max + y_range * margin)

# Adicionar nomes dos estados (agora usando siglas)
SIGLAS = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA", "Ceará": "CE",
    "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA", "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR",
    "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC",
    "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"
}

for _, row in estados.iterrows():
    centroid = row.geometry.centroid
    nome_estado = str(row["name"])
    sigla = SIGLAS.get(nome_estado, nome_estado)
    ax.text(centroid.x, centroid.y, sigla, fontsize=7, ha='center',
            color='darkblue', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8,
                      edgecolor='gray', linewidth=0.5),
            zorder=3)

# Definir limites do plot
ax.set_xlim(x_min - x_range * margin, x_max + x_range * margin)
ax.set_ylim(y_min - y_range * margin, y_max + y_range * margin)

# Base map
try:
    ctx.add_basemap(ax, clip_path=brasil, clip_on=True)
except:
    print("Base map não disponível, usando mapa simples")


# Criar legenda customizada
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cores["0"], edgecolor='black', label='0 PCDs'),
    Patch(facecolor=cores["1-10"], edgecolor='black', label='1-10 PCDs'),
    Patch(facecolor=cores["11-20"], edgecolor='black', label='11-20 PCDs'),
    Patch(facecolor=cores["21-30"], edgecolor='black', label='21-30 PCDs'),
    Patch(facecolor=cores["31-50"], edgecolor='black', label='31-50 PCDs'),
    Patch(facecolor=cores["50+"], edgecolor='black', label='50+ PCDs'),
]

ax.legend(
        handles=legend_elements,
        title='Quantidade de PCDs por Estado',
        loc='upper right',
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=12,
        framealpha=0.9,
        fancybox=True
    )

# Título e ajustes
# ax.set_title("Distribuição de PCDs por Estado no Brasil", fontsize=16, weight='bold', pad=20)
ax.axis("off")

plt.tight_layout()

# === 8. Salvar como PDF ===
plt.savefig("heatmap.pdf", dpi=1500, bbox_inches='tight')
print("Mapa salvo como choropleth_pcds_com_pontos.pdf")

# Mostrar estatísticas
print("\nEstatísticas por categoria:")
for categoria in sorted(cores.keys()):
    count = len(estados[estados['categoria'] == categoria])
    print(f"   {categoria}: {count} estados")
