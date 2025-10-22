import geopandas as gpd
import pandas as pd
import sqlite3
from shapely.wkt import loads
from shapely.geometry import box, Point, Polygon
import numpy as np
import folium
import webbrowser
import os
import sys

# --- КОНФІГУРАЦІЯ ТА ВИМОГИ ЗАВДАННЯ ---
sqlite_path = "ukraine_grid_wkt.sqlite"
table_name = "ukraine_grid_wkt"
geojson_path = "gadm41_UKR_0.json"
OUTPUT_DB_PATH = "sectors_intersections.gpkg"
WGS84_CRS = "EPSG:4326"

# Параметри сітки
GRID_SIDE_KM = 1
GRID_SIDE_M = GRID_SIDE_KM * 1000 

# Параметри секторів
SECTOR_RADIUS_KM = 5
SECTOR_RADIUS_M = SECTOR_RADIUS_KM * 1000 # 50 км
SECTOR_AZIMUTHS_INPUT = [0, 120, 240] 

# Розкриття: 120 градусів (±60 градусів від осі) для повного кола
SECTOR_ANGLE_HALF = 60 
SEGMENTS = 30 # Кількість сегментів для апроксимації дуги

PROJECTED_CRS = "EPSG:32635" # UTM Zone 35N для метричних обчислень

# --- ПІДГОТОВКА: Створення сітки та збереження у SQLite ---
print(f"0. Створення сітки ({GRID_SIDE_KM}x{GRID_SIDE_KM} км) з ПРЯМИМИ ЛІНІЯМИ (WGS84/lat/lon-tiled).")
if not os.path.exists(geojson_path):
    print(f"Помилка: Файл кордонів '{geojson_path}' не знайдено.")
    sys.exit(1)

country = gpd.read_file(geojson_path)
minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = country.total_bounds

LAT_STEP_DEG_WGS84 = GRID_SIDE_KM / 111.0
LON_STEP_DEG_WGS84 = GRID_SIDE_KM / 73.0

grid_cells_wgs84 = []
x = minx_wgs
while x < maxx_wgs:
    y = miny_wgs
    while y < maxy_wgs:
        cell = box(x, y, x + LON_STEP_DEG_WGS84, y + LAT_STEP_DEG_WGS84)
        grid_cells_wgs84.append(cell)
        y += LAT_STEP_DEG_WGS84
    x += LON_STEP_DEG_WGS84

grid_proj = gpd.GeoDataFrame(geometry=grid_cells_wgs84, crs=WGS84_CRS)
grid_inside_wgs84 = gpd.overlay(grid_proj, country, how='intersection')

grid_df = grid_inside_wgs84.drop(columns=['geometry']).copy() 
grid_df['WKT_geometry'] = grid_inside_wgs84.geometry.apply(lambda geom: geom.wkt)
conn = sqlite3.connect(sqlite_path)
grid_df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

print(f"   Сітку ({GRID_SIDE_KM}x{GRID_SIDE_KM} км) збережено у файл: {sqlite_path}. Осередків: {len(grid_inside_wgs84)}")

def read_grid_data(db_path, table_name, target_crs):
    """Зчитує дані сітки з SQLite та виділяє внутрішні вершини."""
    if not os.path.exists(db_path):
        print(f"Помилка: Файл бази даних '{db_path}' не знайдено.")
        sys.exit(1)

    print(f"\n1. Зчитування даних з {db_path}...")
    conn = sqlite3.connect(db_path)
    df_from_sqlite = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    geometry = df_from_sqlite['WKT_geometry'].apply(loads)
    country = gpd.read_file(geojson_path) 
    
    grid_cells_gdf = gpd.GeoDataFrame(
        df_from_sqlite.drop(columns=['WKT_geometry']),
        geometry=geometry,
        crs=country.crs
    ).rename_axis('cell_id').reset_index()

    print("1.1. Виділення унікальних вершин квадратів.")
    all_coords = []
    for geom in grid_cells_gdf.geometry:
        if geom.geom_type in ['Polygon']:
            all_coords.extend(list(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for single_geom in geom.geoms:
                all_coords.extend(list(single_geom.exterior.coords))
    
    unique_points = pd.DataFrame(all_coords, columns=['lon', 'lat']).drop_duplicates()
    vertices_gdf = gpd.GeoDataFrame(
        unique_points, 
        geometry=gpd.points_from_xy(unique_points.lon, unique_points.lat),
        crs=WGS84_CRS
    ).reset_index(drop=True).rename_axis('vertex_id').reset_index()

    print(f"1.2. Перепроєктування даних у метричну CRS: {target_crs} та фільтрація внутрішніх вершин.")
    
    vertices_projected = vertices_gdf.to_crs(target_crs)
    country_proj = country.to_crs(target_crs) 

    internal_buffer_m = -100 
    internal_country_area = country_proj.geometry.buffer(internal_buffer_m)
    
    internal_country_gdf = gpd.GeoDataFrame(
        geometry=internal_country_area.explode(ignore_index=True), 
        crs=target_crs
    )

    internal_vertices = gpd.sjoin(
        vertices_projected, 
        internal_country_gdf, 
        how='inner', 
        predicate='within'
    )
    internal_vertices = internal_vertices[vertices_projected.columns].copy()

    print(f"   Фільтрація завершена: Вихідна кількість вершин: {len(vertices_projected)}. Внутрішніх: {len(internal_vertices)}")

    grid_cells_projected = grid_cells_gdf.to_crs(target_crs)
    return grid_cells_projected, internal_vertices, grid_cells_gdf

# --- ФУНКЦІЯ СТВОРЕННЯ СЕКТОРА (ДУГА КОЛА) ---

def create_sector_polygon(center_point: Point, radius: float, start_angle: float, end_angle: float, segments: int) -> Polygon:
    """Створює ГЕОМЕТРІЮ СЕКТОРА (клину) з дугою кола."""
    
    # Конвертуємо азимут (0=північ) у стандартний математичний кут (0=схід)
    start_angle_math = 90 - start_angle
    end_angle_math = 90 - end_angle

    # Визначаємо кути для генерації дуги.
    angles_range = np.linspace(end_angle_math, start_angle_math, num=segments)

    # Конвертуємо в радіани
    angles_rad = np.deg2rad(angles_range)
    
    # Обчислення координат точок на дузі
    x_coords = center_point.x + radius * np.cos(angles_rad)
    y_coords = center_point.y + radius * np.sin(angles_rad)
    
    # Формування фінального полігону: Центр -> Точки на дузі -> Замикання на Центр
    points = [(center_point.x, center_point.y)] + list(zip(x_coords, y_coords)) + [(center_point.x, center_point.y)]
    
    return Polygon(points)


def generate_sectors(vertices_gdf: gpd.GeoDataFrame, radius_m: float, azimuths: list, angle_half: float):
    """Генерує GeoDataFrame секторів."""
    
    print(f"\n2. Генерація секторів кола (3 на кожну ВНУТРІШНЮ вершину, радіус {radius_m/1000} км).")
    sectors_data = []
    
    for _, row in vertices_gdf.iterrows():
        point = row.geometry
        vertex_id = row['vertex_id']

        for azimuth in azimuths:
            start_angle = azimuth - angle_half
            end_angle = azimuth + angle_half
            
            sector_geom = create_sector_polygon(point, radius_m, start_angle, end_angle, SEGMENTS)
            
            sectors_data.append({
                'vertex_id_center': int(vertex_id),
                'azimuth': azimuth, 
                'radius_km': radius_m / 1000,
                'geometry': sector_geom
            })

    sectors_gdf = gpd.GeoDataFrame(sectors_data, crs=vertices_gdf.crs)
    sectors_gdf.reset_index(names='sector_id', inplace=True)
    
    print(f"   Створено {len(sectors_gdf)} секторів.")
    return sectors_gdf


# --- АЛГОРИТМ ОБЧИСЛЕННЯ ПЕРЕТИНІВ (ОНОВЛЕНО ДЛЯ ВИВОДУ В КОНСОЛЬ) ---

def calculate_and_save_intersections(grid_cells: gpd.GeoDataFrame, sectors_gdf: gpd.GeoDataFrame, output_path: str):
    """Обчислює перетини, зберігає результат у GeoPackage та виводить у консоль."""
    
    print("\n3. Виконання просторового об'єднання (sjoin) для пошуку перетинів.")
    
    grid_cells = grid_cells[grid_cells.geometry.is_valid & ~grid_cells.geometry.is_empty].copy()
    sectors_gdf = sectors_gdf[sectors_gdf.geometry.is_valid & ~sectors_gdf.geometry.is_empty].copy()

    if grid_cells.empty or sectors_gdf.empty:
        print("Помилка: Один з GeoDataFrames порожній після очищення. Неможливо виконати sjoin.")
        return gpd.GeoDataFrame()

    # 1. Знаходження потенційних перетинів
    intersection_result = gpd.sjoin(
        sectors_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'radius_km', 'geometry']], 
        grid_cells[['cell_id', 'geometry']], 
        how='inner', 
        predicate='intersects', 
        lsuffix='sec', 
        rsuffix='cell'
    )
    
    print(f"   Знайдено {len(intersection_result)} потенційних перетинів.")

    grid_cells_map = grid_cells.set_index('cell_id')['geometry']
    
    # 2. Обчислення фактичного полігону перетину (intersection)
    final_result_geom = intersection_result.apply(
        lambda row: row.geometry.intersection(grid_cells_map.loc[row['cell_id']]), 
        axis=1
    )
    
    final_gdf = gpd.GeoDataFrame(
        intersection_result.drop(columns=['geometry', 'index_cell']),
        geometry=final_result_geom,
        crs=PROJECTED_CRS
    )
    
    # Фінальна очистка
    final_gdf = final_gdf[final_gdf.geometry.is_valid & ~final_gdf.geometry.is_empty].copy()
    final_gdf = final_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'radius_km', 'cell_id', 'geometry']].reset_index(drop=True)
    
    print(f"   Знайдено {len(final_gdf)} фінальних, дійсних перетинів (полігонів).")

    # ВИВІД ДАНИХ ДО КОНСОЛІ (Вимога)
    print("\n[КОНСОЛЬНИЙ ВИВІД РЕЗУЛЬТАТІВ ПЕРЕТИНУ]")
    print("---------------------------------------------------------")
    # Виводимо перші 5 рядків без геометрії (WKT) для кращої читабельності
    print(final_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'cell_id']].head())
    print(f"\n... (Всього фінальних перетинів: {len(final_gdf)})")
    print("---------------------------------------------------------")
    
    # 3. Збереження результатів у GeoPackage
    print(f"4. Збереження результатів перетину у GeoPackage: {output_path}")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"   (Старий файл {output_path} видалено).")

    final_gdf.to_file(
        output_path, 
        driver="GPKG", 
        layer="sectors_intersections_result"
    )
    
    print("✅ Розрахунок та збереження результатів перетину завершено.")

    return final_gdf

# --- ФУНКЦІЯ ВІЗУАЛІЗАЦІЇ (БЕЗ ЗМІН) ---

def visualize_results(grid_cells_4326, sectors_gdf, output_path):
    """Створює інтерактивну карту Folium."""
    
    print("\n5. Створення інтерактивної карти Folium для візуалізації.")
    
    sectors_4326 = sectors_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'geometry']].to_crs(WGS84_CRS)
    
    m = folium.Map(location=[49, 32], zoom_start=6, tiles="cartodbpositron")
    
    country_border = gpd.read_file(geojson_path).to_crs(WGS84_CRS)
    folium.GeoJson(
        country_border.__geo_interface__,
        name="Кордон України",
        style_function=lambda x: {'color': 'black', 'weight': 2.0, 'fillOpacity': 0}
    ).add_to(m)

    folium.GeoJson(
        grid_cells_4326.__geo_interface__,
        name=f"Сітка квадратів ({GRID_SIDE_KM} км)",
        style_function=lambda x: {'color': 'gray', 'weight': 1.0, 'fillOpacity': 0.1}
    ).add_to(m)
    
    color_map = {0: 'red', 120: 'blue', 240: 'green'} 
    
    folium.GeoJson(
        sectors_4326.__geo_interface__,
        name=f"Сектори ({SECTOR_RADIUS_M/1000} км, Азимути)",
        style_function=lambda feature: {
            'fillColor': color_map.get(feature['properties']['azimuth'], 'purple'),
            'color': color_map.get(feature['properties']['azimuth'], 'purple'),
            'weight': 1.0,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=['azimuth', 'vertex_id_center'], aliases=['Азимут', 'Центр ID'])
    ).add_to(m)

    folium.LayerControl().add_to(m)

    map_html_path = output_path.replace('.gpkg', '_map.html')
    m.save(map_html_path)
    print(f"✅ Карта збережена у: {map_html_path}. Відкриваю...")
    webbrowser.open(map_html_path)


if __name__ == '__main__':
    
    # 1. Зчитування, виділення внутрішніх вершин та проєктування
    grid_cells_proj, vertices_proj, grid_cells_4326 = read_grid_data(sqlite_path, table_name, PROJECTED_CRS)
    
    # 2. Генерація секторів
    sectors_gdf = generate_sectors(vertices_proj, SECTOR_RADIUS_M, SECTOR_AZIMUTHS_INPUT, SECTOR_ANGLE_HALF)
    
    # 3. Обчислення перетинів та збереження
    intersections = calculate_and_save_intersections(grid_cells_proj, sectors_gdf, OUTPUT_DB_PATH)
    
    # 4. Візуалізація
    if not intersections.empty:
        visualize_results(grid_cells_4326, sectors_gdf, OUTPUT_DB_PATH)
    else:
        print("Неможливо створити карту, оскільки не знайдено жодного перетину.")