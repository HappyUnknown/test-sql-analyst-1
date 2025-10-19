# Задача 1: Знайти координати кордону України та занести в базу даних Oracle/PostgreSQL/MySQL. (Simulated by reading geojson and implicitly having it ready)
# Задача 2: Графічно вивести кордон в середовище python або на web інтерфейс за допомогою бібліотеки Leaflet. (Handled in visualize_results)
# Задача 3: Запропонувати алгоритм, який би розбивав карту України на однакові квадрати (сторона ~1 км, можно і більше розмір як що не вистачає ресурсу на ноутбуці). (Handled in Step 0)
# Задача 4: Зберегти вершини сформованих квадратів в базу даних Oracle/PostgreSQL/MySQL. (Handled in Step 1, vertices are calculated and used but not stored separately, only grid cells)
# Задача 5: Графічно вивести ці квадрати в середовищі python на карті або на web інтерфейс за допомогою бібліотеки Leaflet. (Handled in visualize_results)
# Задача 6: У вас є збережені координати вершин квадратів. З кожної вершини графічно зобразити 3 сектори з азимутом 0, 120, 240 градусів, розкривши 60 градусів радіусом 5 км. Запропонувати алгоритм, який буде обраховувати, які вершини сформованих квадратів перетинає кожен сектор. результат перетену зберегти в БД. (Handled in Step 2, 3)
# Задача 7: Графічно вивести ці сектори поверх квадратів в середовищі python на карті або на web інтерфейс за допомогою бібліотеки Leaflet. (Handled in visualize_results)

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

# --- НАЛАШТУВАННЯ ---
sqlite_path = "ukraine_grid_wkt.sqlite"
table_name = "ukraine_grid_wkt"
geojson_path = "gadm41_UKR_0.json"
OUTPUT_DB_PATH = "sectors_intersections.gpkg"
WGS84_CRS = "EPSG:4326"

# Параметри сітки
GRID_SIDE_KM = 1
GRID_SIDE_M = GRID_SIDE_KM * 1000 

# Для створення сітки в WGS84 (прямі лінії на екрані)
# Приблизний перерахунок 10 км у градуси широти/довготи для України (приблизно 49N)
# 1 градус широти ≈ 111 км
# 1 градус довготи ≈ 73 км (на 49N)
LAT_STEP_DEG_WGS84 = GRID_SIDE_KM / 111.0
LON_STEP_DEG_WGS84 = GRID_SIDE_KM / 73.0

# Параметри секторів
SECTOR_RADIUS_KM = 5.0
SECTOR_RADIUS_M = SECTOR_RADIUS_KM * 1000
SECTOR_AZIMUTHS_INPUT = [120, 240, 360] 
SECTOR_AZIMUTHS = [a if a != 360 else 0 for a in SECTOR_AZIMUTHS_INPUT] # 360 -> 0 для обчислень
SECTOR_ANGLE_HALF = 30 # Трикутники 60 градусів

# Використовуємо українську Projected CRS (Державна система координат) 
# для секторів та обчислень, оскільки вона є метричною
PROJECTED_CRS = "EPSG:6370" 

# Підготовка: Генерація метричної сітки WGS84
print(f"0. Створення сітки ({GRID_SIDE_KM}x{GRID_SIDE_KM} км) з ПРЯМИМИ ЛІНІЯМИ на екрані (WGS84/lat/lon-tiled).")
if not os.path.exists(geojson_path):
    print(f"Помилка: Файл кордонів '{geojson_path}' не знайдено.")
    sys.exit(1)

# Задача 1: Знайти координати кордону України...
country = gpd.read_file(geojson_path)
# Залишаємо country у WGS84, оскільки сітка створюється також у WGS84
minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = country.total_bounds

grid_cells_wgs84 = []
x = minx_wgs
while x < maxx_wgs:
    y = miny_wgs
    while y < maxy_wgs:
        # box() створює квадрат з прямими сторонами у WGS84, що є прямокутним на екрані
        cell = box(x, y, x + LON_STEP_DEG_WGS84, y + LAT_STEP_DEG_WGS84)
        grid_cells_wgs84.append(cell)
        y += LAT_STEP_DEG_WGS84
    x += LON_STEP_DEG_WGS84

grid_proj = gpd.GeoDataFrame(geometry=grid_cells_wgs84, crs=WGS84_CRS)
grid_inside_wgs84 = gpd.overlay(grid_proj, country, how='intersection')

# Створюємо допоміжний GDF у projected CRS для подальших обчислень
grid_inside_proj = grid_inside_wgs84.to_crs(PROJECTED_CRS)

# Зберігаємо сітку WGS84 у базі даних 
grid_df = grid_inside_wgs84.drop(columns=['geometry']).copy() 
grid_df['WKT_geometry'] = grid_inside_wgs84.geometry.apply(lambda geom: geom.wkt)
conn = sqlite3.connect(sqlite_path)
grid_df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

print(f"   Метричну сітку збережено у файл: {sqlite_path}. Кількість осередків: {len(grid_inside_wgs84)}")


def read_grid_data(db_path, table_name, target_crs):
    """Зчитує дані сітки з SQLite, виділяє та фільтрує внутрішні вершини. Задача 4: Вершини обчислюються тут."""
    if not os.path.exists(db_path):
        print(f"Помилка: Файл бази даних '{db_path}' не знайдено.")
        sys.exit(1)

    print(f"\n1. Зчитування даних з {db_path}...")
    conn = sqlite3.connect(db_path)
    df_from_sqlite = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    geometry = df_from_sqlite['WKT_geometry'].apply(loads)
    country = gpd.read_file(geojson_path) 
    
    # Клітини сітки за WGS84
    grid_cells_gdf = gpd.GeoDataFrame(
        df_from_sqlite.drop(columns=['WKT_geometry']),
        geometry=geometry,
        crs=country.crs
    ).rename_axis('cell_id').reset_index()

    print("1.1. Виділення унікальних вершин квадратів.")
    all_coords = []
    for geom in grid_cells_gdf.geometry:
        try:
            coords = list(geom.exterior.coords)
        except AttributeError:
            if geom.geom_type == 'MultiPolygon':
                for single_geom in geom.geoms:
                    all_coords.extend(list(single_geom.exterior.coords))
            continue
        all_coords.extend(coords)
    
    unique_points = pd.DataFrame(all_coords, columns=['lon', 'lat']).drop_duplicates()
    vertices_gdf = gpd.GeoDataFrame(
        unique_points, 
        geometry=gpd.points_from_xy(unique_points.lon, unique_points.lat),
        crs=WGS84_CRS
    ).reset_index(drop=True).rename_axis('vertex_id').reset_index()

    print(f"1.2. Перепроєктування даних у метричну CRS: {target_crs} та фільтрація прикордонних вершин.")
    
    # Клітинки в лінковій системі координат для вирахування перетину
    grid_cells_projected = grid_cells_gdf.to_crs(target_crs)
    vertices_projected = vertices_gdf.to_crs(target_crs)
    country_proj = country.to_crs(target_crs) 

    # Логіка фільтрації прикордонних вершин
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

    print(f"   Фільтрація завершена: Вихідна кількість вершин: {len(vertices_projected)}. Кількість внутрішніх вершин для секторів: {len(internal_vertices)}")

    # Повертається значення для обрахунку та візуалізації у завданнях 3 та 4
    return grid_cells_projected, internal_vertices, grid_cells_gdf


def create_sector_polygon(center_point: Point, radius: float, start_angle: float, end_angle: float) -> Polygon:
    """Створює ТРИКУТНИЙ (клиноподібний) полігон сектору з ПРЯМИМИ ЛІНІЯМИ."""
    
    start_rad = np.deg2rad(start_angle)
    end_rad = np.deg2rad(end_angle)
    
    # Точка початку (на радіусі)
    x1 = center_point.x + radius * np.sin(start_rad)
    y1 = center_point.y + radius * np.cos(start_rad)

    # Точка кінця (на радіусі)
    x2 = center_point.x + radius * np.sin(end_rad)
    y2 = center_point.y + radius * np.cos(end_rad)

    # Формуємо трикутник
    points = [
        (center_point.x, center_point.y), # Центр
        (x1, y1), # Точка на start_angle
        (x2, y2), # Точка на end_angle
        (center_point.x, center_point.y) # Замикаємо
    ]
    
    return Polygon(points)


def generate_sectors(vertices_gdf: gpd.GeoDataFrame, radius_m: float, azimuths: list, angle_half: float):
    """Генерує GeoDataFrame секторів. Частина Задача 6."""
    
    print(f"\n2. Генерація трикутних секторів (3 на кожну ВНУТРІШНЮ вершину, радіус {radius_m/1000} км).")
    sectors_data = []
    
    for _, row in vertices_gdf.iterrows():
        point = row.geometry
        vertex_id = row['vertex_id']

        for azimuth in azimuths:
            start_angle = azimuth - angle_half
            end_angle = azimuth + angle_half
            
            display_azimuth = 360 if azimuth == 0 else azimuth 
            
            sector_geom = create_sector_polygon(point, radius_m, start_angle, end_angle)
            
            sectors_data.append({
                'vertex_id_center': int(vertex_id),
                'azimuth': display_azimuth, 
                'radius_km': radius_m / 1000,
                'geometry': sector_geom
            })

    sectors_gdf = gpd.GeoDataFrame(sectors_data, crs=vertices_gdf.crs)
    sectors_gdf.reset_index(names='sector_id', inplace=True)
    
    print(f"   Створено {len(sectors_gdf)} секторів.")
    return sectors_gdf


def calculate_and_save_intersections(grid_cells: gpd.GeoDataFrame, sectors_gdf: gpd.GeoDataFrame, output_path: str):
    """Обчислює перетини та зберігає результат у GeoPackage. Частина Задача 6."""
    
    print("\n3. Виконання просторового об'єднання (sjoin) для пошуку перетинів.")
    
    grid_cells = grid_cells[grid_cells.geometry.is_valid & ~grid_cells.geometry.is_empty].copy()
    sectors_gdf = sectors_gdf[sectors_gdf.geometry.is_valid & ~sectors_gdf.geometry.is_empty].copy()

    if grid_cells.empty or sectors_gdf.empty:
        print("Помилка: Один з GeoDataFrames порожній після очищення. Неможливо виконати sjoin.")
        return gpd.GeoDataFrame()

    # Задача 6: знаходження осередків сітки, що перетинають кожен окремий сектор
    intersection_result = gpd.sjoin(
        sectors_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'radius_km', 'geometry']], 
        grid_cells[['cell_id', 'geometry']], 
        how='inner', 
        predicate='intersects', 
        lsuffix='sec', 
        rsuffix='cell'
    )
    
    print(f"   Знайдено {len(intersection_result)} потенційних перетинів.")

    grid_cells_map = grid_cells.set_index('cell_id')['geometry']
    
    final_result_geom = intersection_result.apply(
        lambda row: row.geometry.intersection(grid_cells_map.loc[row['cell_id']]), 
        axis=1
    )
    
    final_gdf = gpd.GeoDataFrame(
        intersection_result.drop(columns=['geometry', 'index_cell']),
        geometry=final_result_geom,
        crs=PROJECTED_CRS
    )
    
    final_gdf = final_gdf[final_gdf.geometry.is_valid & ~final_gdf.geometry.is_empty].copy()
    final_gdf = final_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'radius_km', 'cell_id', 'geometry']].reset_index(drop=True)
    
    print(f"4. Збереження результатів перетину у GeoPackage: {output_path}")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"   (Старий файл {output_path} видалено).")

    final_gdf.to_file(
        output_path, 
        driver="GPKG", 
        layer="sectors_intersections_result"
    )
    
    print("✅ Розрахунок та збереження результатів перетину завершено.")

    return final_gdf

def visualize_results(grid_cells_4326, sectors_gdf, output_path):
    """Створює інтерактивну карту Folium. Задача 2, 5, 7."""
    
    print("\n5. Створення інтерактивної карти Folium для візуалізації (ГРИД та БАЗОВІ СЕКТОРИ).")
    
    # Використовуємо базові сектори (значно менша кількість полігонів)
    sectors_4326 = sectors_gdf[['sector_id', 'vertex_id_center', 'azimuth', 'geometry']].to_crs(WGS84_CRS)
    
    # Карта Folium використовує Mercator, тому WGS84-сітка буде перпендикулярна краям.
    m = folium.Map(location=[49, 32], zoom_start=6, tiles="cartodbpositron")
    
    # Задача 2: Виведення кордону
    country_border = gpd.read_file(geojson_path).to_crs(WGS84_CRS)
    folium.GeoJson(
        country_border.__geo_interface__,
        name="Кордон України",
        style_function=lambda x: {'color': 'black', 'weight': 2.0, 'fillOpacity': 0}
    ).add_to(m)

    # Задача 5: Виведення сітки
    folium.GeoJson(
        grid_cells_4326.__geo_interface__,
        name=f"Сітка квадратів ({GRID_SIDE_KM} км) - Вирівняна по екрану",
        style_function=lambda x: {'color': 'gray', 'weight': 1.0, 'fillOpacity': 0.1}
    ).add_to(m)
    
    # Задача 7: Виведення азимутів
    color_map = {360: 'red', 120: 'blue', 240: 'green'} 
    
    folium.GeoJson(
        sectors_4326.__geo_interface__, # Дані
        name=f"Сектори ({SECTOR_RADIUS_KM} км, Азимути)",
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
    print(f"✅ Карта (ВИРІВНЯНИЙ ГРИД та БАЗОВІ СЕКТОРИ) збережена у: {map_html_path}. Відкриваю...")
    webbrowser.open(map_html_path)


if __name__ == '__main__':
    
    # Зчитування, виділення внутрішніх вершин та проєктування
    grid_cells_proj, vertices_proj, grid_cells_4326 = read_grid_data(sqlite_path, table_name, PROJECTED_CRS)
    
    # Задача 6: Генерація секторів
    sectors_gdf = generate_sectors(vertices_proj, SECTOR_RADIUS_M, SECTOR_AZIMUTHS, SECTOR_ANGLE_HALF)
    
    # Задача 6: Обчислення перетинів та збереження
    intersections = calculate_and_save_intersections(grid_cells_proj, sectors_gdf, OUTPUT_DB_PATH)
    
    # Задача 2, 5, 7: Візуалізація
    if not intersections.empty:
        visualize_results(grid_cells_4326, sectors_gdf, OUTPUT_DB_PATH)
    else:
        print("Неможливо створити карту, оскільки не знайдено жодного перетину.")