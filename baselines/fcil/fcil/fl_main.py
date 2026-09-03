import geopandas as gpd
import fiona
import os
from pathlib import Path


def read_gdb_data(gdb_path):
    # 注册GDB驱动
    fiona.supported_drivers['FileGDB'] = 'rw'

    try:
        # 列出GDB中的所有图层
        layers = fiona.listlayers(gdb_path)
        print(f"Available layers: {layers}")

        # 创建字典存储所有图层数据
        gdb_data = {}

        # 读取每个图层
        for layer in layers:
            try:
                gdf = gpd.read_file(gdb_path, layer=layer)
                gdb_data[layer] = gdf
                print(f"Successfully read layer: {layer}")
                print(f"Shape: {gdf.shape}")
                print(f"Columns: {gdf.columns.tolist()}\n")
            except Exception as e:
                print(f"Error reading layer {layer}: {str(e)}")

        return gdb_data

    except Exception as e:
        print(f"Error accessing GDB: {str(e)}")
        return None


def analyze_gdb_data(gdb_data):
    results = {}

    for layer_name, gdf in gdb_data.items():
        # 基本统计信息
        results[layer_name] = {
            'feature_count': len(gdf),
            'geometry_type': gdf.geom_type.unique().tolist(),
            'crs': gdf.crs,
            'bounds': gdf.total_bounds.tolist()
        }

        # 如果有属性数据，计算基本统计量
        numeric_columns = gdf.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) > 0:
            results[layer_name]['statistics'] = gdf[numeric_columns].describe()

    return results


def main():
    # 设置数据路径
    gdb_path = "SLD_Trans45.gdb"

    # 检查文件是否存在
    if not os.path.exists(gdb_path):
        print(f"GDB file not found at: {gdb_path}")
        return

    # 读取数据
    print("Reading GDB data...")
    gdb_data = read_gdb_data(gdb_path)

    if gdb_data:
        # 分析数据
        print("\nAnalyzing data...")
        analysis_results = analyze_gdb_data(gdb_data)

        # 基本可视化
        for layer_name, gdf in gdb_data.items():
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                gdf.plot(ax=ax)
                plt.title(f"Layer: {layer_name}")
                plt.show()
            except Exception as e:
                print(f"Error plotting layer {layer_name}: {str(e)}")


if __name__ == "__main__":
    main()