# SatNav 相机模型说明文档

## 1. 设计目标

SatNav 的相机模型旨在**模拟无人机在特定高度下的俯视相机拍摄**。系统根据智能体的位置（经纬度、高度）和相机参数（HFOV），从卫星地图数据中裁剪出对应区域的图像，以模拟无人机在该位置和高度下拍摄到的景象。

## 2. 相机模型类型

### 2.1 模型选择：简化的俯视针孔相机模型

SatNav 使用一个**简化的、垂直俯瞰的针孔相机模型**。这个模型具有以下特点：

- **俯视角度**：相机（无人机）垂直向下拍摄，模拟典型的无人机俯瞰视角
- **透视效果**：虽然使用针孔相机模型，但由于是俯视且高度相对固定，透视变形较小
- **参数化**：通过高度和视场角来控制视野范围，符合真实无人机的物理特性

### 2.2 与传统相机模型的对比

| 特性 | 传统针孔相机（如Habitat） | SatNav无人机相机模型 |
|------|-------------------------|---------------------|
| **视角** | 任意角度 | 垂直俯视 |
| **透视效果** | 明显（近大远小） | 较小（俯视+固定高度） |
| **参数** | 内参（焦距、主点）+ 外参（位置、姿态） | 高度 + HFOV + 位置（经纬度） |
| **图像来源** | 3D场景渲染 | 卫星地图裁剪 |
| **应用场景** | 室内导航 | 室外/大范围导航 |

## 3. 相机参数定义

### 3.1 内部参数 (Intrinsics)

```yaml
RGB_SENSOR:
  WIDTH: 224      # 图像宽度（像素）
  HEIGHT: 224     # 图像高度（像素）
  HFOV: 90        # 水平视场角（度）
```

**参数说明**：
- **WIDTH / HEIGHT**: 输出图像的像素尺寸
- **HFOV (Horizontal Field of View)**: 水平视场角，定义相机在水平方向上能看到的范围
  - 这是模拟真实无人机相机的重要参数
  - 典型值：60-120度

### 3.2 外部参数 (Extrinsics)

外部参数由智能体的状态决定：

```python
AgentState:
  position: [longitude, latitude, altitude]  # 经度、纬度、高度（米）
  rotation: roll  # 航向角（0-360度，0表示正北）
```

**参数说明**：
- **position**: 相机（无人机）的地理位置
  - `longitude, latitude`: 图像中心点的地理坐标
  - `altitude`: 无人机的高度（米），决定视野范围
- **rotation**: 相机的航向角（roll角度）
  - 定义图像相对于正北方向的旋转
  - 0度表示图像"上"方向为正北

## 4. 视野范围计算

### 4.1 核心公式

从**高度 (Altitude)** 和 **水平视场角 (HFOV)** 计算地面视野宽度：

\[
\text{Ground\_Width} = 2 \times \text{Altitude} \times \tan\left(\frac{\text{HFOV}}{2}\right)
\]

**Python 实现**：
```python
import math

def calculate_ground_width(altitude_meters, hfov_degrees):
    """
    计算在地面上覆盖的视野宽度
    
    Args:
        altitude_meters: 无人机高度（米）
        hfov_degrees: 水平视场角（度）
    
    Returns:
        地面视野宽度（米）
    """
    hfov_radians = math.radians(hfov_degrees)
    ground_width = 2 * altitude_meters * math.tan(hfov_radians / 2)
    return ground_width
```

### 4.2 地面采样距离 (GSD) 计算

**地面采样距离 (Ground Sample Distance, GSD)** 表示每个像素对应地面上的实际距离：

\[
\text{GSD} = \frac{\text{Ground\_Width}}{\text{WIDTH}}
\]

**Python 实现**：
```python
def calculate_gsd(altitude_meters, hfov_degrees, image_width_pixels):
    """
    计算地面采样距离（米/像素）
    
    Args:
        altitude_meters: 无人机高度（米）
        hfov_degrees: 水平视场角（度）
        image_width_pixels: 图像宽度（像素）
    
    Returns:
        GSD（米/像素）
    """
    ground_width = calculate_ground_width(altitude_meters, hfov_degrees)
    gsd = ground_width / image_width_pixels
    return gsd
```

### 4.3 计算示例

**示例配置**：
- 高度：100米
- HFOV：90度
- 图像宽度：224像素

**计算过程**：
1. 计算地面视野宽度：
   - `Ground_Width = 2 * 100 * tan(90° / 2)`
   - `Ground_Width = 2 * 100 * tan(45°)`
   - `tan(45°) = 1`
   - `Ground_Width = 200 米`

2. 计算GSD：
   - `GSD = 200 / 224 ≈ 0.89 米/像素`

3. 图像覆盖的地面范围：
   - 水平：200米
   - 垂直：如果图像高度也是224像素，且垂直视场角与水平相同，则垂直范围也是200米
   - 总覆盖面积：200米 × 200米 = 40,000平方米

## 5. 图像生成流程

### 5.1 完整流程

```
1. 获取智能体状态
   ├── position: [longitude, latitude, altitude]
   └── rotation: roll角度

2. 获取相机配置
   ├── WIDTH: 224
   ├── HEIGHT: 224
   └── HFOV: 90度

3. 计算视野范围
   ├── Ground_Width = 2 * altitude * tan(HFOV / 2)
   └── GSD = Ground_Width / WIDTH

4. 确定裁剪区域
   ├── 中心点: (longitude, latitude)
   ├── 地面范围: Ground_Width × Ground_Height (米)
   └── 旋转角度: roll

5. 从卫星地图裁剪
   └── 使用计算出的参数从地图数据源（如Google Earth API）获取图像
```

### 5.2 与Google Earth API的对接

Google Earth API或类似的地图服务通常接受以下参数：

- **`region` 或 `bounds`**: 地理区域范围（经纬度边界）
- **`scale` 或 `resolution`**: 分辨率（米/像素），对应我们计算的 **GSD**
- **`dimensions`**: 输出图像尺寸（像素），对应 `WIDTH x HEIGHT`

**对接示例**：
```python
# 1. 计算地面范围
ground_width = calculate_ground_width(altitude, hfov)
ground_height = calculate_ground_width(altitude, vfov)  # 如果有VFOV

# 2. 计算边界框（考虑roll角度）
# 以(longitude, latitude)为中心，ground_width x ground_height的矩形
# 根据roll角度旋转

# 3. 计算GSD
gsd = ground_width / width

# 4. 调用地图API
image = map_api.get_image(
    center=(longitude, latitude),
    dimensions=(width, height),
    scale=gsd,  # 或使用resolution参数
    rotation=roll
)
```

## 6. 相机模型的合理性

### 6.1 为什么使用HFOV而不是GSD？

**使用HFOV的优势**：
1. **物理意义明确**：HFOV是真实相机（无人机）的物理参数，更符合实际应用
2. **高度自适应**：当智能体高度变化时，视野范围自动调整，GSD也随之变化
3. **模拟真实场景**：真实无人机在不同高度下，视野范围确实会变化

**如果直接使用GSD**：
- 需要为每个高度单独配置GSD
- 失去了高度与视野范围的物理关联
- 不符合真实无人机的行为

### 6.2 俯视模型的合理性

**俯视模型的优势**：
1. **符合无人机应用**：大多数无人机导航任务使用俯视视角
2. **简化计算**：垂直俯视避免了复杂的3D投影计算
3. **地图数据匹配**：卫星地图本身就是俯视的，直接对应

**局限性**：
- 不适用于需要侧视或斜视的场景
- 透视变形较小，可能无法完全模拟真实相机的所有特性

## 7. 实现注意事项

### 7.1 坐标系统

- **地理坐标**：使用WGS84坐标系（经度、纬度）
- **高度**：相对于海平面的高度（米）
- **角度**：roll角度使用度数，0-360度，0表示正北

### 7.2 边界处理

当裁剪区域超出地图数据范围时：
- 返回边界内的部分
- 或使用填充（如黑色、白色、重复边缘像素）

### 7.3 性能优化

- **缓存**：对于相同位置和参数的请求，可以缓存结果
- **预加载**：可以预加载智能体可能到达的区域
- **分辨率层次**：根据高度使用不同分辨率的地图数据

## 8. 总结

SatNav 的相机模型是一个**简化的俯视针孔相机模型**，通过以下参数完整定义：

- **内部参数**：`WIDTH`, `HEIGHT`, `HFOV`
- **外部参数**：`(longitude, latitude, altitude)`, `roll`

**核心计算**：
- 从高度和HFOV计算地面视野范围
- 从视野范围和图像尺寸计算GSD
- 使用GSD和地理坐标从地图数据源获取图像

这个模型既符合真实无人机的物理特性，又能与现有的地图API（如Google Earth）无缝对接，是一个合理且实用的设计选择。

